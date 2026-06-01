//===----------------------------------------------------------------------===//
// KSPackPass — Pack B operand of linalg.matmul for cache-efficient access
//
// Milestone: M4 (RISC-V RVV Target)
// Design: DES-009 (docs/design/DES-009-m4-rvv-lowering.md)
//
// Packing strategy:
//   Before packing, B has layout [K, N] — accessing N-columns of B in the
//   inner loop causes strided reads with poor cache utilization.
//
//   After packing, B has layout [N/NR, K, NR] — the NR inner columns are
//   contiguous in memory, enabling stride-free vector loads in the
//   micro-kernel.
//
//   linalg.matmul ins(%A[M,K], %B[K,N]) outs(%C[M,N])
//   ──►
//   %Bp = linalg.pack %B inner_dims_pos=[1] inner_tiles=[NR]
//           : tensor<KxNxf32> -> tensor<Nx/NRxKxNRxf32>
//   %Cp = linalg.pack %C inner_dims_pos=[1] inner_tiles=[NR]
//           : tensor<MxNxf32> -> tensor<N/NRxMxNRxf32>
//   linalg.generic {packed matmul} ins(%A, %Bp) outs(%Cp)
//   %C_out = linalg.unpack %Cp ...
//
// This pass uses linalg::pack (the transform builder) from MLIR.
// The packed op is expressed as a linalg.generic with explicit indexing maps
// because linalg.matmul does not accept packed operands directly.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSPACKPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Packing pattern
//===----------------------------------------------------------------------===//

/// Pack linalg.matmul B operand into column-panel layout.
///
/// The packed layout for B is [N/NR, K, NR]:
///   - Outer dim 0: N panels of width NR
///   - Dim 1: K reduction axis
///   - Inner dim 2: NR contiguous elements (stride-free load)
///
/// This matches the GEBP (General-to-Blocked Panel) micro-kernel layout
/// used by high-performance BLAS implementations (BLIS, OpenBLAS, etc.).
struct PackMatmulBPattern : public OpRewritePattern<linalg::MatmulOp> {
  PackMatmulBPattern(MLIRContext *ctx, int64_t packFactor)
      : OpRewritePattern(ctx), packFactor(packFactor) {}

  LogicalResult matchAndRewrite(linalg::MatmulOp matmul,
                                PatternRewriter &rewriter) const override {
    if (packFactor <= 0)
      return failure();

    Location loc = matmul.getLoc();
    Value A = matmul.getOperand(0); // [M, K]
    Value B = matmul.getOperand(1); // [K, N]
    Value C = matmul.getOperand(2); // [M, N] output/init

    auto bType = dyn_cast<RankedTensorType>(B.getType());
    auto cType = dyn_cast<RankedTensorType>(C.getType());
    if (!bType || !cType)
      return failure();

    // Only handle static shapes for now.
    if (!bType.hasStaticShape() || !cType.hasStaticShape())
      return rewriter.notifyMatchFailure(matmul,
                                         "dynamic shapes not yet supported");

    int64_t K = bType.getDimSize(0);
    int64_t N = bType.getDimSize(1);

    // Require N to be divisible by pack factor for clean tiling.
    // (Tail handling is left to a future --ks-pad pass.)
    if (N % packFactor != 0)
      return rewriter.notifyMatchFailure(
          matmul, "N not divisible by pack-factor; skipping");

    int64_t numPanels = N / packFactor;

    // === Pack B: [K, N] -> [numPanels, K, NR] ===
    // inner_dims_pos=[1] means we tile dim-1 (N) with tiles of size NR.
    auto packedBType = RankedTensorType::get({numPanels, K, packFactor},
                                             bType.getElementType());
    SmallVector<OpFoldResult> innerTiles{rewriter.getIndexAttr(packFactor)};
    SmallVector<int64_t> outerDimsPerm{1, 0};
    Value packedBInit = rewriter.create<tensor::EmptyOp>(
        loc, packedBType.getShape(), packedBType.getElementType());
    auto packB =
        rewriter.create<linalg::PackOp>(loc, B, packedBInit,
                                        /*innerDimsPos=*/ArrayRef<int64_t>{1},
                                        /*innerTiles=*/innerTiles,
                                        /*paddingValue=*/std::optional<Value>{},
                                        /*outerDimsPerm=*/outerDimsPerm);

    // === Pack C: [M, N] -> [numPanels, M, NR] ===
    int64_t M = cType.getDimSize(0);
    auto packedCType = RankedTensorType::get({numPanels, M, packFactor},
                                             cType.getElementType());
    Value packedCInit = rewriter.create<tensor::EmptyOp>(
        loc, packedCType.getShape(), packedCType.getElementType());
    auto packC =
        rewriter.create<linalg::PackOp>(loc, C, packedCInit,
                                        /*innerDimsPos=*/ArrayRef<int64_t>{1},
                                        /*innerTiles=*/innerTiles,
                                        /*paddingValue=*/std::optional<Value>{},
                                        /*outerDimsPerm=*/outerDimsPerm);

    // === Packed matmul as linalg.generic ===
    //
    // Indexing maps for packed GEMM with B in [numPanels, K, NR] layout:
    //   A[m, k]          -> (n_panel, m, k, nr) -> A[m, k]
    //   B[n_panel, k, nr]-> (n_panel, m, k, nr) -> B[n_panel, k, nr]
    //   C[n_panel, m, nr]-> (n_panel, m, k, nr) -> C[n_panel, m, nr]
    //
    // Iteration domain: (n_panel, m, k, nr)
    //   n_panel: [0, numPanels)  parallel
    //   m:       [0, M)          parallel
    //   k:       [0, K)          reduction
    //   nr:      [0, NR)         parallel (vectorized)

    MLIRContext *ctx = rewriter.getContext();
    AffineExpr dNp = rewriter.getAffineDimExpr(0); // n_panel
    AffineExpr dM = rewriter.getAffineDimExpr(1);  // m
    AffineExpr dK = rewriter.getAffineDimExpr(2);  // k
    AffineExpr dNr = rewriter.getAffineDimExpr(3); // nr

    // A[m, k]
    auto mapA = AffineMap::get(4, 0, {dM, dK}, ctx);
    // B[n_panel, k, nr]
    auto mapB = AffineMap::get(4, 0, {dNp, dK, dNr}, ctx);
    // C[n_panel, m, nr]
    auto mapC = AffineMap::get(4, 0, {dNp, dM, dNr}, ctx);

    auto parallel = utils::IteratorType::parallel;
    auto reduction = utils::IteratorType::reduction;
    SmallVector<utils::IteratorType> iters = {parallel, parallel, reduction,
                                              parallel};

    auto packedGeneric = rewriter.create<linalg::GenericOp>(
        loc,
        /*resultTypes=*/TypeRange{packedCType},
        /*inputs=*/ValueRange{A, packB.getResult(), packC.getResult()},
        /*outputs=*/ValueRange{packC.getResult()},
        /*indexingMaps=*/ArrayRef<AffineMap>{mapA, mapB, mapC, mapC},
        /*iteratorTypes=*/iters,
        /*doc=*/"packed matmul: C[np,m,nr] += A[m,k] * B[np,k,nr]",
        /*libraryCall=*/"");

    Region &region = packedGeneric.getRegion();
    Block *body = nullptr;
    if (region.empty()) {
      body =
          rewriter.createBlock(&region, region.end(),
                               {bType.getElementType(), bType.getElementType(),
                                cType.getElementType(), cType.getElementType()},
                               {loc, loc, loc, loc});
    } else {
      body = &region.front();
      while (!body->empty())
        body->back().erase();
      if (body->getNumArguments() == 0) {
        body->addArguments({bType.getElementType(), bType.getElementType(),
                            cType.getElementType(), cType.getElementType()},
                           {loc, loc, loc, loc});
      }
    }
    {
      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointToEnd(body);
      Value a = body->getArgument(0);
      Value b = body->getArgument(1);
      Value acc = body->getArgument(3);
      Value mul = rewriter.create<arith::MulFOp>(loc, a, b);
      Value add = rewriter.create<arith::AddFOp>(loc, acc, mul);
      rewriter.create<linalg::YieldOp>(loc, add);
    }

    // === Unpack C: [numPanels, M, NR] -> [M, N] ===
    rewriter.setInsertionPointAfter(packedGeneric);
    Value emptyC = rewriter.create<tensor::EmptyOp>(loc, cType.getShape(),
                                                    cType.getElementType());
    auto unpackC = rewriter.create<linalg::UnPackOp>(
        loc, packedGeneric.getResult(0), emptyC,
        /*innerDimsPos=*/ArrayRef<int64_t>{1},
        /*innerTiles=*/innerTiles,
        /*outerDimsPerm=*/outerDimsPerm);

    rewriter.replaceOp(matmul, unpackC.getResult());
    return success();
  }

private:
  int64_t packFactor;
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSPackPass : impl::KSPackPassBase<KSPackPass> {
  using KSPackPassBase::KSPackPassBase;

  void runOnOperation() override {
    if (packFactor == 0)
      return; // no-op

    RewritePatternSet patterns(&getContext());
    patterns.add<PackMatmulBPattern>(&getContext(), packFactor);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
