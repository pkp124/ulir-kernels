//===----------------------------------------------------------------------===//
// KSLowerToLinalgPass — Lower ks.matmul to linalg.matmul
//
// Design: DES-008 (docs/design/DES-008-m3-linalg-matmul-lowering.md)
//
// Lowering rule:
//   ks.matmul %A, %B : tensor<MxKxT>, tensor<KxNxT> -> tensor<MxNxT>
//   ──►
//   %empty  = tensor.empty() : tensor<MxNxT>
//   %zero   = arith.constant 0 : T
//   %filled = linalg.fill ins(%zero) outs(%empty) -> tensor<MxNxT>
//   %C      = linalg.matmul ins(%A, %B) outs(%filled) -> tensor<MxNxT>
//
// The zero-fill is required because linalg.matmul accumulates (C += A*B),
// while ks.matmul has pure-replacement semantics (C = A*B).
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith {

// Pull in the generated pass base class.
#define GEN_PASS_DEF_KSLOWERTOLINALGPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Build a zero-filled tensor of the given result type.
///
/// For dynamic dimensions: M comes from lhs dim-0, N comes from rhs dim-1.
/// This covers the only two dynamic dimensions possible in a 2-D matmul result.
static Value createZeroFilledTensor(OpBuilder &b, Location loc,
                                    RankedTensorType resultType, Value lhs,
                                    Value rhs) {
  // Collect runtime values for dynamic dimensions in row-major order.
  SmallVector<Value> dynamicSizes;
  if (resultType.isDynamicDim(0))
    dynamicSizes.push_back(b.create<tensor::DimOp>(loc, lhs, 0));
  if (resultType.isDynamicDim(1))
    dynamicSizes.push_back(b.create<tensor::DimOp>(loc, rhs, 1));

  Value empty = b.create<tensor::EmptyOp>(
      loc, resultType.getShape(), resultType.getElementType(), dynamicSizes);

  // Zero constant matching the element type (float or integer).
  Value zero = b.create<arith::ConstantOp>(
      loc, b.getZeroAttr(resultType.getElementType()));

  return b.create<linalg::FillOp>(loc, zero, empty).getResult(0);
}

//===----------------------------------------------------------------------===//
// Lowering pattern
//===----------------------------------------------------------------------===//

/// Lower ks.matmul to linalg.matmul with a zero-initialized accumulator.
///
/// Uses OpRewritePattern (greedy driver) — same infrastructure as
/// LowerActivationsPass. The named linalg.matmul op is chosen over
/// linalg.generic because:
///   1. It is directly tileable by KSTilePass via TilingInterface.
///   2. Future packing (M4) relies on linalg::packMatmulGreedily which
///      pattern-matches linalg.matmul by name.
///   3. Vectorization transforms also prefer named ops.
struct MatmulToLinalgPattern : public OpRewritePattern<ks::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ks::MatmulOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto resultType = cast<RankedTensorType>(op.getType());

    Value filled = createZeroFilledTensor(rewriter, loc, resultType,
                                          op.getLhs(), op.getRhs());

    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
        op, TypeRange{resultType}, ValueRange{op.getLhs(), op.getRhs()},
        ValueRange{filled});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSLowerToLinalgPass
    : impl::KSLowerToLinalgPassBase<KSLowerToLinalgPass> {
  using KSLowerToLinalgPassBase::KSLowerToLinalgPassBase;

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<MatmulToLinalgPattern>(&getContext());

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace kernelsmith
