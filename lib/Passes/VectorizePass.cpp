//===----------------------------------------------------------------------===//
// KSVectorizePass — Vectorize inner linalg ops to the vector dialect
//
// Milestone: M4 (RISC-V RVV Target)
// Design: DES-009 (docs/design/DES-009-m4-rvv-lowering.md)
//
// This pass applies MLIR's linalg vectorization to the innermost tiled ops.
// It converts linalg.matmul, linalg.generic (from activations and packed gemm),
// and similar structured ops to vector.transfer_read/write + vector.contract ops.
//
// The resulting vector dialect IR is target-independent. The --ks-lower-to-rvv
// pass then lowers it to LLVM dialect with RVV semantics (via the RISC-V
// vector backend in LLVM).
//
// Vectorization requirements:
//   - All linalg ops must have fully static shapes (set by --ks-tile).
//   - The inner tile N dimension should be a multiple of VLMAX for the target
//     to enable clean vector register allocation without padding.
//
// Typical pipeline for RVV matmul:
//   --ks-lower-to-linalg                  ks.matmul -> linalg.matmul
//   --ks-tile="tile-size-m=128 ..."       L2 tiling
//   --ks-pack                             B packing -> linalg.generic
//   --ks-tile="tile-size-m=16 ..."        register tiling
//   --ks-vectorize                        linalg.* -> vector.*
//   --ks-lower-to-rvv                     vector.* -> llvm.* (RVV)
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSVECTORIZEPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSVectorizePass : impl::KSVectorizePassBase<KSVectorizePass> {
  using KSVectorizePassBase::KSVectorizePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func->getContext());

    // Collect all linalg ops that are vectorizable.
    // We collect before transforming to avoid iterator invalidation.
    SmallVector<linalg::LinalgOp> linalgOps;
    func.walk([&](linalg::LinalgOp op) {
      // Skip ops that are not at the innermost level (i.e., those that still
      // have tiling loops around them will be inside scf.for — they are fine).
      // The vectorizer handles them in place.
      linalgOps.push_back(op);
    });

    if (linalgOps.empty())
      return;

    // Apply linalg vectorization to each op.
    // linalg::vectorize rewrites the op into vector.contract + transfer ops.
    // It requires:
    //   - Static shapes on all dimensions.
    //   - No indexing maps with unsupported patterns.
    for (linalg::LinalgOp linalgOp : linalgOps) {
      // Skip pack/unpack ops — they are not directly vectorizable.
      if (isa<linalg::PackOp, linalg::UnPackOp>(linalgOp.getOperation()))
        continue;

      rewriter.setInsertionPoint(linalgOp);

      // linalg::vectorize: vectorizeNDExtract=false (standard GEMM path).
      if (failed(linalg::vectorize(rewriter, linalgOp, /*inputVectorSizes=*/{},
                                    /*inputScalableVecDims=*/{},
                                    /*vectorizeNDExtract=*/false,
                                    /*flatten1DDepthwiseConv=*/false))) {
        // Non-fatal: some ops (e.g. dynamic-shape linalg.generic from partial
        // tiles) cannot be vectorized. Emit a remark and continue.
        linalgOp.emitRemark("ks-vectorize: skipping non-vectorizable op");
      }
    }
  }
};

} // namespace kernelsmith
