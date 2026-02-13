//===----------------------------------------------------------------------===//
// KSTilePass — Single-level tiling of linalg ops using SCF for loops
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSTILEPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSTilePass : impl::KSTilePassBase<KSTilePass> {
  using KSTilePassBase::KSTilePassBase;

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect linalg.matmul ops to tile (can't modify while iterating).
    SmallVector<linalg::MatmulOp> matmulOps;
    funcOp.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

    for (auto matmulOp : matmulOps) {
      // Build tile sizes: [M, N, K]. A tile size of 0 means no tiling.
      SmallVector<OpFoldResult> tileSizes = {
          rewriter.getIndexAttr(tileM),
          rewriter.getIndexAttr(tileN),
          rewriter.getIndexAttr(tileK),
      };

      scf::SCFTilingOptions options;
      options.setTileSizes(tileSizes);

      rewriter.setInsertionPoint(matmulOp);
      auto tilingInterface =
          cast<TilingInterface>(matmulOp.getOperation());
      FailureOr<scf::SCFTilingResult> tilingResult =
          scf::tileUsingSCF(rewriter, tilingInterface, options);

      if (failed(tilingResult)) {
        matmulOp.emitError("failed to tile matmul operation");
        return signalPassFailure();
      }

      rewriter.replaceOp(matmulOp, tilingResult->mergeResult.replacements);
    }
  }
};

} // namespace kernelsmith
