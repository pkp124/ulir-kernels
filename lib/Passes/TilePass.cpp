//===----------------------------------------------------------------------===//
// KSTilePass — Tile linalg structured ops for cache efficiency
//
// Design: DES-008 (docs/design/DES-008-m3-linalg-matmul-lowering.md)
//
// Tiling rule (example, tile_m=64, tile_n=64, tile_k=256):
//
//   %C = linalg.matmul ins(%A, %B) outs(%zero)
//   ──►
//   %C = scf.for %m = 0 to M step 64 iter_args(%c0 = %zero) {
//     %C1 = scf.for %n = 0 to N step 64 iter_args(%c1 = %c0) {
//       %C2 = scf.for %k = 0 to K step 256 iter_args(%c2 = %c1) {
//         %a_tile = tensor.extract_slice %A[%m,%k][tm,tk][1,1]
//         %b_tile = tensor.extract_slice %B[%k,%n][tk,tn][1,1]
//         %c_tile = tensor.extract_slice %c2[%m,%n][tm,tn][1,1]
//         %c_new  = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
//         %c2_new = tensor.insert_slice %c_new into %c2[%m,%n][tm,tn][1,1]
//         scf.yield %c2_new
//       }
//       scf.yield %C2
//     }
//     scf.yield %C1
//   }
//
// Uses scf::tileUsingSCF (MLIR 20 recommended TilingInterface API).
// Tile size = 0 on any dimension means "do not tile that dimension".
// Non-divisible dimensions produce correct tail handling via affine.min.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/TileUsingInterface.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/TilingInterface.h"

namespace kernelsmith {

// Pull in the generated pass base class.
#define GEN_PASS_DEF_KSTILEPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSTilePass : impl::KSTilePassBase<KSTilePass> {
  using KSTilePassBase::KSTilePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func->getContext());

    // Collect all linalg.matmul ops before transforming to avoid
    // invalidating the walk iterator during in-place rewriting.
    SmallVector<linalg::MatmulOp> matmulOps;
    func.walk([&](linalg::MatmulOp op) { matmulOps.push_back(op); });

    if (matmulOps.empty())
      return;

    // Build tile sizes as OpFoldResult (attribute constants, index type).
    // A size of 0 means "do not tile this dimension".
    SmallVector<OpFoldResult> tileSizes = {
        rewriter.getIndexAttr(tileSizeM),
        rewriter.getIndexAttr(tileSizeN),
        rewriter.getIndexAttr(tileSizeK),
    };

    scf::SCFTilingOptions tilingOptions;
    tilingOptions.setTileSizes(tileSizes);

    for (linalg::MatmulOp matmulOp : matmulOps) {
      rewriter.setInsertionPoint(matmulOp);

      auto tilingInterface =
          dyn_cast<TilingInterface>(matmulOp.getOperation());
      if (!tilingInterface) {
        matmulOp->emitError("linalg.matmul does not implement TilingInterface");
        signalPassFailure();
        return;
      }

      FailureOr<scf::SCFTilingResult> result =
          scf::tileUsingSCF(rewriter, tilingInterface, tilingOptions);

      if (failed(result)) {
        matmulOp->emitError("scf::tileUsingSCF failed");
        signalPassFailure();
        return;
      }

      // Replace the original op with the results produced by the tiled loops.
      rewriter.replaceOp(matmulOp, result->replacements);
    }
  }
};

} // namespace kernelsmith
