//===----------------------------------------------------------------------===//
// KSLowerToRVVPass — Lower vector dialect to LLVM IR for RISC-V RVV
//
// Milestone: M4 (RISC-V RVV Target)
// Design: DES-009 (docs/design/DES-009-m4-rvv-lowering.md)
//
// This pass applies a complete lowering pipeline from the mixed
// vector/linalg/scf/tensor dialect level to the LLVM dialect. When the LLVM IR
// is subsequently compiled with `llc -march=riscv64 -mattr=+v`, the RISC-V
// vector backend emits VLE/VSE/VFMACC and related RVV instructions.
//
// Pipeline stages (in order):
//   1. one-shot-bufferize           tensor -> memref
//      (alloc at function boundary)
//   2. convert-linalg-to-loops      remaining linalg.generic -> scf.for
//   3. vector preparation           multi-reduction/transfer -> LLVM-ready ops
//   4. expand-strided-metadata      subview -> metadata + affine
//   5. lower-affine                  affine.apply -> arith
//   6. convert-scf-to-cf            scf.for/if -> cf.br (LLVM-compatible CFG)
//   7. convert-vector-to-llvm       vector.* -> llvm.* (RVV via LLVM backend)
//   8. convert-math/ub-to-llvm      math.*, ub.* -> llvm.*
//   9. finalize-memref-to-llvm      memref.* -> llvm.* (pointer arithmetic)
//  10. convert-arith-to-llvm        arith.* -> llvm.*
//  11. convert-func-to-llvm         func.func -> llvm.func
//  12. reconcile-unrealized-casts   clean up cast chains
//
// After this pass, use mlir-translate + llc to produce assembly:
//   mlir-translate --mlir-to-llvmir module.mlir -o module.ll
//   llc -march=riscv64 -mattr=+v,+zve64d -float-abi=hard \
//       -filetype=obj module.ll -o module.o
//
// Or use the provided scripts/compile-rvv.sh driver.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"
#include "KernelSmith/Passes/Passes.h"

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/UBToLLVM/UBToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSLOWERTORVVPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

struct PrepareVectorsForLLVMPass
    : PassWrapper<PrepareVectorsForLLVMPass, OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PrepareVectorsForLLVMPass)

  void runOnOperation() override {
    // Bound compile-time expansion while retaining the 16x32 RVV register tile
    // described by the target profile.
    constexpr int64_t maxReductionResultElements = 16 * 32;
    WalkResult sizeCheck =
        getOperation().walk([&](vector::MultiDimReductionOp op) {
          auto resultType = dyn_cast<VectorType>(op.getResult().getType());
          if (!resultType ||
              resultType.getNumElements() <= maxReductionResultElements)
            return WalkResult::advance();

          op.emitError()
              << "--ks-lower-to-rvv does not support "
                 "vector.multi_reduction with "
              << resultType.getNumElements()
              << " result elements; maximum is " << maxReductionResultElements
              << "; tile the operation to bound each vector reduction result";
          return WalkResult::interrupt();
        });
    if (sizeCheck.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    RewritePatternSet patterns(&getContext());
    vector::populateVectorMultiReductionLoweringPatterns(
        patterns, vector::VectorMultiReductionLowering::InnerParallel);
    populateVectorToSCFConversionPatterns(patterns);
    LogicalResult result =
        applyPatternsGreedily(getOperation(), std::move(patterns));
    if (failed(result)) {
      getOperation().emitError(
          "--ks-lower-to-rvv failed while preparing vector operations");
      signalPassFailure();
    }
  }
};

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSLowerToRVVPass : impl::KSLowerToRVVPassBase<KSLowerToRVVPass> {
  using KSLowerToRVVPassBase::KSLowerToRVVPassBase;

  void runOnOperation() override {
    ModuleOp module = getOperation();

    // Build a nested pass manager that applies the lowering pipeline.
    // Running nested passes keeps each stage's rewrites visible to the next.
    PassManager pm(module->getContext());

    // Stage 1: Bufferize tensors to memrefs.
    // one-shot-bufferize with function-boundary allocation:
    //   - function arguments stay as memref arguments (no internal alloc)
    //   - result tensors become memref returns (caller owns storage)
    bufferization::OneShotBufferizePassOptions bufOpts;
    bufOpts.allowReturnAllocsFromLoops = true;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

    // Stage 2: Lower remaining linalg.generic -> scf.for + memref.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

    // Stage 3: Prepare vector operations that do not lower directly to LLVM.
    // Transfers must become rank-1 operations, and multi-reductions must become
    // vector.reduction operations. Reject reductions whose expansion would
    // exceed the bounded result-size limit.
    pm.addNestedPass<func::FuncOp>(
        std::make_unique<PrepareVectorsForLLVMPass>());

    // Stage 4: Expand memref.subview before affine lowering. Tiling introduces
    // subviews, and the expansion itself may introduce affine.apply.
    pm.addPass(memref::createExpandStridedMetadataPass());

    // Stage 5: Lower affine.apply -> arith (required before SCF->CF).
    pm.addNestedPass<func::FuncOp>(createLowerAffinePass());

    // Stage 6: Convert scf.for/if -> cf.br (flat CFG for LLVM).
    pm.addPass(createSCFToControlFlowPass());

    // Stage 6b: Convert cf.br/cf.cond_br -> llvm.br/llvm.cond_br.
    // FuncToLLVM only converts the entry block signature; the remaining
    // unstructured control flow (loop back-edges, conditionals) must be
    // lowered explicitly here before the LLVM dialect conversion passes.
    pm.addPass(createConvertControlFlowToLLVMPass());

    // Stage 7: Lower vector dialect -> LLVM dialect.
    // The RISC-V V backend in LLVM translates vector.* intrinsics to RVV when
    // the target triple and +v feature are set (via llc flags at compile time).
    pm.addPass(createConvertVectorToLLVMPass());

    // Stage 8: Lower math operations emitted by normalization kernels and
    // poison padding values emitted by vectorization.
    pm.addPass(createConvertMathToLLVMPass());
    pm.addPass(createUBToLLVMConversionPass());

    // Stage 9: Lower memref -> LLVM (pointer arithmetic, GEPs).
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    // Stage 10: Lower arith -> LLVM.
    pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());

    // Stage 11: Lower func.func -> llvm.func (calling convention, linkage).
    pm.addPass(createConvertFuncToLLVMPass());

    // Stage 12: Clean up unrealized casts left by conversions.
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(module))) {
      signalPassFailure();
      return;
    }

    Operation *untranslatedOp = nullptr;
    module.walk([&](Operation *op) {
      if (isa<ModuleOp>(op) ||
          op->getName().getDialectNamespace() ==
              LLVM::LLVMDialect::getDialectNamespace())
        return WalkResult::advance();
      untranslatedOp = op;
      return WalkResult::interrupt();
    });
    if (untranslatedOp) {
      untranslatedOp->emitError()
          << "--ks-lower-to-rvv failed to produce translatable LLVM dialect "
             "IR; unsupported operation remains: "
          << untranslatedOp->getName();
      signalPassFailure();
    }
  }
};

} // namespace kernelsmith
