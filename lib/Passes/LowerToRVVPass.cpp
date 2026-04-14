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
//   1. one-shot-bufferize           tensor -> memref (alloc at function boundary)
//   2. convert-linalg-to-loops      remaining linalg.generic -> scf.for
//   3. lower-affine                  affine.apply -> arith
//   4. convert-scf-to-cf            scf.for/if -> cf.br (LLVM-compatible CFG)
//   5. convert-vector-to-llvm       vector.* -> llvm.* (RVV via LLVM backend)
//   6. finalize-memref-to-llvm      memref.* -> llvm.* (pointer arithmetic)
//   7. convert-arith-to-llvm        arith.* -> llvm.*
//   8. convert-func-to-llvm         func.func -> llvm.func
//   9. reconcile-unrealized-casts   clean up cast chains
//
// After this pass, use mlir-translate + llc to produce assembly:
//   mlir-translate --mlir-to-llvmir module.mlir -o module.ll
//   llc -march=riscv64 -mattr=+v,+zve64d -float-abi=double \
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
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/LinalgToStandard/LinalgToStandard.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/CF/IR/CFOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSLOWERTORVVPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

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
    bufferization::OneShotBufferizationOptions bufOpts;
    bufOpts.allowReturnAllocsFromLoops = true;
    bufOpts.bufferizeFunctionBoundaries = true;
    pm.addPass(bufferization::createOneShotBufferizePass(bufOpts));

    // Stage 2: Lower remaining linalg.generic -> scf.for + memref.
    pm.addNestedPass<func::FuncOp>(createConvertLinalgToLoopsPass());

    // Stage 3: Lower affine.apply -> arith (required before SCF->CF).
    pm.addNestedPass<func::FuncOp>(affine::createLowerAffinePass());

    // Stage 4: Convert scf.for/if -> cf.br (flat CFG for LLVM).
    pm.addPass(createConvertSCFToCFPass());

    // Stage 4b: Convert cf.br/cf.cond_br -> llvm.br/llvm.cond_br.
    // FuncToLLVM only converts the entry block signature; the remaining
    // unstructured control flow (loop back-edges, conditionals) must be
    // lowered explicitly here before the LLVM dialect conversion passes.
    pm.addPass(createConvertControlFlowToLLVMPass());

    // Stage 5: Lower vector dialect -> LLVM dialect.
    // The RISC-V V backend in LLVM translates vector.* intrinsics to RVV when
    // the target triple and +v feature are set (via llc flags at compile time).
    pm.addPass(createConvertVectorToLLVMPass());

    // Stage 6: Lower memref -> LLVM (pointer arithmetic, GEPs).
    pm.addPass(createFinalizeMemRefToLLVMConversionPass());

    // Stage 7: Lower arith -> LLVM.
    pm.addNestedPass<func::FuncOp>(createArithToLLVMConversionPass());

    // Stage 8: Lower func.func -> llvm.func (calling convention, linkage).
    pm.addPass(createConvertFuncToLLVMPass());

    // Stage 9: Clean up unrealized casts left by conversions.
    pm.addPass(createReconcileUnrealizedCastsPass());

    if (failed(pm.run(module)))
      signalPassFailure();
  }
};

} // namespace kernelsmith
