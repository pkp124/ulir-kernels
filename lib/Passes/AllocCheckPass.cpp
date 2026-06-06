//===----------------------------------------------------------------------===//
// KSAllocCheckPass
//
// Rejects internal memref.alloc operations after bufferization. KernelSmith
// generated kernels must use caller-provided buffers and explicit workspace
// arguments so the shipped C library has predictable memory usage.
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSALLOCCHECKPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

struct KSAllocCheckPass : impl::KSAllocCheckPassBase<KSAllocCheckPass> {
  using KSAllocCheckPassBase::KSAllocCheckPassBase;

  void runOnOperation() override {
    bool foundAlloc = false;

    getOperation().walk([&](memref::AllocOp allocOp) {
      foundAlloc = true;
      allocOp.emitError("KernelSmith generated kernels must not contain "
                        "memref.alloc; use caller-provided workspace buffers");
    });

    if (foundAlloc)
      signalPassFailure();
  }
};

} // namespace kernelsmith
