//===----------------------------------------------------------------------===//
// KSAllocCheckPass — Reject stray memref.alloc operations
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace kernelsmith {

#define GEN_PASS_DEF_KSALLOCCHECKPASS
#include "KernelSmith/Passes/Passes.h.inc"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Pass implementation
//===----------------------------------------------------------------------===//

struct KSAllocCheckPass
    : impl::KSAllocCheckPassBase<KSAllocCheckPass> {
  using KSAllocCheckPassBase::KSAllocCheckPassBase;

  void runOnOperation() override {
    bool foundAlloc = false;
    getOperation().walk([&](memref::AllocOp allocOp) {
      allocOp.emitError(
          "unexpected memref.alloc: all buffers must be function arguments "
          "(zero internal malloc)");
      foundAlloc = true;
    });
    if (foundAlloc)
      signalPassFailure();
  }
};

} // namespace kernelsmith
