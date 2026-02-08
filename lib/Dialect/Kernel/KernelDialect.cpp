//===----------------------------------------------------------------------===//
// KernelSmith Kernel Dialect Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace kernelsmith::ks;

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.cpp.inc"

void KSDialect::initialize() {
  registerTypes();

  addOperations<
#define GET_OP_LIST
#include "KernelSmith/Dialect/Kernel/KernelOps.cpp.inc"
  >();
}
