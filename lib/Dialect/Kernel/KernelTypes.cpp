//===----------------------------------------------------------------------===//
// KernelSmith Kernel Types Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace kernelsmith::ks;

//===----------------------------------------------------------------------===//
// TileType
//===----------------------------------------------------------------------===//

int64_t TileType::getNumElements() const {
  int64_t num = 1;
  for (int64_t dim : getShape()) {
    if (dim < 0)
      return -1;
    num *= dim;
  }
  return num;
}

//===----------------------------------------------------------------------===//
// TableGen Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "KernelSmith/Dialect/Kernel/KernelTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Dialect Type Registration
//===----------------------------------------------------------------------===//

void KSDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "KernelSmith/Dialect/Kernel/KernelTypes.cpp.inc"
      >();
}
