//===----------------------------------------------------------------------===//
// KernelSmith Kernel Dialect Header
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_H
#define KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"

#include "KernelSmith/Dialect/Kernel/KernelDialect.h.inc"

#define GET_TYPEDEF_CLASSES
#include "KernelSmith/Dialect/Kernel/KernelTypes.h.inc"

#define GET_OP_CLASSES
#include "KernelSmith/Dialect/Kernel/KernelOps.h.inc"

#endif // KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_H
