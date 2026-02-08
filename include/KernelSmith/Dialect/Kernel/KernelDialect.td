//===----------------------------------------------------------------------===//
// KernelSmith Kernel Dialect Definition
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_TD
#define KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_TD

include "mlir/IR/OpBase.td"
include "mlir/IR/AttrTypeBase.td"

//===----------------------------------------------------------------------===//
// Kernel Dialect
//===----------------------------------------------------------------------===//

def KS_Dialect : Dialect {
  let name = "ks";
  let cppNamespace = "::kernelsmith::ks";
  let summary = "KernelSmith dialect for AI accelerator kernels";
  let description = [{
    The KernelSmith (ks) dialect provides high-level operations for defining
    AI accelerator kernels such as matrix multiplication, convolution,
    attention, and activation functions.

    Operations are lowered through linalg and vector dialects to
    target-specific code (RISC-V RVV, ARM SVE, etc.).
  }];

  let useDefaultTypePrinterParser = 1;

  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::memref::MemRefDialect",
    "mlir::func::FuncDialect"
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];
}

//===----------------------------------------------------------------------===//
// Base Classes
//===----------------------------------------------------------------------===//

class KS_Op<string mnemonic, list<Trait> traits = []> :
    Op<KS_Dialect, mnemonic, traits>;

class KS_Type<string name, string typeMnemonic, list<Trait> traits = []>
    : TypeDef<KS_Dialect, name, traits> {
  let mnemonic = typeMnemonic;
}

#endif // KERNELSMITH_DIALECT_KERNEL_KERNELDIALECT_TD
