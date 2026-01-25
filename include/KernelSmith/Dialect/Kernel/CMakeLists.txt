# Generate Kernel Dialect TableGen outputs

set(LLVM_TARGET_DEFINITIONS KernelOps.td)
mlir_tablegen(KernelOps.h.inc -gen-op-decls)
mlir_tablegen(KernelOps.cpp.inc -gen-op-defs)
mlir_tablegen(KernelDialect.h.inc -gen-dialect-decls -dialect=ks)
mlir_tablegen(KernelDialect.cpp.inc -gen-dialect-defs -dialect=ks)
add_public_tablegen_target(KSKernelOpsIncGen)

set(LLVM_TARGET_DEFINITIONS KernelTypes.td)
mlir_tablegen(KernelTypes.h.inc -gen-typedef-decls -typedefs-dialect=ks)
mlir_tablegen(KernelTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=ks)
add_public_tablegen_target(KSKernelTypesIncGen)

# Combined target
add_custom_target(KSKernelIncGen
  DEPENDS KSKernelOpsIncGen KSKernelTypesIncGen
)
