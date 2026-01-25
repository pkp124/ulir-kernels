# KernelSmith Kernel Dialect Library

add_mlir_dialect_library(KernelSmithDialect
  KernelDialect.cpp
  KernelOps.cpp
  KernelTypes.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/KernelSmith/Dialect/Kernel

  DEPENDS
  KSKernelIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRDialect
  MLIRArithDialect
  MLIRTensorDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRMemRefDialect
  MLIRFuncDialect
  MLIRSupport
)
