# KernelSmith Passes Library

add_mlir_library(KernelSmithPasses
  PassRegistration.cpp
  LowerActivationsPass.cpp
  LowerToLinalgPass.cpp
  TilePass.cpp
  PackPass.cpp
  MaterializePackWorkspacePass.cpp
  AllocCheckPass.cpp
  VectorizePass.cpp
  LowerToRVVPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/KernelSmith/Passes

  DEPENDS
  KSPassesIncGen

  LINK_LIBS PUBLIC
  KernelSmithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  # Dialect libraries
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRAffineDialect
  MLIRVectorDialect
  MLIRMemRefDialect
  MLIRControlFlowDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRLLVMDialect
  # Conversion libraries (M4: vectorize + lower-to-rvv pipeline)
  MLIRAffineToStandard
  MLIRArithToLLVM
  MLIRControlFlowToLLVM
  MLIRFuncToLLVM
  MLIRLinalgToStandard
  MLIRMemRefToLLVM
  MLIRSCFToControlFlow
  MLIRVectorToLLVMPass
)
