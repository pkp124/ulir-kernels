# KernelSmith Passes Library

add_mlir_library(KernelSmithPasses
  PassRegistration.cpp
  LowerActivationsPass.cpp
  LowerToLinalgPass.cpp
  TilePass.cpp

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
  MLIRLinalgDialect
  MLIRLinalgTransforms
  MLIRArithDialect
  MLIRMathDialect
  MLIRTensorDialect
  MLIRFuncDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRAffineDialect
)
