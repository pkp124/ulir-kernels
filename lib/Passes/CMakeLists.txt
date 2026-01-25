# KernelSmith Passes Library

add_mlir_library(KernelSmithPasses
  PassRegistration.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/KernelSmith/Passes

  LINK_LIBS PUBLIC
  KernelSmithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
)
