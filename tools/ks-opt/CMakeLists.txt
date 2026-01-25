# ks-opt tool

set(LLVM_LINK_COMPONENTS
  Support
)

get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

add_llvm_executable(ks-opt
  ks-opt.cpp
)

llvm_update_compile_flags(ks-opt)

target_link_libraries(ks-opt
  PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  KernelSmithDialect
  KernelSmithPasses
  MLIROptLib
  MLIRPass
  MLIRTransforms
  MLIRSupport
)

mlir_check_all_link_libraries(ks-opt)

install(TARGETS ks-opt
  RUNTIME DESTINATION bin
)
