# KernelSmith Tests (CTest)

# Configure lit site config
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit/lit.site.cfg.py
  @ONLY
)

# Add lit tests target
if(LIT_COMMAND)
  add_custom_target(check-kernelsmith-lit
    COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/lit
      -v
      --param ks_tools_dir=${LLVM_RUNTIME_OUTPUT_INTDIR}
    DEPENDS ks-opt
    COMMENT "Running KernelSmith lit tests"
  )

  # Register lit tests with CTest
  add_test(NAME kernelsmith-lit
    COMMAND ${LIT_COMMAND} ${CMAKE_CURRENT_BINARY_DIR}/lit
      -v
      --param ks_tools_dir=${LLVM_RUNTIME_OUTPUT_INTDIR}
  )
endif()

# Unit tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit/CMakeLists.txt)
  add_subdirectory(unit)
endif()

# Combined target for all tests
add_custom_target(check-kernelsmith
  COMMENT "Running all KernelSmith tests"
)
add_dependencies(check-kernelsmith check-kernelsmith-lit)
if(TARGET kernelsmith-unit-tests)
  add_dependencies(check-kernelsmith kernelsmith-unit-tests)
endif()
