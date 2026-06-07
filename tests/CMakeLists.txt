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

# Python golden-reference tests
find_program(PYTEST_COMMAND pytest)
if(PYTEST_COMMAND)
  add_custom_target(check-kernelsmith-golden
    COMMAND ${PYTEST_COMMAND} tests/test_golden_verify.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Running KernelSmith golden-reference pytest tests"
  )

  add_test(NAME kernelsmith-golden-pytest
    COMMAND ${PYTEST_COMMAND} tests/test_golden_verify.py
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  )
endif()

# Unit tests subdirectory
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/unit/CMakeLists.txt)
  add_subdirectory(unit)
endif()

# C API smoke tests
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/capi/CMakeLists.txt)
  add_subdirectory(capi)
endif()

# Combined target for all tests
add_custom_target(check-kernelsmith
  COMMENT "Running all KernelSmith tests"
)
add_dependencies(check-kernelsmith check-kernelsmith-lit)
if(TARGET kernelsmith-unit-tests)
  add_dependencies(check-kernelsmith kernelsmith-unit-tests)
endif()
if(TARGET check-kernelsmith-golden)
  add_dependencies(check-kernelsmith check-kernelsmith-golden)
endif()
