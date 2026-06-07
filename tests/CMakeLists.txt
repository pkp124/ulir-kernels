# KernelSmith Tests (CTest)

find_package(Python3 COMPONENTS Interpreter)

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

add_subdirectory(host_reference)

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

if(Python3_FOUND)
  set(KS_GOLDEN_HOST_OUTPUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/golden-host
  )
  set(KS_GOLDEN_VERIFY_COMMAND
    ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/verify.py
    --target host_reference
    --manifest-dir ${CMAKE_CURRENT_SOURCE_DIR}/golden/generated
    --output-dir ${KS_GOLDEN_HOST_OUTPUT_DIR}
    --host-runner $<TARGET_FILE:host-reference-runner>
  )

  add_test(NAME kernelsmith-host-golden-relu
    COMMAND ${KS_GOLDEN_VERIFY_COMMAND}
      --case ${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/relu_f32_smoke.json
  )
  add_test(NAME kernelsmith-host-golden-matmul
    COMMAND ${KS_GOLDEN_VERIFY_COMMAND}
      --case ${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/matmul_f32_smoke.json
  )

  add_custom_target(check-kernelsmith-host-golden
    COMMAND ${KS_GOLDEN_VERIFY_COMMAND}
      --case ${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/relu_f32_smoke.json
    COMMAND ${KS_GOLDEN_VERIFY_COMMAND}
      --case ${CMAKE_CURRENT_SOURCE_DIR}/golden/cases/matmul_f32_smoke.json
    DEPENDS host-reference-runner
    COMMENT "Running KernelSmith host golden-reference checks"
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
if(TARGET check-kernelsmith-host-golden)
  add_dependencies(check-kernelsmith check-kernelsmith-host-golden)
endif()
