# C API smoke tests for libkernelsmith

add_executable(test-matmul test_matmul.c)
target_link_libraries(test-matmul PRIVATE kernelsmith m)
add_test(NAME capi-matmul COMMAND test-matmul)

add_executable(test-activations test_activations.c)
target_link_libraries(test-activations PRIVATE kernelsmith m)
add_test(NAME capi-activations COMMAND test-activations)

add_executable(test-quantized test_quantized.c)
target_link_libraries(test-quantized PRIVATE kernelsmith m)
add_test(NAME capi-quantized COMMAND test-quantized)

if(KS_QUANTIZED_INT8_BACKEND STREQUAL "external")
  add_executable(test-quantized-generated-override
    test_quantized_generated_override.c
  )
  target_link_libraries(test-quantized-generated-override PRIVATE kernelsmith m)
  add_test(NAME capi-quantized-generated-override
    COMMAND test-quantized-generated-override
  )
endif()

# Python numpy validation
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME capi-numpy-validation
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_numpy.py
  )

  add_test(NAME capi-int8-generated-profile-build
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/check_int8_generated_profile_build.py
      --source-dir ${CMAKE_SOURCE_DIR}
      --build-dir ${CMAKE_CURRENT_BINARY_DIR}/int8-generated-profile-build
      --cmake ${CMAKE_COMMAND}
      --ctest ${CMAKE_CTEST_COMMAND}
      --c-compiler ${CMAKE_C_COMPILER}
      --cxx-compiler ${CMAKE_CXX_COMPILER}
      --generator ${CMAKE_GENERATOR}
      --mlir-dir ${MLIR_DIR}
      --llvm-dir ${LLVM_DIR}
      --lit-command ${LIT_COMMAND}
  )
endif()
