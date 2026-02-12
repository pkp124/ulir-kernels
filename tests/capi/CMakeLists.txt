# C API smoke tests for libkernelsmith

add_executable(test-matmul test_matmul.c)
target_link_libraries(test-matmul PRIVATE kernelsmith m)
add_test(NAME capi-matmul COMMAND test-matmul)

add_executable(test-activations test_activations.c)
target_link_libraries(test-activations PRIVATE kernelsmith m)
add_test(NAME capi-activations COMMAND test-activations)

# Python numpy validation
find_package(Python3 COMPONENTS Interpreter)
if(Python3_FOUND)
  add_test(NAME capi-numpy-validation
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/validate_numpy.py
  )
endif()
