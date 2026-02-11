# C API smoke tests for libkernelsmith

add_executable(test-matmul test_matmul.c)
target_link_libraries(test-matmul PRIVATE kernelsmith m)
add_test(NAME capi-matmul COMMAND test-matmul)

add_executable(test-activations test_activations.c)
target_link_libraries(test-activations PRIVATE kernelsmith m)
add_test(NAME capi-activations COMMAND test-activations)
