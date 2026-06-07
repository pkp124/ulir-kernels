add_executable(host-reference-runner host_reference_runner.c)
target_link_libraries(host-reference-runner PRIVATE kernelsmith m)
