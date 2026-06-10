# KernelSmith C Kernel Library
#
# Builds libkernelsmith.a — a pure C99 static library with no MLIR dependency.
# Target profile is selected via KS_TARGET_PROFILE (default: generic).

set(KS_TARGET_PROFILE "generic" CACHE STRING "Target profile name (generic, x86_avx2, ...)")

set(KS_TARGET_PROFILE_HEADER "${KS_SOURCE_DIR}/target/${KS_TARGET_PROFILE}.h")
if(NOT EXISTS "${KS_TARGET_PROFILE_HEADER}")
  message(FATAL_ERROR "Target profile not found: ${KS_TARGET_PROFILE_HEADER}")
endif()
message(STATUS "KernelSmith target profile: ${KS_TARGET_PROFILE}")

add_library(kernelsmith STATIC
  ks_common.c
  ks_matmul.c
  ks_activations.c
  ks_quantized.c
)

# C99 for the kernel library (independent of C++17 used by MLIR code).
set_target_properties(kernelsmith PROPERTIES
  C_STANDARD 99
  C_STANDARD_REQUIRED ON
  C_EXTENSIONS OFF
)

target_include_directories(kernelsmith
  PUBLIC
    $<BUILD_INTERFACE:${KS_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# Inject the target profile header via forced include — every .c file sees it
# as "ks_target_profile.h" without needing a path in source code.
target_compile_options(kernelsmith PRIVATE
  -include "${KS_TARGET_PROFILE_HEADER}"
)

target_link_libraries(kernelsmith PRIVATE m)

# Install the static library.
install(TARGETS kernelsmith
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

# Install public C headers.
install(DIRECTORY ${KS_SOURCE_DIR}/include/kernelsmith
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING PATTERN "*.h"
)
