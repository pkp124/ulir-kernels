# KernelSmith C Kernel Library
#
# Builds libkernelsmith.a — a pure C99 static library with no MLIR dependency.
# Target profile is selected via KS_TARGET_PROFILE (default: generic).

set(KS_TARGET_PROFILE "generic" CACHE STRING "Target profile name (generic, x86_avx2, ...)")
set(KS_INT8_RVV_OBJECTS "" CACHE STRING
  "Semicolon-separated generated INT8 RVV object files for ks_dot_i8/ks_matvec_i8"
)

set(KS_TARGET_PROFILE_HEADER "${KS_SOURCE_DIR}/target/${KS_TARGET_PROFILE}.h")
if(NOT EXISTS "${KS_TARGET_PROFILE_HEADER}")
  message(FATAL_ERROR "Target profile not found: ${KS_TARGET_PROFILE_HEADER}")
endif()
message(STATUS "KernelSmith target profile: ${KS_TARGET_PROFILE}")

set(KS_QUANTIZED_INT8_BACKEND "reference" CACHE INTERNAL
  "Selected INT8 quantized C API backend"
  FORCE
)
set(KS_INT8_EXTERNAL_OBJECTS "")
if(KS_INT8_RVV_OBJECTS)
  if(NOT KS_TARGET_PROFILE STREQUAL "riscv_rvv_256")
    message(FATAL_ERROR
      "KS_INT8_RVV_OBJECTS is only supported with KS_TARGET_PROFILE=riscv_rvv_256"
    )
  endif()

  foreach(KS_INT8_RVV_OBJECT IN LISTS KS_INT8_RVV_OBJECTS)
    if(NOT EXISTS "${KS_INT8_RVV_OBJECT}")
      message(FATAL_ERROR "Generated INT8 RVV object not found: ${KS_INT8_RVV_OBJECT}")
    endif()
    set_source_files_properties("${KS_INT8_RVV_OBJECT}" PROPERTIES
      EXTERNAL_OBJECT TRUE
      GENERATED TRUE
    )
    list(APPEND KS_INT8_EXTERNAL_OBJECTS "${KS_INT8_RVV_OBJECT}")
  endforeach()

  set(KS_QUANTIZED_INT8_BACKEND "external" CACHE INTERNAL
    "Selected INT8 quantized C API backend"
    FORCE
  )
  message(STATUS "KernelSmith INT8 RVV objects: ${KS_INT8_RVV_OBJECTS}")
else()
  message(STATUS "KernelSmith INT8 quantized backend: reference")
endif()

add_library(kernelsmith STATIC
  ks_common.c
  ks_matmul.c
  ks_activations.c
  ks_quantized.c
  ${KS_INT8_EXTERNAL_OBJECTS}
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
if(KS_QUANTIZED_INT8_BACKEND STREQUAL "external")
  target_compile_definitions(kernelsmith PRIVATE KS_QUANTIZED_INT8_EXTERNAL=1)
endif()

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
