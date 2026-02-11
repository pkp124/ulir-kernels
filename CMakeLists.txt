#===----------------------------------------------------------------------===#
# KernelSmith - Craft Optimized AI Kernels for Any Architecture
#===----------------------------------------------------------------------===#

cmake_minimum_required(VERSION 3.20)
project(KernelSmith
  VERSION 0.1.0
  LANGUAGES CXX C
  DESCRIPTION "MLIR-based AI accelerator kernel generation"
)

#===----------------------------------------------------------------------===#
# Project Options
#===----------------------------------------------------------------------===#

option(KS_ENABLE_RISCV "Enable RISC-V RVV target" ON)
option(KS_ENABLE_TESTS "Enable testing with CTest" ON)
option(KS_BUILD_EXAMPLES "Build example kernels" ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

#===----------------------------------------------------------------------===#
# LLVM/MLIR Configuration
#===----------------------------------------------------------------------===#

find_package(MLIR REQUIRED CONFIG)
find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found MLIR ${MLIR_PACKAGE_VERSION}")
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

set(LLVM_RUNTIME_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/bin)
set(LLVM_LIBRARY_OUTPUT_INTDIR ${CMAKE_BINARY_DIR}/lib)
set(MLIR_BINARY_DIR ${CMAKE_BINARY_DIR})

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/include)

link_directories(${LLVM_BUILD_LIBRARY_DIR})
add_definitions(${LLVM_DEFINITIONS})

#===----------------------------------------------------------------------===#
# KernelSmith Configuration
#===----------------------------------------------------------------------===#

set(KS_SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(KS_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

# Target architectures
set(KS_TARGETS "")
if(KS_ENABLE_RISCV)
  list(APPEND KS_TARGETS "RISCV")
  add_definitions(-DKS_ENABLE_RISCV)
endif()

message(STATUS "KernelSmith enabled targets: ${KS_TARGETS}")

#===----------------------------------------------------------------------===#
# Testing Configuration (CTest)
#===----------------------------------------------------------------------===#

if(KS_ENABLE_TESTS)
  enable_testing()
  include(CTest)

  # Find lit for MLIR tests
  find_program(LIT_COMMAND lit)
  if(LIT_COMMAND)
    message(STATUS "Found lit: ${LIT_COMMAND}")
  else()
    message(STATUS "lit not found, install with: pip install lit")
  endif()

  # Find FileCheck
  find_program(FILECHECK_COMMAND FileCheck
    PATHS ${LLVM_TOOLS_BINARY_DIR}
  )
  if(FILECHECK_COMMAND)
    message(STATUS "Found FileCheck: ${FILECHECK_COMMAND}")
  endif()
endif()

#===----------------------------------------------------------------------===#
# Subdirectories
#===----------------------------------------------------------------------===#

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)

if(KS_ENABLE_TESTS)
  add_subdirectory(tests)
endif()

if(KS_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

#===----------------------------------------------------------------------===#
# Installation
#===----------------------------------------------------------------------===#

include(GNUInstallDirs)

install(DIRECTORY include/KernelSmith
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
  PATTERN "*.td"
)

install(DIRECTORY include/kernelsmith
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h"
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/include/KernelSmith
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.h.inc"
)

#===----------------------------------------------------------------------===#
# Summary
#===----------------------------------------------------------------------===#

message(STATUS "")
message(STATUS "KernelSmith Configuration Summary:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  Install prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "  RISC-V RVV:     ${KS_ENABLE_RISCV}")
message(STATUS "  Tests:          ${KS_ENABLE_TESTS}")
message(STATUS "  Examples:       ${KS_BUILD_EXAMPLES}")
message(STATUS "")
