# KernelSmith Output Strategy: From MLIR to Usable Libraries

## Current Output (Status Quo)

### What Gets Built Today

```
$ make build

Output in build/ directory:
├─ bin/
│  └─ ks-opt          # MLIR dialect + optimization tool (like mlir-opt)
└─ lib/
   ├─ libKernelSmithDialect.a    # MLIR dialect definitions only
   ├─ libKernelSmithPasses.a     # Lowering passes (no compiled code)
   └─ (other LLVM/MLIR libs)
```

### What These Are
- **ks-opt**: Command-line tool that transforms MLIR IR
  - Takes: `ks.matmul` operations
  - Outputs: MLIR text or LLVM IR (human-readable)
  - Usage: `ks-opt --ks-lower-to-linalg --ks-vectorize input.mlir`

- **Dialect Libraries**: Compiler framework code only
  - Not machine code
  - Not callable from C/C++
  - Only for MLIR developers

### Problem: NOT USABLE by External Projects
```cpp
// Can't do this:
#include <kernelsmith/kernels.h>
kernelsmith::matmul(A, B, C, M, N, K);  // ❌ Header doesn't exist
                                        // ❌ No compiled code
                                        // ❌ Can't link
```

---

## Desired Output (DES-005 Design)

### What Should Get Built

```
$ make build && make package

Output in build/ directory:
├─ bin/
│  └─ ks-opt          # Still here (useful for development)
├─ lib/
│  ├─ libKernelSmithDialect.a    # Still here (framework)
│  ├─ libKernelSmithPasses.a     # Still here (framework)
│  ├─ libks-kernels.a            # ✨ NEW: Compiled RISC-V kernels
│  │                             #   - matmul implementation
│  │                             #   - conv2d implementation
│  │                             #   - attention implementation
│  │                             #   - + activation, normalization, reduction
│  └─ libks-runtime.a            # ✨ NEW: Runtime utilities
│                                #   - profiling, configuration
│                                #   - memory management helpers
├─ include/kernelsmith/
│  ├─ kernels.h                  # ✨ NEW: Public kernel API
│  │                             #   void matmul(...)
│  │                             #   void conv2d(...)
│  │                             #   void attention(...)
│  ├─ runtime.h                  # ✨ NEW: Runtime config
│  │                             #   RuntimeInfo, Config, profiling
│  ├─ types.h                    # ✨ NEW: Data types
│  └─ config.h                   # ✨ NEW: Build config
├─ cmake/
│  └─ KernelSmithConfig.cmake    # ✨ NEW: CMake integration
│                                #   find_package(KernelSmith)
└─ kernelsmith-0.1.0-riscv64-rvv.tar.gz  # ✨ NEW: Distribution package
```

### Now You CAN Do This

```cpp
// In external project:
#include <kernelsmith/kernels.h>
#include <kernelsmith/runtime.h>

kernelsmith::init({.vlen_bits = 256});
kernelsmith::matmul(A, B, C, M, N, K);  // ✅ Works!
                                        // ✅ Compiled RISC-V code
                                        // ✅ Linked against libks-kernels.a
```

---

## The Compilation Pipeline

### From High-Level Kernel to Library

```
Stage 1: KernelSmith Kernel Definition
├─ Input: ks.matmul operation
├─ Tool: KernelSmith dialect
└─ Output: MLIR IR

Stage 2: Lowering to Linalg
├─ Pass: ks-lower-to-linalg
├─ Tool: ks-opt
└─ Output: linalg.matmul + loops

Stage 3: Tiling for Cache Efficiency
├─ Pass: ks-tile (tile_m=64, tile_n=64, tile_k=32)
├─ Tool: linalg.tiling pass
└─ Output: Nested scf.for loops with smaller matmuls

Stage 4: Vectorization
├─ Pass: ks-vectorize
├─ Operation: Convert scalar loops to vector operations
└─ Output: vector.load, vector.fma, vector.store

Stage 5: RVV Lowering (DES-001)
├─ Pass: ks-lower-to-rvv
├─ Mapping: vector.load → vle32.v, vector.fma → vfmacc.vv
└─ Output: LLVM IR with RVV intrinsics

Stage 6: LLVM IR to RISC-V Assembly
├─ Tool: llvm-translate, llc
├─ Target: riscv64 with RVV extension
└─ Output: matmul.s (RISC-V assembly text)

Stage 7: Assembly to Object File
├─ Tool: riscv64-unknown-elf-as
├─ Input: matmul.s
└─ Output: matmul.o (RISC-V machine code)

Stage 8: Link Objects to Library
├─ Objects: matmul.o, conv2d.o, attention.o, ...
├─ Tool: ar rcs libks-kernels.a
└─ Output: libks-kernels.a (static library)

Stage 9: Install with Headers
├─ Library: libks-kernels.a → /usr/local/lib/
├─ Headers: kernels.h, runtime.h → /usr/local/include/kernelsmith/
└─ CMake Config: KernelSmithConfig.cmake → /usr/local/share/cmake/

Stage 10: Use in External Project
├─ find_package(KernelSmith)
├─ target_link_libraries(myapp kernelsmith::kernelsmith)
└─ Call: kernelsmith::matmul(A, B, C, M, N, K)
```

---

## File Structure: Before vs. After

### Before (Current v0.1.0)
```
/home/user/ulir-kernels/
├─ lib/
│  ├─ Dialect/
│  │  └─ Kernel/           # MLIR dialect definitions
│  │     ├─ KernelOps.cpp
│  │     └─ KernelDialect.cpp
│  └─ Passes/              # Lowering passes (framework code)
│     └─ PassRegistration.cpp
├─ include/
│  └─ KernelSmith/
│     └─ Dialect/
│        └─ Kernel/        # MLIR headers only
├─ tools/
│  └─ ks-opt/              # CLI tool
└─ specs/
   └─ kernels/             # Specifications (not code)
       ├─ matmul.md
       ├─ conv2d.md
       └─ attention.md
```

### After (Desired v1.0.0)
```
/home/user/ulir-kernels/
├─ lib/
│  ├─ Dialect/             # (existing) MLIR framework
│  ├─ Passes/              # (existing) Lowering infrastructure
│  └─ Kernels/             # ✨ NEW: Compiled kernel implementations
│     ├─ CMakeLists.txt    # Compilation pipeline
│     ├─ matmul.cpp        # MatMul wrapper + helpers
│     ├─ conv2d.cpp        # Conv2D wrapper + helpers
│     ├─ attention.cpp     # Attention wrapper + helpers
│     ├─ activations.cpp   # Activation functions
│     ├─ normalizations.cpp# Normalization functions
│     └─ reductions.cpp    # Reduction functions
├─ include/
│  ├─ KernelSmith/
│  │  └─ Dialect/Kernel/   # (existing) MLIR headers
│  └─ kernelsmith/         # ✨ NEW: Public C/C++ API
│     ├─ kernels.h         # Kernel declarations
│     ├─ matmul.h          # MatMul specifics (optional)
│     ├─ runtime.h         # Runtime config
│     ├─ types.h           # Data types
│     └─ config.h          # Build config
├─ tools/
│  ├─ ks-opt/              # (existing) CLI tool
│  └─ generate-kernels.sh  # ✨ NEW: Kernel generation script
├─ specs/
│  └─ kernels/             # (existing) Specifications
├─ docs/
│  ├─ design/
│  │  ├─ DES-001-005       # (existing) Design docs
│  │  └─ DES-005-library-packaging.md  # Packaging design
│  └─ guides/
│     ├─ testing-guide.md  # (existing)
│     └─ using-library.md  # ✨ NEW: How to use the library
└─ build/
   ├─ lib/
   │  ├─ libks-kernels.a       # ✨ Generated: Compiled kernels
   │  └─ libks-runtime.a       # ✨ Generated: Runtime
   ├─ include/kernelsmith/     # ✨ Generated: Headers
   └─ cmake/                   # ✨ Generated: CMake configs
```

---

## CMake Build System Changes

### New Targets

```makefile
# In top-level CMakeLists.txt

add_library(ks-kernels STATIC)           # Compiled kernel objects
add_library(ks-runtime STATIC)           # Runtime library
add_library(kernelsmith INTERFACE)       # Public interface

# Installation
install(TARGETS ks-kernels ks-runtime kernelsmith
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib
  RUNTIME DESTINATION bin)

install(DIRECTORY include/kernelsmith
  DESTINATION include)

install(FILES
  "${CMAKE_CURRENT_BINARY_DIR}/KernelSmithConfig.cmake"
  DESTINATION lib/cmake/KernelSmith)

# Packaging
include(CPack)
cpack_add_component(libraries ...)
cpack_add_component(headers ...)
```

### New Makefile Targets

```makefile
# In Makefile

build-kernels:           # Compile MLIR kernels to RISC-V
build-library:           # Link to static libraries
package:                 # Create distribution tarball
install-library:         # Install to system paths
verify-library:          # Check installation
```

---

## Example: How It Works

### User Project: matmul_app

```
user-project/
├─ CMakeLists.txt
├─ main.cpp
└─ data.bin
```

### CMakeLists.txt

```cmake
cmake_minimum_required(VERSION 3.20)
project(matmul_app)

# Find KernelSmith library
find_package(KernelSmith 0.1.0 REQUIRED)

# Create executable
add_executable(matmul_app main.cpp)

# Link against KernelSmith
target_link_libraries(matmul_app PRIVATE
  KernelSmith::kernelsmith  # This is libks-kernels.a + libks-runtime.a
)
```

### main.cpp

```cpp
#include <kernelsmith/kernels.h>
#include <kernelsmith/runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize runtime
    kernelsmith::init({.vlen_bits = 256, .enable_profiling = true});

    // Create test matrices
    size_t M = 256, K = 512, N = 1024;
    std::vector<float> A(M * K), B(K * N), C(M * N);

    // Fill with data (omitted for brevity)
    for (auto& x : A) x = 1.0f;
    for (auto& x : B) x = 2.0f;

    // Call optimized RISC-V kernel
    std::cout << "Computing " << M << "x" << N << " MatMul...\n";
    kernelsmith::matmul(A.data(), B.data(), C.data(), M, N, K);
    std::cout << "Done!\n";

    // Query runtime info
    auto info = kernelsmith::get_runtime_info();
    std::cout << "Running on: " << info.target << "\n";
    std::cout << "VLEN: " << info.vlen_bits << " bits\n";

    kernelsmith::shutdown();
    return 0;
}
```

### Build and Run

```bash
# User downloads and installs KernelSmith
wget kernelsmith-0.1.0-riscv64-rvv.tar.gz
tar xzf kernelsmith-0.1.0-riscv64-rvv.tar.gz
export KernelSmith_DIR=$PWD/kernelsmith/lib/cmake/KernelSmith

# Build user app
mkdir build && cd build
cmake ..  # find_package(KernelSmith) now works
make

# Run
./matmul_app
# Output:
# Computing 256x1024 MatMul...
# Done!
# Running on: riscv64-rvv
# VLEN: 256 bits
```

---

## Comparison: MLIR vs. Library Output

### ❌ Current (MLIR Only)

| What | Value |
|-----|-------|
| Can generate MLIR IR? | ✅ Yes |
| Can view transformations? | ✅ Yes (human readable) |
| Can be used by C/C++ projects? | ❌ No |
| Can be installed system-wide? | ❌ No |
| Can be distributed? | ❌ No (just source) |
| Can call kernels? | ❌ No API |
| Performance benchmarks? | ⏳ No (not compiled) |

### ✅ Desired (Library Output)

| What | Value |
|-----|-------|
| Can generate MLIR IR? | ✅ Yes |
| Can view transformations? | ✅ Yes (ks-opt tool) |
| Can be used by C/C++ projects? | ✅ Yes (headers + libs) |
| Can be installed system-wide? | ✅ Yes (CMake package) |
| Can be distributed? | ✅ Yes (tarball/rpm/deb) |
| Can call kernels? | ✅ Yes (C/C++ API) |
| Performance benchmarks? | ✅ Yes (compiled) |

---

## Implementation Timeline

### Phase 1 (✅ Complete)
- Test infrastructure
- Design documentation

### Phase 2-3 (Next)
- Implement vector operations lowering (DES-001)
- Implement MatMul (DES-002)
- Generate RISC-V assembly

### Phase 4 (New - Add Based on DES-005)
- **Wrapper code** (C/C++ API)
  - Create lib/Kernels/wrapper.cpp
  - Link compiled objects to .a files
- **Compilation pipeline**
  - MLIR → Assembly → Objects → Library
  - Scripts and CMake for automation
- **Headers and CMake config**
  - Install public headers
  - Create KernelSmithConfig.cmake

### Phase 5
- Packaging and distribution
- Examples and documentation

---

## What Changes for Users

### Installation

```bash
# Before (v0.1.0): Can't really install
$ make install
# No public headers or libraries to install

# After (v1.0.0): Standard installation
$ make package
$ sudo dpkg -i kernelsmith-0.1.0-riscv64-rvv.deb
$ export KernelSmith_DIR=/usr/share/cmake/KernelSmith
# Done! Can now use in any project
```

### In Their Code

```cpp
// Before (v0.1.0): Can't use
#include <kernelsmith/kernels.h>  // ❌ Doesn't exist

// After (v1.0.0): Standard usage
#include <kernelsmith/kernels.h>  // ✅ Available
kernelsmith::matmul(...);          // ✅ Works
```

### Build System

```cmake
# Before (v0.1.0): Manual linking
target_link_directories(myapp PRIVATE /path/to/kernelsmith/lib)
target_include_directories(myapp PRIVATE /path/to/kernelsmith/include)
target_link_libraries(myapp PRIVATE ks-kernels ks-runtime)

# After (v1.0.0): Standard CMake
find_package(KernelSmith REQUIRED)
target_link_libraries(myapp PRIVATE KernelSmith::kernelsmith)
```

---

## Key Deliverables Summary

### For v0.1.0 (Current + Design)
- ✅ MLIR dialect framework
- ✅ Lowering passes infrastructure
- ✅ Test framework
- ✅ Design documentation (DES-001 through DES-005)

### For v1.0.0 (Production Ready)
- ✅ Compiled kernel libraries (.a)
- ✅ Public C/C++ API (headers)
- ✅ Runtime utilities
- ✅ CMake package config
- ✅ Distribution packages
- ✅ Examples and docs

---

## Next Steps

1. **Implement DES-001** (Vector operations lowering)
2. **Implement DES-002** (MatMul kernel)
3. **Create wrapper code** (following DES-005)
4. **Set up compilation pipeline** (CMake targets)
5. **Build and test libraries** (make build-kernels)
6. **Create distribution** (make package)
7. **Install and verify** (make install-library)

---

## Reference

- **Current Design**: DES-001 through DES-004
- **Library Design**: DES-005-library-packaging.md (this file's source)
- **Test Infrastructure**: Phase 1 (complete)
- **Development Plan**: RISC-V_RVV_KERNEL_LIBRARY_PLAN.md

