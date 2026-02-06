# Kernel Library Packaging Architecture

## Current vs. Desired Output

### Current Output (MLIR Compiler)
```
ks.matmul (IR) → ks-opt tool → LLVM IR → Assembly
                                          ↑
                                    Used internally only
                                    Not packaged for external use
```

### Desired Output (Usable Library)
```
ks.matmul (IR)
  ↓
[Compilation Pipeline]
  ├─ Lower to LLVM IR
  ├─ Generate RISC-V ASM
  ├─ Compile to object files (.o)
  ├─ Link to static library (.a)
  └─ Install with headers
    ↓
Distributable Artifacts
  ├─ libks-kernels.a (compiled kernels)
  ├─ libks-runtime.a (runtime support)
  ├─ include/kernelsmith/*.h (C/C++ interfaces)
  └─ include/kernelsmith/runtime.h (runtime API)
    ↓
Used by External Projects
  └─ #include <kernelsmith/kernels.h>
     kernelsmith::matmul(A, B, C, M, N, K);
```

---

## Library Structure

### Output Artifacts

```
build/lib/
├─ libks-kernels.a          # Compiled kernel implementations (RISC-V)
├─ libks-runtime.a          # Runtime library (memory, utilities)
└─ cmake/
   └─ KernelSmithConfig.cmake  # CMake package config

install/
├─ lib/
│  ├─ libks-kernels.a
│  └─ libks-runtime.a
├─ include/kernelsmith/
│  ├─ kernels.h              # Public kernel API
│  ├─ matmul.h               # MatMul kernel header
│  ├─ conv2d.h               # Conv2D kernel header
│  ├─ attention.h            # Attention kernel header
│  ├─ activations.h          # Activation functions
│  ├─ runtime.h              # Runtime utilities
│  ├─ types.h                # Type definitions
│  └─ config.h               # Build configuration
└─ cmake/
   └─ KernelSmithConfig.cmake
```

---

## C/C++ Public API Design

### Header: `include/kernelsmith/kernels.h`

```cpp
#pragma once

#include "kernelsmith/types.h"
#include <cstddef>

namespace kernelsmith {

// MatMul: C = A @ B
// A: shape (M, K), B: shape (K, N), C: shape (M, N)
// All row-major layout
void matmul(
    const float* A,      // Input matrix A
    const float* B,      // Input matrix B
    float* C,            // Output matrix C (preallocated)
    size_t M,            // Rows of A
    size_t N,            // Columns of B
    size_t K             // Shared dimension
);

// MatMul with different dtypes
void matmul_f16(
    const float16_t* A,
    const float16_t* B,
    float16_t* C,
    size_t M, size_t N, size_t K
);

void matmul_i8(
    const int8_t* A,
    const int8_t* B,
    int32_t* C,          // Output in higher precision
    size_t M, size_t N, size_t K
);

// Conv2D: Output = Conv(Input, Kernel)
// Input: (batch, height, width, in_channels)  [NHWC format]
// Kernel: (k_h, k_w, in_channels, out_channels)
// Output: (batch, out_height, out_width, out_channels)
void conv2d(
    const float* input,
    const float* kernel,
    float* output,
    size_t batch, size_t in_h, size_t in_w, size_t in_c,
    size_t k_h, size_t k_w, size_t out_c,
    size_t stride_h, size_t stride_w,
    size_t pad_h, size_t pad_w
);

// Attention: Output = SDPA(Q, K, V)
// Q, K, V: (batch, seq_len, embed_dim)
// Output: (batch, seq_len, embed_dim)
void attention(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    size_t batch,
    size_t seq_len,
    size_t embed_dim,
    bool causal_mask = false
);

// Activation functions
void relu(const float* input, float* output, size_t numel);
void gelu(const float* input, float* output, size_t numel);
void silu(const float* input, float* output, size_t numel);

// Normalization
void layer_norm(
    const float* input,
    float* output,
    size_t batch, size_t seq_len, size_t embed_dim,
    const float* gamma,   // Scale parameters
    const float* beta     // Bias parameters
);

// Reductions
void reduce_sum(
    const float* input,
    float* output,
    size_t numel
);

void reduce_max(
    const float* input,
    float* output,
    size_t numel
);

}  // namespace kernelsmith
```

### Header: `include/kernelsmith/runtime.h`

```cpp
#pragma once

#include <cstddef>
#include <cstdint>

namespace kernelsmith {

// Runtime configuration
struct Config {
    size_t vlen_bits;      // Vector length (128, 256, 512)
    bool enable_profiling; // Enable timing measurements
    bool enable_validation; // Validate results
};

// Initialize runtime
void init(const Config& config);

// Cleanup
void shutdown();

// Get runtime information
struct RuntimeInfo {
    size_t vlen_bits;      // Detected VLEN
    bool has_rvv;          // Has RVV support
    const char* target;    // "riscv64-rvv"
};

RuntimeInfo get_runtime_info();

// Performance profiling
struct ProfileResult {
    const char* kernel_name;
    double time_ms;
    size_t operations;     // FLOPs or operations count
};

void enable_profiling();
void disable_profiling();
ProfileResult* get_profile_results(size_t& count);

}  // namespace kernelsmith
```

### Header: `include/kernelsmith/types.h`

```cpp
#pragma once

#include <cstdint>

namespace kernelsmith {

// Floating point types
typedef float float32_t;
typedef struct {
    uint16_t value;
} float16_t;

// Integer types
typedef int8_t int8_t;
typedef int16_t int16_t;
typedef int32_t int32_t;

// Data layout
enum class Layout {
    RowMajor,              // C-style (default)
    ColMajor               // Fortran-style
};

// Data type enumeration
enum class DataType {
    F32,
    F16,
    I8,
    I32
};

}  // namespace kernelsmith
```

---

## Compilation Pipeline

### From MLIR to Library

```cpp
// Conceptual pipeline
stage 1: MLIR Kernel IR (ks.matmul, ks.conv2d, ...)
           ↓ [ks-lower-to-linalg]
stage 2: Linalg IR (linalg.matmul + loops)
           ↓ [ks-tile]
stage 3: Tiled IR (scf.for loops)
           ↓ [ks-vectorize]
stage 4: Vector IR (vector.load, vector.fma, ...)
           ↓ [ks-lower-to-rvv] (DES-001)
stage 5: LLVM RVV IR (llvm.call @llvm.riscv.*)
           ↓ [llvm-translate]
stage 6: RISC-V Assembly (ASM text)
           ↓ [riscv64-unknown-elf-as]
stage 7: Object Files (.o)
           ↓ [ar rcs libks-kernels.a *.o]
stage 8: Static Library (.a)
```

### CMake Build System

```cmake
# lib/Kernels/CMakeLists.txt

# For each kernel, generate MLIR → object file

add_custom_command(
    OUTPUT matmul.o
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/generate_kernel.sh matmul
      --output-mlir matmul.mlir
    COMMAND ks-opt matmul.mlir
      --ks-lower-to-linalg
      --ks-tile="tile_m=64,tile_n=64,tile_k=32"
      --ks-vectorize
      --ks-lower-to-rvv
      --convert-to-llvm
      -o matmul.llvm.mlir
    COMMAND llvm-translate -mlir-to-llvmir matmul.llvm.mlir -o matmul.llvm.ir
    COMMAND llc -march=riscv64 -mattr=+v matmul.llvm.ir -o matmul.s
    COMMAND riscv64-unknown-elf-as matmul.s -o matmul.o
    DEPENDS ${KERNEL_SOURCE_FILES}
)

# Create library
add_library(ks-kernels STATIC matmul.o conv2d.o attention.o ...)

# Wrapper library with C++ API
add_library(ks-kernels-cpp OBJECT
    src/wrapper.cpp      # Wraps kernel objects with C++ interface
)

target_link_libraries(ks-kernels-cpp PUBLIC ks-kernels)

# Runtime utilities library
add_library(ks-runtime STATIC
    src/runtime.cpp
    src/profiling.cpp
    src/config.cpp
)

# Combined public library
add_library(kernelsmith INTERFACE)
target_link_libraries(kernelsmith INTERFACE
    ks-kernels
    ks-kernels-cpp
    ks-runtime
)

# Installation
install(TARGETS kernelsmith ks-kernels ks-runtime
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib)

install(FILES
    ../include/kernelsmith/kernels.h
    ../include/kernelsmith/runtime.h
    ../include/kernelsmith/types.h
    DESTINATION include/kernelsmith)
```

---

## Example Usage

### User Code (External Project)

```cpp
// main.cpp - Using KernelSmith library

#include <kernelsmith/kernels.h>
#include <kernelsmith/runtime.h>
#include <iostream>
#include <vector>

int main() {
    // Initialize runtime
    kernelsmith::Config config{
        .vlen_bits = 256,
        .enable_profiling = true,
        .enable_validation = true
    };
    kernelsmith::init(config);

    // Create test matrices
    size_t M = 64, N = 64, K = 128;
    std::vector<float> A(M * K, 1.0f);
    std::vector<float> B(K * N, 2.0f);
    std::vector<float> C(M * N, 0.0f);

    // Call optimized kernel
    kernelsmith::matmul(A.data(), B.data(), C.data(), M, N, K);

    // Get runtime info
    auto info = kernelsmith::get_runtime_info();
    std::cout << "Running on: " << info.target << "\n";
    std::cout << "VLEN: " << info.vlen_bits << " bits\n";

    // Cleanup
    kernelsmith::shutdown();

    return 0;
}
```

### CMakeLists.txt for User Project

```cmake
cmake_minimum_required(VERSION 3.20)
project(myapp)

# Find KernelSmith
find_package(KernelSmith REQUIRED)

# Create executable
add_executable(myapp main.cpp)

# Link against KernelSmith
target_link_libraries(myapp PRIVATE KernelSmith::kernelsmith)
```

### Build and Run

```bash
# Build user project
mkdir build && cd build
cmake ..
make

# Run
./myapp
# Output:
# Running on: riscv64-rvv
# VLEN: 256 bits
```

---

## Distribution Package

### Tarball Contents

```
kernelsmith-0.1.0-riscv64-rvv.tar.gz
├─ lib/
│  ├─ libks-kernels.a (compiled RISC-V kernels)
│  ├─ libks-runtime.a (runtime utilities)
│  └─ libkernelsmith.a (combined)
├─ include/kernelsmith/
│  ├─ kernels.h
│  ├─ runtime.h
│  ├─ types.h
│  └─ config.h
├─ share/cmake/KernelSmith/
│  ├─ KernelSmithConfig.cmake
│  └─ KernelSmithTargets.cmake
├─ doc/
│  ├─ API.md
│  ├─ PERFORMANCE.md
│  └─ EXAMPLES.md
└─ examples/
   ├─ matmul.cpp
   ├─ conv2d.cpp
   └─ CMakeLists.txt
```

### Installation for System-wide Use

```bash
# Extract
tar xzf kernelsmith-0.1.0-riscv64-rvv.tar.gz

# Install
mkdir -p /opt/kernelsmith
cp -r kernelsmith-0.1.0/* /opt/kernelsmith/

# Set environment
export KernelSmith_DIR=/opt/kernelsmith/share/cmake/KernelSmith
export LD_LIBRARY_PATH=/opt/kernelsmith/lib:$LD_LIBRARY_PATH

# Now CMake can find it
find_package(KernelSmith REQUIRED)
```

---

## Build Targets

### Makefile Additions

```makefile
# Add to Makefile

.PHONY: build-kernels package install-package

build-kernels: build
	@echo "Compiling MLIR kernels to RISC-V..."
	@cmake --build $(BUILD_DIR) --target ks-kernels
	@cmake --build $(BUILD_DIR) --target ks-runtime

package: build-kernels
	@echo "Creating distribution package..."
	@cmake --build $(BUILD_DIR) --target package
	@ls -lh $(BUILD_DIR)/kernelsmith-*.tar.gz

install-package: package
	@echo "Installing to system..."
	@mkdir -p /opt/kernelsmith
	@tar xzf $(BUILD_DIR)/kernelsmith-*.tar.gz -C /opt/kernelsmith

verify-install:
	@echo "Verifying installation..."
	@ls -l /opt/kernelsmith/lib/libks-*.a
	@ls -l /opt/kernelsmith/include/kernelsmith/*.h
	@echo "✓ KernelSmith installed successfully"
```

---

## Current vs. Future

### Current Output (v0.1.0)
- ❌ No compiled kernels
- ❌ No public C/C++ API
- ❌ No runtime library
- ✅ MLIR dialect and passes
- ✅ ks-opt compiler tool

### Future Output (v1.0.0)
- ✅ Compiled kernel library (libks-kernels.a)
- ✅ Runtime library (libks-runtime.a)
- ✅ Public C/C++ headers (kernelsmith/*.h)
- ✅ CMake package (KernelSmithConfig.cmake)
- ✅ Distribution packages (tarball, rpm, deb)
- ✅ Examples and documentation

---

## Dependencies for Compilation Pipeline

1. **LLVM/MLIR 18+**: Dialect definitions, lowering framework
2. **RISC-V GNU Toolchain**: riscv64-unknown-elf-as, riscv64-unknown-elf-gcc
3. **QEMU (optional)**: For testing compiled binaries
4. **CMake 3.20+**: Build system

---

## Implementation Phases

### Phase 1 (Current): ✅ Test Infrastructure & Design
- Design documents complete
- No library output yet

### Phase 2-3: Implementation (Vector Ops + MatMul)
- Generate MLIR kernels
- Lower to RISC-V
- Create wrapper code

### Phase 4: Library Packaging
- Compile RISC-V kernels to objects
- Link to libraries
- Install headers
- Create CMake configs

### Phase 5: Distribution
- Create packages (tarball, rpm, deb)
- Documentation and examples
- Performance benchmarks

---

## Key Design Decisions

1. **Static Linking**: Using .a files (static libraries) for:
   - No runtime dependency issues
   - Better performance for embedded systems
   - Easy distribution

2. **C API + C++ Wrappers**:
   - Core functions in C (stable ABI)
   - C++ wrappers for convenience
   - Interop with other languages

3. **Row-Major Layout** (default):
   - Compatible with most frameworks
   - Can add ColMajor variant later

4. **NHWC Format** (Conv2D):
   - Better for vectorization
   - Most RVV-optimized implementations use this

5. **Multi-precision Support**:
   - f32, f16, i8 with auto-promotion
   - Extensible for future types

