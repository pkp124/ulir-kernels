# KernelSmith

*Craft optimized AI kernels for any architecture*

## Overview

KernelSmith is a framework for generating high-performance AI accelerator kernels using the MLIR compiler infrastructure. It provides a collection of optimized kernels targeting multiple architectures, with a focus on:

- **RISC-V RVV** (Vector Extension) - Primary target
- Future: ARM SVE, x86 AVX-512, GPU backends

Kernels are crafted through a multi-level lowering process using MLIR, enabling both portability and architecture-specific optimizations.

## Features

- **High-Level Kernel Operations**: Matrix multiplication, convolution, attention, activation functions
- **Automatic Tiling**: Configurable tiling strategies for cache and register optimization
- **Target-Specific Lowering**: Optimized code generation for each architecture
- **Comprehensive Testing**: Unit, integration, and lit tests with CTest

## Quick Start

### Prerequisites

- LLVM/MLIR 18+ (with MLIR enabled)
- CMake 3.20+
- Python 3.10+
- C++17 compatible compiler

### Build

```bash
# Configure
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir

# Build
cmake --build build --parallel

# Test
ctest --test-dir build
```

### Basic Usage

```mlir
// Define a matrix multiplication kernel
func.func @matmul_kernel(%A: tensor<64x128xf32>, 
                         %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
```

Parse and print (lowering passes are not yet implemented):
```bash
ks-opt input.mlir
```

## Project Structure

```
.
├── include/KernelSmith/ # Public headers and TableGen
│   └── Dialect/Kernel/  # Kernel dialect definitions (.td, .h)
├── lib/                 # Implementation
│   ├── Dialect/Kernel/  # Dialect, ops, types (.cpp)
│   └── Passes/          # Pass implementations (stub)
├── tools/ks-opt/        # CLI optimizer entry point
├── tests/
│   ├── lit/             # MLIR FileCheck tests (.mlir)
│   └── unit/            # C++ unit tests (Google Test)
├── specs/               # Feature specifications
├── docs/design/         # Design decisions (DES-XXX format)
├── tasks/               # Development task tracking
└── examples/            # Example kernels
```

## Development Workflow

KernelSmith uses a rigorous development process:

1. **Specification**: Define behavior in `specs/`
2. **Design Review**: Document decisions in `docs/design/`
3. **TDD**: Write tests before implementation
4. **Implementation**: Incremental, reviewed changes
5. **Verification**: `ctest` for all tests

See [CLAUDE.md](CLAUDE.md) for development guidelines.

## Supported Kernels

| Kernel | Parse/Print | Verifier | Lowering |
|--------|:-----------:|:--------:|:--------:|
| `ks.matmul` | Yes | Yes | Not yet |
| `ks.batch_matmul` | Yes | Yes | Not yet |
| `ks.conv2d` | Yes | Yes | Not yet |
| `ks.attention` | Yes | Yes | Not yet |
| `ks.softmax` | Yes | - | Not yet |
| `ks.layer_norm` | Yes | Yes | Not yet |
| `ks.rms_norm` | Yes | - | Not yet |
| `ks.gelu` | Yes | - | Not yet |
| `ks.relu` | Yes | - | Not yet |
| `ks.silu` | Yes | - | Not yet |
| `ks.reduce_sum` | Yes | - | Not yet |
| `ks.reduce_max` | Yes | - | Not yet |

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE).
