# KernelSmith

*Optimized ML inference kernels for edge, embedded, and physical AI*

## Overview

KernelSmith is an MLIR-based compiler framework that generates optimized ML inference kernels for resource-constrained hardware. It ships as a C kernel library (`libkernelsmith.a` + headers) with no runtime dependencies — works with any toolchain, any RTOS.

- **RISC-V RVV** (Vector Extension) — Primary target
- **ARM NEON** (ARMv8-A) — Secondary target
- **Quantization** (INT8, INT4/W4A8) — Core feature

The MVP target is a quantized transformer running end-to-end on RISC-V RVV.

## Features

- **High-Level Kernel Operations**: Matrix multiplication, convolution, attention, activation functions, normalization
- **Quantization-First**: INT8 and INT4 kernels with accumulator promotion for edge inference
- **Profile-Driven Tiling**: Target profiles drive tile sizes, packing, and vectorization at build time
- **RVV-Specific Lowering**: Custom `--ks-lower-to-rvv` pass for vector-length-agnostic code
- **Comprehensive Testing**: Lit, unit, and numpy-validated C tests with CTest

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

See [CLAUDE.md](CLAUDE.md) for development guidelines and [ROADMAP.md](ROADMAP.md) for the milestone plan.

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
