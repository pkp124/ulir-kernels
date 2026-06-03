# KernelSmith

*Optimized ML inference kernels for edge, embedded, and physical AI*

## Overview

KernelSmith is an MLIR-based compiler framework that generates optimized ML inference kernels for resource-constrained hardware. It ships as a C kernel library (`libkernelsmith.a` + headers) with no runtime dependencies — works with any toolchain, any RTOS.

- **RISC-V RVV** (Vector Extension) — Primary target
- **ARM NEON** (ARMv8-A) — Secondary target
- **Quantization** (INT8, INT4/W4A8) — Core feature

The MVP target is a quantized transformer running end-to-end on RISC-V RVV.

## Current Features

- **High-Level Kernel Operations**: Matrix multiplication, convolution, attention, activation functions, normalization
- **C Kernel Library**: `libkernelsmith.a` with stable C headers and handwritten f32 reference kernels
- **MLIR Lowering Passes**: Activations lower to `linalg.generic`; matmul lowers through linalg, tiling, packing, vectorization, and RVV/LLVM paths
- **Profile-Driven Tiling**: Generic target tile sizes are available as `--ks-tile` pass options
- **Comprehensive Testing**: Lit, unit, and numpy-validated C tests with CTest

Planned work includes stronger RVV end-to-end coverage, INT8 and INT4/W4A8 quantized kernels,
and broader edge operator coverage.

## Quick Start

### Prerequisites

- LLVM/MLIR 21+ (with MLIR enabled)
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
ctest --test-dir build --output-on-failure
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

Parse, print, and run the currently implemented lowering passes:
```bash
ks-opt input.mlir
ks-opt input.mlir --ks-lower-to-linalg --ks-tile
```

## Project Structure

```
.
├── include/KernelSmith/ # Public headers and TableGen
│   └── Dialect/Kernel/  # Kernel dialect definitions (.td, .h)
├── lib/                 # Implementation
│   ├── Dialect/Kernel/  # Dialect, ops, types (.cpp)
│   └── Passes/          # Pass implementations
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
| `ks.matmul` | Yes | Yes | linalg + tile |
| `ks.batch_matmul` | Yes | Yes | Not yet |
| `ks.conv2d` | Yes | Yes | Not yet |
| `ks.attention` | Yes | Yes | Not yet |
| `ks.softmax` | Yes | Yes | Not yet |
| `ks.layer_norm` | Yes | Yes | Not yet |
| `ks.rms_norm` | Yes | Yes | Not yet |
| `ks.gelu` | Yes | - | activation pass |
| `ks.relu` | Yes | - | activation pass |
| `ks.silu` | Yes | - | activation pass |
| `ks.reduce_sum` | Yes | Yes | Not yet |
| `ks.reduce_max` | Yes | Yes | Not yet |

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE).
