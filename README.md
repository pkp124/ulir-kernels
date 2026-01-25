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

Compile to RISC-V RVV:
```bash
ks-opt input.mlir \
  --ks-lower-to-linalg \
  --ks-tile \
  --ks-vectorize \
  --ks-lower-to-rvv \
  --convert-to-llvm | \
llc -march=riscv64 -mattr=+v -o output.s
```

## Project Structure

```
.
├── .cursor/rules/       # Cursor AI development rules
├── .agents/             # Agent skills and workflows
├── docs/                # Documentation
│   ├── architecture/    # Architecture design docs
│   ├── design/          # Design decisions and reviews
│   └── guides/          # User and developer guides
├── specs/               # Feature specifications
│   ├── kernels/         # Kernel operation specs
│   └── targets/         # Target architecture specs
├── include/KernelSmith/ # Public headers
│   ├── Dialect/Kernel/  # Kernel dialect definitions
│   ├── Passes/          # Pass declarations
│   └── Targets/         # Target-specific headers
├── lib/                 # Implementation
│   ├── Dialect/         # Dialect implementations
│   ├── Passes/          # Pass implementations
│   └── Targets/         # Target backends
├── tools/               # CLI tools
│   └── ks-opt/          # KernelSmith optimizer
├── tests/               # Test suites
│   ├── lit/             # MLIR FileCheck tests
│   └── unit/            # C++ unit tests
├── examples/            # Example kernels
└── tasks/               # Development task tracking
```

## Development Workflow

KernelSmith uses a rigorous development process:

1. **Specification**: Define behavior in `specs/`
2. **Design Review**: Document decisions in `docs/design/`
3. **TDD**: Write tests before implementation
4. **Implementation**: Incremental, reviewed changes
5. **Verification**: `ctest` for all tests

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Supported Kernels

| Kernel | Description | Status |
|--------|-------------|--------|
| `ks.matmul` | Matrix multiplication | Planned |
| `ks.batch_matmul` | Batched matrix multiplication | Planned |
| `ks.conv2d` | 2D convolution | Planned |
| `ks.attention` | Scaled dot-product attention | Planned |
| `ks.softmax` | Softmax activation | Planned |
| `ks.layer_norm` | Layer normalization | Planned |
| `ks.gelu` | GELU activation | Planned |
| `ks.relu` | ReLU activation | Planned |

## License

GNU Affero General Public License v3.0 - see [LICENSE](LICENSE).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
