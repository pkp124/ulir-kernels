# MLIR Kernel Generation

A framework for generating optimized AI accelerator kernels using the MLIR compiler infrastructure.

## Overview

This project provides a collection of high-performance kernels for AI workloads, targeting multiple architectures with a focus on:

- **RISC-V RVV** (Vector Extension) - Primary target
- Future: ARM SVE, x86 AVX-512, GPU backends

The kernels are generated through a multi-level lowering process using MLIR, enabling both portability and architecture-specific optimizations.

## Features

- **High-Level Kernel Operations**: Matrix multiplication, convolution, attention, activation functions
- **Automatic Tiling**: Configurable tiling strategies for cache and register optimization
- **Target-Specific Lowering**: Optimized code generation for each architecture
- **Python Bindings**: Easy integration with ML frameworks
- **Comprehensive Testing**: Unit, integration, and performance tests

## Quick Start

### Prerequisites

- LLVM/MLIR 18+ (with MLIR enabled)
- CMake 3.20+
- Python 3.10+
- C++17 compatible compiler

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd mlir-kernel-generation

# Run setup script
make setup

# Build the project
make build

# Run tests
make test
```

### Basic Usage

```mlir
// Define a matrix multiplication kernel
func.func @matmul_kernel(%A: tensor<64x128xf32>, 
                         %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
  %C = kernel.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                               -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
```

Compile to RISC-V RVV:
```bash
aikernel-opt input.mlir \
  --tile-kernels \
  --lower-to-vector \
  --lower-to-rvv \
  --convert-to-llvm | \
llc -march=riscv64 -mattr=+v -o output.s
```

## Project Structure

```
.
├── .cursor/rules/       # Cursor AI rules for development
├── docs/                # Documentation
│   ├── architecture/    # Architecture design docs
│   ├── guides/          # User and developer guides
│   └── api/             # API reference
├── specs/               # Feature specifications
│   ├── kernels/         # Kernel operation specs
│   └── targets/         # Target architecture specs
├── src/                 # Source code
│   ├── dialects/        # MLIR dialect definitions
│   ├── passes/          # Transformation passes
│   ├── targets/         # Target-specific lowering
│   └── runtime/         # Runtime library
├── tests/               # Test suites
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── lit/             # MLIR FileCheck tests
├── examples/            # Example kernels and usage
├── tasks/               # Development task tracking
└── scripts/             # Development scripts
```

## Development with Cursor

This project is optimized for development with Cursor AI. The `.cursor/rules/` directory contains context rules that help the AI understand:

- Project architecture and conventions
- MLIR-specific patterns and best practices
- Target-specific (RVV) development guidelines
- Testing requirements

### Key Commands

```bash
make help              # Show all available commands
make verify            # Run full verification (lint + test + build)
make new-kernel NAME=softmax  # Create new kernel from template
make new-pass NAME=tile-conv  # Create new pass from template
make new-task ID=001 TITLE="My task"  # Create new task
```

## Documentation

- [Architecture Overview](docs/architecture/overview.md)
- [Getting Started Guide](docs/guides/getting-started.md)
- [Adding New Kernels](docs/guides/adding-kernels.md)
- [Target Development](docs/guides/target-development.md)
- [Contributing](CONTRIBUTING.md)

## Supported Kernels

| Kernel | Description | RVV |
|--------|-------------|-----|
| `matmul` | Matrix multiplication | ⏳ |
| `batch_matmul` | Batched matrix multiplication | ⏳ |
| `conv2d` | 2D convolution | ⏳ |
| `attention` | Scaled dot-product attention | ⏳ |
| `softmax` | Softmax activation | ⏳ |
| `layer_norm` | Layer normalization | ⏳ |
| `gelu` | GELU activation | ⏳ |

Legend: ✅ Complete | ⏳ In Progress | ❌ Not Started

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on contributing to this project.
