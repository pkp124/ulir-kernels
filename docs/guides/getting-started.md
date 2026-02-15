# Getting Started

This guide walks you through setting up the project and generating your first kernel.

## Prerequisites

### Required

- **LLVM/MLIR 18+**: With MLIR enabled
- **CMake 3.20+**: Build system
- **Python 3.10+**: For Python bindings and scripts
- **C++17 compiler**: GCC 10+ or Clang 13+

### Optional

- **Ninja**: Faster builds
- **QEMU**: For RISC-V testing
- **Spike**: RISC-V ISA simulator

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd ulir-kernels
```

### 2. Install LLVM/MLIR

If you don't have LLVM with MLIR:

```bash
# Option 1: Package manager (Ubuntu/Debian)
sudo apt install llvm-18 llvm-18-dev mlir-18-tools libmlir-18-dev

# Option 2: Build from source
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
cmake -S llvm -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_PROJECTS="mlir" \
  -DLLVM_TARGETS_TO_BUILD="host;RISCV" \
  -DLLVM_ENABLE_ASSERTIONS=ON
cmake --build build
```

### 3. Run Setup

```bash
make setup
```

This will:
- Check for required dependencies
- Create Python virtual environment
- Install Python packages
- Configure the build

### 4. Build

```bash
make build
```

### 5. Verify

```bash
make test
```

## Your First Kernel

### 1. Create a Kernel File

Create `examples/first_kernel.mlir`:

```mlir
// A simple ReLU activation kernel
func.func @relu_kernel(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %output = ks.relu %input : tensor<1024xf32>
  return %output : tensor<1024xf32>
}
```

### 2. Lower to Vector IR

```bash
./build/bin/ks-opt examples/first_kernel.mlir \
  --ks-lower-activations \
  -o examples/first_kernel_vector.mlir
```

Output:
```mlir
func.func @relu_kernel(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %c0 = arith.constant 0 : index
  %c1024 = arith.constant 1024 : index
  %vl = arith.constant 32 : index
  %zero = arith.constant dense<0.0> : vector<32xf32>
  
  %output = scf.for %i = %c0 to %c1024 step %vl iter_args(%out = %init) {
    %v = vector.load %input[%i] : vector<32xf32>
    %relu = arith.maxf %v, %zero : vector<32xf32>
    vector.store %relu, %out[%i] : vector<32xf32>
    scf.yield %out
  }
  return %output : tensor<1024xf32>
}
```

### 3. Lower to RISC-V RVV

```bash
./build/bin/ks-opt examples/first_kernel_vector.mlir \
  --ks-lower-to-rvv \
  --convert-to-llvm \
  -o examples/first_kernel_llvm.mlir
```

### 4. Generate Assembly

```bash
mlir-translate --mlir-to-llvmir examples/first_kernel_llvm.mlir | \
llc -march=riscv64 -mattr=+v -o examples/first_kernel.s
```

## Development Workflow

### Using Make Commands

```bash
# See all available commands
make help

# Build with debug symbols
make build-debug

# Run specific tests
make test-unit
make test-lit

# Full verification before committing
make verify
```

### Creating New Kernels

```bash
# Use the scaffolding script
make new-kernel NAME=my_kernel

# This creates:
# - specs/kernels/my_kernel.md (specification)
# - tests/lit/Dialect/Kernel/my_kernel.mlir (tests)
# - tasks/KERNEL-MY_KERNEL.md (task tracking)
```

### Creating New Passes

```bash
make new-pass NAME=optimize-something

# This creates:
# - src/passes/OptimizeSomething.cpp
# - tests/lit/Transforms/optimize-something.mlir
```

## Using Python Bindings

> **Note**: Python bindings are not yet implemented. This shows the planned API.

```python
import kernelsmith

# Load an MLIR module
module = kernelsmith.load("kernel.mlir")

# Apply transformations
module = kernelsmith.tile(module, tile_sizes=[64, 64, 32])
module = kernelsmith.vectorize(module)
module = kernelsmith.lower_to_rvv(module)

# Generate code
code = kernelsmith.compile(module, target="riscv64+v")

# Save or execute
kernelsmith.save(code, "kernel.o")
```

## Next Steps

- Read the [Architecture Overview](../architecture/overview.md)
- Explore [kernel specifications](../../specs/kernels/)
- Try the [examples](../../examples/)
- Learn about [adding new kernels](adding-kernels.md)
