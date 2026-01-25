# Examples

This directory contains example kernels and usage demonstrations.

## Examples

| Example | Description | Status |
|---------|-------------|--------|
| `matmul/` | Matrix multiplication kernels | Planned |
| `conv2d/` | 2D convolution kernels | Planned |
| `attention/` | Transformer attention | Planned |

## Running Examples

Once the project is built:

```bash
# Run matmul example
./build/bin/aikernel-opt examples/matmul/matmul.mlir \
    --tile-kernels --lower-to-vector --lower-to-rvv

# Generate RISC-V assembly
./build/bin/aikernel-opt examples/matmul/matmul.mlir \
    --tile-kernels --lower-to-vector --lower-to-rvv --convert-to-llvm | \
    mlir-translate --mlir-to-llvmir | \
    llc -march=riscv64 -mattr=+v -o matmul.s
```

## Adding Examples

1. Create a directory for your example
2. Add MLIR source files
3. Add a README explaining the example
4. Add any test/benchmark scripts
