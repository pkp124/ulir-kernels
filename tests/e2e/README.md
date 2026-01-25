# End-to-End Tests

This directory contains end-to-end tests that verify the complete pipeline from KernelSmith IR to execution on target hardware/simulators.

## M1 Tests (Matmul on RVV)

| File | Description |
|------|-------------|
| `matmul_64x64.mlir` | 64x64 f32 matrix multiplication |
| `matmul_128x128.mlir` | 128x128 f32 matrix multiplication |
| `matmul_256x256.mlir` | 256x256 f32 matrix multiplication |
| `matmul_harness.c` | C test harness for verification |
| `matmul_*_expected.txt` | Expected output for comparison |

## Running Tests

```bash
# Run single test
./scripts/run-matmul-test.sh 64

# Run with specific VLEN
./scripts/run-matmul-test.sh 64 256

# Run full M1 verification
./scripts/verify-m1.sh
```

## Test Flow

```
1. ks-opt compiles .mlir → LLVM IR
2. llc generates RISC-V assembly
3. riscv64-gcc links with test harness
4. qemu-riscv64 executes with RVV
5. Output compared to expected
```

## Prerequisites

- `ks-opt` built
- `mlir-translate` and `llc` in PATH
- RISC-V cross-compiler (`riscv64-linux-gnu-gcc`)
- QEMU with RVV support (`qemu-riscv64`)
