# KernelSmith Testing Guide

## Overview

This guide covers the three levels of testing in KernelSmith:

1. **Lit Tests** - MLIR parsing and transformation verification
2. **Unit Tests** - C++ functionality tests
3. **Functional Tests** - Full kernel validation with reference implementations
4. **Integration Tests** - Multi-VLEN QEMU validation

## Test Pyramid

```
                    ∆ Integration Tests (Full pipeline on QEMU)
                   ∆∆ Functional Tests (Correctness vs reference)
                  ∆∆∆ Unit Tests (Pass isolation)
                 ∆∆∆∆ Lit Tests (Parsing/Lowering)
                ∆∆∆∆∆ Specification Tests (Type checking)
```

---

## 1. Lit Tests (MLIR IR Tests)

Lit tests verify:
- Operation parsing and printing
- Type verification (verifier)
- Transformation pass correctness
- IR lowering stages

### Running Lit Tests

```bash
# Run all lit tests
./scripts/run-tests.sh --lit

# Run with verbose output
./scripts/run-tests.sh --lit --verbose

# Run specific test category
lit tests/lit/Dialect/Kernel/

# Run single test file
lit tests/lit/Dialect/Kernel/matmul.mlir
```

### Writing Lit Tests

Lit tests use FileCheck syntax:

```mlir
// RUN: %ks-opt %s | %FileCheck %s

// CHECK-LABEL marks test function
// CHECK: checks for exact match
// CHECK-SAME: same line as previous CHECK
// CHECK-NEXT: next line

func.func @test_matmul_parsing(%A: tensor<64x128xf32>, %B: tensor<128x256xf32>)
    -> tensor<64x256xf32> {
  // CHECK: ks.matmul
  // CHECK-SAME: tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x256xf32>
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32>
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
```

### Common Lit Test Patterns

**Verifier Tests** (`tests/lit/Dialect/Kernel/verifier.mlir`):
```mlir
// Test invalid operations that verifier should reject
// RUN: %ks-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// CHECK: error: dimension mismatch
func.func @test_invalid_matmul() {
  // This should fail verification
  %A = arith.constant dense<1.0> : tensor<32x64xf32>
  %B = arith.constant dense<1.0> : tensor<32x64xf32>  // Wrong dimensions!
  // ...
}
```

**Lowering Tests** (`tests/lit/Transforms/lower_to_rvv.mlir`):
```mlir
// Test full lowering pipeline
// RUN: %ks-opt %s --ks-lower-to-rvv 2>&1 | %FileCheck %s

func.func @test_matmul_lowering(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>)
    -> tensor<64x64xf32> {
  // CHECK: linalg.matmul
  // CHECK: scf.for
  // CHECK: vector.fma
  // CHECK: llvm.call @llvm.riscv.
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32>
                          -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}
```

---

## 2. Unit Tests (C++ Tests)

Unit tests verify:
- Dialect registration
- Operation construction and verification
- Pass functionality in isolation

### Running Unit Tests

```bash
# Run all unit tests
./scripts/run-tests.sh --unit

# Run with verbose output
./scripts/run-tests.sh --unit --verbose

# Run specific test
ctest -R "MatMulTest" --output-on-failure
```

### Writing C++ Unit Tests

Using Google Test (GTest):

```cpp
#include <gtest/gtest.h>
#include "KernelSmith/Dialect/Kernel/KernelOps.h"
#include "mlir/IR/MLIRContext.h"
// ...

class MatMulTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  mlir::OpBuilder builder{&context};
};

TEST_F(MatMulTest, ParseMatMulOperation) {
  // Setup
  auto loc = builder.getUnknownLoc();

  // Create matrix types
  auto f32_type = builder.getF32Type();
  auto A_type = RankedTensorType::get({64, 64}, f32_type);
  auto B_type = RankedTensorType::get({64, 64}, f32_type);
  auto C_type = RankedTensorType::get({64, 64}, f32_type);

  // Test: Create matmul operation
  auto matmul_op = builder.create<ks::MatMulOp>(
    loc, C_type, /*operands*/);

  // Verify
  EXPECT_TRUE(matmul_op);
  EXPECT_EQ(matmul_op.getResultType(), C_type);
}

TEST_F(MatMulTest, VerifyDimensionMismatch) {
  // Should fail verification
  // Create invalid dimensions
  // Verify that verifier catches error
}
```

---

## 3. Test Data Generation

Generate test tensors for functional validation:

```bash
# Generate all test data
python3 tests/test_data_generator.py

# Output saved to tests/test_data/
# - matmul_small_A.bin, matmul_small_B.bin, matmul_small_ref.bin
# - matmul_small_A.meta, etc. (metadata files)
```

### Golden-reference bundles

Golden-reference cases use a target-neutral JSON descriptor plus generated
NumPy artifacts:

```bash
python3 tests/golden/generate.py \
  --case tests/golden/cases/matmul_f32_smoke.json \
  --print-manifest
```

The generator writes a small bundle under `tests/golden/generated/<case>/`:

```text
manifest.json
input_<name>.npy
expected_<name>.npy
```

Each descriptor records:

- `name`, `kernel`, and optional source program path;
- deterministic NumPy generator backend, function, seed, and distribution;
- input and output tensor names, shapes, and dtypes;
- comparison mode and tolerance;
- target names and optional RVV VLENs.

Each manifest records the schema version, generator backend/version, seed,
distribution, tensor metadata, comparison policy, quantization policy fields,
and SHA-256 hash for every generated `.npy` file.

Run host/reference KernelSmith outputs against the committed golden data:

```bash
python3 tests/verify.py \
  --case tests/golden/cases/relu_f32_smoke.json \
  --target host_reference \
  --host-runner build/tests/host_reference/host-reference-runner
```

`tests/verify.py` validates manifest hashes, stages descriptor inputs for the
host runner, captures `actual_<name>.npy` outputs under `build/golden-actual/`,
compares them with the manifest comparison policy, and emits JSON result lines.
CTest runs the smoke cases through `kernelsmith-host-golden-*`, including the
f32 relu/matmul cases and the quantized i8/W4A8 dot/GEMV cases.

Run RISC-V RVV outputs under QEMU against the same golden bundle:

```bash
python3 tests/verify.py \
  --case tests/golden/cases/relu_f32_smoke.json \
  --target riscv_rvv_256 \
  --riscv-runner build-rvv/bin/riscv-golden-runner \
  --host-runner build/tests/host_reference/host-reference-runner \
  --vlens 256 512
```

The RISC-V path rejects VLENs below the profile baseline, restores executable
bits on downloaded runner artifacts, writes per-VLEN `actual_<name>.npy`
outputs, and emits JSON lines with case name, VLEN, pass/fail status, max error,
and mismatch count.

Use `PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional` to run
the local descriptor-backed RVV smoke set. It validates relu, matmul, INT8 dot,
and INT8 GEMV against golden outputs through QEMU when the RISC-V simulator and
cross compiler are installed.

Add `--benchmark --benchmark-runs 3 --benchmark-warmup 1` to record
`benchmarks` in the JSON report. For RISC-V targets, benchmark reports include
the host/reference runner plus each selected RVV VLEN so generic/reference and
RVV path timings are captured in one reproducible artifact.

To add a new f32 smoke case:

1. Add a descriptor in `tests/golden/cases/`.
2. Use `kernel` and `generator.function` values supported by
   `tests/golden/generate.py`.
3. Use `mode: allclose` for f32 outputs with explicit `rtol` and `atol`.
4. Generate the bundle and inspect `manifest.json`.
5. Add or update pytest coverage in `tests/test_golden_verify.py`.

Quantized cases must include explicit `rounding` and `saturation` fields in the
comparison policy. Use `quantized_exact` for integer equality and
`dequantized_allclose` when diagnostics should be reported in real values.

### Using Test Data

```python
import numpy as np

# Load test data
A = np.fromfile("tests/test_data/matmul_small_A.bin", dtype=np.float32)
B = np.fromfile("tests/test_data/matmul_small_B.bin", dtype=np.float32)
expected_C = np.fromfile("tests/test_data/matmul_small_ref.bin", dtype=np.float32)

# Read metadata
with open("tests/test_data/matmul_small_A.meta") as f:
    metadata = f.read()
    # shape: (4, 4)
    # dtype: float32
```

---

## 4. Functional Validation

Validate kernel outputs against reference implementations:

```bash
# Run functional validation tests
python3 tests/functional_validator.py

# Validates MatMul, Conv2D, Attention, Activations
# Generates report in validation_report.json
```

Golden-reference comparators are available through
`tests.functional_validator.compare_arrays`:

- `exact`: exact value comparison with optional strict shape/dtype checks.
- `allclose`: NumPy-style tolerance comparison for f32/f16 results.
- `quantized_exact`: exact integer comparison for quantized outputs.
- `dequantized_allclose`: dequantizes integer arrays with manifest scale and
  zero-point fields, then applies allclose.

Comparison reports include pass/fail, max and mean absolute error, max relative
error, mismatch count, and the first failing indices.

### Validation Workflow

1. Run kernel to generate output
2. Load test data and reference results
3. Compare using error metrics
4. Generate validation report

Example:

```python
from tests.functional_validator import FunctionalValidator

validator = FunctionalValidator(verbose=True)

# Load test data
computed_C = load_kernel_output("matmul_output.bin")
A = np.fromfile("tests/test_data/matmul_small_A.bin", dtype=np.float32)
B = np.fromfile("tests/test_data/matmul_small_B.bin", dtype=np.float32)

# Validate
result = validator.validate_matmul(computed_C, A, B)
print(f"Pass: {result.passed}, Max Error: {result.max_error}")

# Get summary
summary = validator.get_summary()
print(f"Passed: {summary['passed']}/{summary['total']}")
```

---

## 5. Multi-VLEN Testing (QEMU)

Test kernels with different vector lengths:

```bash
# Run integration tests on QEMU with different VLEN values
./scripts/run-tests.sh --integration --qemu-vlen 128
./scripts/run-tests.sh --integration --qemu-vlen 256
./scripts/run-tests.sh --integration --qemu-vlen 512
```

### QEMU Configuration

The `QEMURunner` class handles running RISC-V binaries on QEMU:

```python
from tests.qemu_runner import QEMURunner

runner = QEMURunner()

# Run with specific VLEN
ret_code, stdout, stderr = runner.run_with_vlen(
    binary_path="build/bin/test_matmul",
    vlen=256
)

# Run with multiple VLEN values
results = runner.run_multi_vlen(
    binary_path="build/bin/test_matmul",
    vlens=[128, 256, 512]
)

# Check consistency
consistent = runner.validate_consistent_results(results)
print(f"Results consistent: {consistent}")
```

### Installing QEMU

```bash
# Ubuntu/Debian
sudo apt install qemu-user

# Verify installation
qemu-riscv64 --version
```

---

## 6. Full Test Workflow

### Running All Tests

```bash
# Run all tests (lit + unit + integration)
./scripts/run-tests.sh --all

# Run with verbose output
./scripts/run-tests.sh --all --verbose
```

### Using Makefile Targets

```bash
# Run tests via make
make test              # All tests
make test-lit          # Lit tests only
make test-unit         # Unit tests only

# Full verification (lint + build + test)
make verify
```

---

## 7. TDD Development Cycle

### For Each New Kernel:

1. **RED Phase** - Write tests that fail
   ```bash
   # Create test files
   touch tests/lit/Dialect/Kernel/my_kernel.mlir
   touch tests/unit/MyKernelTest.cpp

   # Write tests (they will fail)
   ./scripts/run-tests.sh --lit
   # Result: FAILED
   ```

2. **GREEN Phase** - Implement to pass tests
   ```bash
   # Implement kernel
   # Add operation to KernelOps.td
   # Implement lowering patterns

   ./scripts/run-tests.sh --lit
   # Result: PASSED
   ```

3. **REFACTOR Phase** - Optimize without breaking tests
   ```bash
   # Optimize implementation
   # Add performance tweaks

   ./scripts/run-tests.sh --all
   # Result: Still PASSING
   ```

4. **Functional Validation**
   ```bash
   # Generate test data
   python3 tests/test_data_generator.py

   # Validate output
   python3 tests/functional_validator.py

   # Multi-VLEN testing
   ./scripts/run-tests.sh --integration --qemu-vlen 256
   ```

---

## 8. Debugging Test Failures

### Lit Test Debugging

```bash
# Run with debug output
lit tests/lit/Dialect/Kernel/matmul.mlir -v

# See actual vs expected
lit tests/lit/Dialect/Kernel/matmul.mlir --verbose

# Run ks-opt directly for inspection
./build/bin/ks-opt tests/lit/Dialect/Kernel/matmul.mlir
```

### Unit Test Debugging

```bash
# Run with gdb
gdb --args ./build/bin/test_kernel --gtest_filter="MatMulTest*"

# Run with verbose output
./build/bin/test_kernel --gtest_filter="MatMulTest*" --gtest_print_time=1
```

### QEMU Debugging

```bash
# Run with QEMU trace
qemu-riscv64 -trace events=cpu_exec,riscv_csr ./binary

# Check for vector extension support
qemu-riscv64 -cpu help | grep riscv
```

---

## 9. CI/CD Integration

Tests run automatically on each commit:

```yaml
# .github/workflows/ci.yml
- name: Run Tests
  run: ./scripts/run-tests.sh --all
```

View results on GitHub Actions after pushing.

---

## 10. Performance Profiling

### Basic Timing

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();
// Run kernel
auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
```

### QEMU Profiling

```python
# Use QEMURunner to benchmark
results = runner.benchmark_kernel(binary_path, vlen=256, runs=5)
print(f"Average time: {results['avg_time']}ms")
```

---

## References

- [LLVM Lit Documentation](https://llvm.org/docs/CommandGuide/lit/)
- [Google Test Documentation](https://google.github.io/googletest/)
- [RISC-V RVV Specification](https://riscv.org/technical/specifications/)
- [MLIR Testing Guide](https://mlir.llvm.org/docs/Testing/)
