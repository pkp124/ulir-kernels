# DES-002: MatMul Kernel Implementation (TDD Example)

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-06 |
| **Phase** | Phase 3 - Core Kernel |
| **Priority** | Critical |
| **TDD Pattern** | Yes - Example for other kernels |

> **Current status (2026-09-28):** Historical TDD sketch. The library shape is
> DES-006, generic matmul lowering is DES-008, and the RVV path is DES-009.
> `ks.matmul` lowers through those passes today.

---

## Context

### Problem Statement

MatMul (matrix multiplication) is the most common operation in ML kernels. We need to:

1. Define how KernelSmith represents MatMul operations (`ks.matmul`)
2. Lower through the compilation pipeline to RISC-V RVV assembly
3. Achieve good performance through tiling and vectorization
4. Provide a comprehensive TDD example for other kernel implementations

### Background

- MatMul specification already exists: `specs/kernels/matmul.md`
- Vector operations lowering (DES-001) provides foundation
- TDD approach: RED (tests fail) → GREEN (implement) → REFACTOR (optimize)

### Scope

This design covers the complete MatMul pipeline:
```
ks.matmul (High-level)
  ↓ Lower to Linalg
linalg.matmul + SCF loops
  ↓ Tiling
Tiled Linalg + nested SCF
  ↓ Vectorization
Vector loads/stores/fma
  ↓ RVV Lowering (DES-001)
LLVM RVV IR
```

---

## Requirements

From specification: `specs/kernels/matmul.md`

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Parse `ks.matmul` operations | Must Have |
| REQ-2 | Verify operand shapes and element types | Must Have |
| REQ-3 | Lower `ks.matmul` to `linalg.matmul` | Must Have |
| REQ-4 | Apply tiling transformation (M=64, N=64, K=32) | Must Have |
| REQ-5 | Vectorize inner matmul (vector loads, fma, stores) | Must Have |
| REQ-6 | Lower to RISC-V RVV assembly | Must Have |
| REQ-7 | Support dynamic shapes (partial) | Should Have |
| REQ-8 | Support different element types (f32, f16, i8) | Should Have |
| REQ-9 | Performance within 80% of reference BLAS | Should Have |
| REQ-10 | Multi-VLEN functional validation (128, 256, 512) | Must Have |

---

## Design

### Architecture Overview

```
INPUT PHASE (Specification)
  ↓
Parse & Verify
  ├─ Check tensor dimensions (M×K and K×N)
  ├─ Verify element type compatibility
  └─ Create ks.matmul operation

Lower to Linalg (MLIR Standard)
  ├─ Convert to linalg.matmul + tensor.empty
  └─ Maintain tensor semantics

Apply Tiling (Cache Efficiency)
  ├─ M×N×K → (M/Tm) × (N/Tn) × (K/Tk) blocks
  ├─ Tile size: M=64, N=64, K=32 (for VLEN=256)
  └─ Generate nested scf.for loops

Vectorize (SIMD Exploitation)
  ├─ Inner-most K loop becomes vector operations
  ├─ Generate vector.load for A, B
  ├─ Generate vector.fma for computation
  └─ Generate vector.store for C

RVV Lowering (DES-001)
  ├─ vector.* → llvm.call @llvm.riscv.*
  └─ Generate vsetvl instructions

LLVM → Assembly
  ├─ LLVM backend
  └─ RISC-V ASM output

PERFORMANCE PHASE (Tuning)
  ├─ Measure on QEMU (multi-VLEN)
  ├─ Identify bottlenecks
  └─ Optimize (register allocation, instruction scheduling)
```

### Component Design

#### 1. ks.matmul Operation Definition

**Operation Semantics**:
```mlir
// C = A × B
// A: shape (M, K)
// B: shape (K, N)
// C: shape (M, N)
// Element types: f32, f16, i8 (extensible)

%C = ks.matmul %A, %B : tensor<MxKxELEM_TYPE>, tensor<KxNxELEM_TYPE>
                         -> tensor<MxNxELEM_TYPE>
```

**Verifier Checks**:
- A is 2D tensor
- B is 2D tensor
- A.shape[1] == B.shape[0] (inner dimensions match)
- Element types are compatible
- Element type can be promoted (e.g., f16 → f32)

**Attributes** (for future extensions):
```mlir
// Optional attributes
%C = ks.matmul %A, %B {
  tile_m = 64,
  tile_n = 64,
  tile_k = 32,
  packed = false  // Data layout optimization
} : ...
```

#### 2. Lowering to Linalg

**Transformation**: `ks.matmul` → `linalg.matmul + tensor.empty`

```mlir
// Before
%C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> -> tensor<64x256xf32>

// After
%C_init = tensor.empty() : tensor<64x256xf32>
%C = linalg.matmul ins(%A, %B : tensor<64x128xf32>, tensor<128x256xf32>)
                   outs(%C_init : tensor<64x256xf32>)
```

**Implementation**:
```cpp
class LowerKsMatmulToLinalgPattern : public ConversionPattern {
  matchAndRewrite(ks::MatMulOp op, ...) {
    // 1. Create empty output tensor
    auto init = rewriter.create<tensor::EmptyOp>(
      op.getLoc(), op.getResultType());

    // 2. Create linalg.matmul
    rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
      op, op.getLhs(), op.getRhs(), init);
  }
};
```

#### 3. Tiling Pass

**Goal**: Break large matmul into cache-friendly tiles

**Strategy**:
- Tile dimensions: M=64, N=64, K=32 (tuned for VLEN=256)
- Generate nested SCF loops for tile iteration
- Apply `linalg.tile` transformation

**Pseudocode**:
```python
def tile_matmul(linalg_matmul, tile_sizes=[64, 64, 32]):
    # Generate: for m in 0 to M step 64
    #             for n in 0 to N step 64
    #               for k in 0 to K step 32
    #                 matmul(A[m:m+64, k:k+32],
    #                        B[k:k+32, n:n+64],
    #                        C[m:m+64, n:n+64])

    return apply_linalg_tiling(linalg_matmul, tile_sizes)
```

**MLIR Before**:
```mlir
%C = linalg.matmul ins(%A, %B) outs(%C_init)
return %C
```

**MLIR After**:
```mlir
%c0 = arith.constant 0 : index
%c64 = arith.constant 64 : index
%c32 = arith.constant 32 : index
%M = tensor.dim %A, %c0 : tensor<64x128xf32>
%N = tensor.dim %B, %c1 : tensor<128x256xf32>
%K = tensor.dim %A, %c1 : tensor<64x128xf32>

%C = scf.for %m = %c0 to %M step %c64 iter_args(%C_i = %C_init) {
  %C_m = scf.for %n = %c0 to %N step %c64 iter_args(%C_mn = %C_i) {
    %C_mn_k = scf.for %k = %c0 to %K step %c32 iter_args(%C_mnk = %C_mn) {
      %A_tile = tensor.extract_slice %A[%m, %k] [64, 32] [1, 1]
      %B_tile = tensor.extract_slice %B[%k, %n] [32, 64] [1, 1]
      %C_tile = tensor.extract_slice %C_mnk[%m, %n] [64, 64] [1, 1]

      %C_tile_new = linalg.matmul ins(%A_tile, %B_tile)
                                   outs(%C_tile)

      %C_mnk_new = tensor.insert_slice %C_tile_new into %C_mnk[%m, %n] [64, 64] [1, 1]
      scf.yield %C_mnk_new
    }
    scf.yield %C_mn_k
  }
  scf.yield %C_m
}
```

#### 4. Vectorization Pass

**Goal**: Convert innermost scalar operations to vector operations

**Strategy**:
- Identify vectorizable loops (innermost K dimension)
- Promote memory operations to vector loads/stores
- Convert scalar FMA to vector FMA

**Pattern Matching**:
```cpp
// Detect pattern:
// for %i = 0 to M step VF:
//   load A[i] → %a
//   load B[i] → %b
//   fma %a, %b, %c → %c
//
// Vectorize to:
// for %i = 0 to M step VF:
//   vector.load A[i] → %va
//   vector.load B[i] → %vb
//   vector.fma %va, %vb, %vc → %vc
```

#### 5. End-to-End Pipeline

```cpp
std::unique_ptr<Pass> createMatMulPipeline() {
  auto pm = std::make_unique<PassManager>(context);

  // Stage 1: Lower kernel to linalg
  pm.addPass(createLowerKsMatmulToLinalgPass());

  // Stage 2: Apply tiling
  pm.addPass(createTileLinalgPass(/*sizes=*/{64, 64, 32}));

  // Stage 3: Vectorization
  pm.addPass(createVectorizeLinalgPass());

  // Stage 4: RVV lowering (from DES-001)
  pm.addPass(createLowerToRVVPass());

  // Stage 5: Standard LLVM lowering
  pm.addPass(createConvertToLLVMPass());

  return pm;
}
```

---

## Interface Design

### MLIR Operation Definition

```tablegen
def MatMul : KernelDialect_Op<"matmul",
  [NoSideEffect, Pure, Elementwise]> {
  let summary = "Matrix multiplication";
  let description = [{
    Performs matrix multiplication: C = A × B
    where A is (M×K) and B is (K×N), producing C of shape (M×N).
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = "$lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)";

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}
```

### Pass Interfaces

```cpp
// Create individual passes
std::unique_ptr<Pass> createLowerKsMatmulToLinalgPass();
std::unique_ptr<Pass> createTileMatmulPass(ArrayRef<int64_t> tileSizes);
std::unique_ptr<Pass> createVectorizeMatmulPass();

// Or use convenience pipeline
std::unique_ptr<Pass> createMatMulPipelinePass();
```

---

## Data Flow

```
ks.matmul operation
  ↓
[Verifier] - Check shapes, types
  ↓
[Lower to Linalg] - ks.matmul → linalg.matmul
  ↓
[Tile] - Apply SCF loops with tile sizes
  ↓
[Vectorize] - Inner loops → vector operations
  ↓
[Lower to RVV] (DES-001)
  ↓
[LLVM Codegen]
  ↓
RISC-V Assembly
```

---

## Test Strategy

### Phase 1: RED - Write Tests First

Tests organized by lowering stage:

#### 1.1 Parsing & Verifier Tests (`tests/lit/Dialect/Kernel/matmul_verifier.mlir`)

```mlir
// RUN: %ks-opt %s --verify-diagnostics 2>&1 | %FileCheck %s

// Valid: Square matrix
// CHECK-LABEL: @valid_square
func.func @valid_square(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// Valid: Rectangular
// CHECK-LABEL: @valid_rect
func.func @valid_rect(%A: tensor<32x128xf32>, %B: tensor<128x64xf32>) -> tensor<32x64xf32> {
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<32x64xf32>
  return %C : tensor<32x64xf32>
}

// Invalid: Mismatched inner dimensions
// CHECK: error: inner dimensions must match
func.func @invalid_mismatch(%A: tensor<32x128xf32>, %B: tensor<64x64xf32>) -> tensor<32x64xf32> {
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<64x64xf32> -> tensor<32x64xf32>
  return %C : tensor<32x64xf32>
}

// Invalid: Wrong output dimensions
// CHECK: error: output dimensions must be
func.func @invalid_output(%A: tensor<32x128xf32>, %B: tensor<128x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<32x128xf32>, tensor<128x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}
```

Tests status: **RED** (tests fail because features not implemented)

#### 1.2 Lowering Tests (`tests/lit/Transforms/lower_matmul_*.mlir`)

```mlir
// RUN: %ks-opt %s --ks-lower-matmul 2>&1 | %FileCheck %s

// CHECK-LABEL: @lower_to_linalg
// CHECK: linalg.matmul
// CHECK: tensor.empty
func.func @lower_to_linalg(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: @after_tiling
// CHECK: scf.for {{.*}} step %{{[0-9]+}}
// CHECK-COUNT-3: scf.for  // Three nested loops
func.func @after_tiling(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: @after_vectorization
// CHECK: vector.load
// CHECK: vector.fma
// CHECK: vector.store
func.func @after_vectorization(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}

// CHECK-LABEL: @rvv_intrinsics
// CHECK: llvm.call @llvm.riscv.vle32.v
// CHECK: llvm.call @llvm.riscv.vfmacc.vv
// CHECK: llvm.call @llvm.riscv.vse32.v
func.func @rvv_intrinsics(%A: tensor<64x64xf32>, %B: tensor<64x64xf32>) -> tensor<64x64xf32> {
  %C = ks.matmul %A, %B : tensor<64x64xf32>, tensor<64x64xf32> -> tensor<64x64xf32>
  return %C : tensor<64x64xf32>
}
```

Tests status: **RED** (lowering passes not yet implemented)

#### 1.3 C++ Unit Tests (`tests/unit/MatMulTest.cpp`)

```cpp
#include <gtest/gtest.h>
#include "KernelSmith/Dialect/Kernel/KernelOps.h"

class MatMulTest : public ::testing::Test {
protected:
  mlir::MLIRContext context;
  mlir::OpBuilder builder{&context};
};

TEST_F(MatMulTest, ParseMatMulOperation) {
  // Create types
  auto f32 = builder.getF32Type();
  auto A_type = RankedTensorType::get({64, 128}, f32);
  auto B_type = RankedTensorType::get({128, 64}, f32);
  auto C_type = RankedTensorType::get({64, 64}, f32);

  // Create operation
  auto matmul_op = builder.create<ks::MatMulOp>(
    builder.getUnknownLoc(),
    C_type,
    /*operands*/);

  EXPECT_TRUE(matmul_op);
}

TEST_F(MatMulTest, VerifyDimensionMismatch) {
  // Should fail verifier for mismatched inner dimensions
  // EXPECT_FALSE(verify(matmul_op));
}

TEST_F(MatMulTest, SupportMultipleDataTypes) {
  // Test f32, f16, i8
}
```

Tests status: **RED**

---

### Phase 2: GREEN - Implement to Pass Tests

#### 2.1 Create ks.matmul Operation

Files to create:
- `include/KernelSmith/Dialect/Kernel/KernelOps.td` - Add MatMul operation definition
- `lib/Dialect/Kernel/KernelOps.cpp` - Add verifier logic

After implementation:
```bash
./scripts/run-tests.sh --lit
# matmul_verifier.mlir: PASSED ✓
```

#### 2.2 Implement Lowering Passes

For each pass:
1. Create pattern in `lib/Passes/MatMulLowering.cpp`
2. Register in pass registry
3. Run tests
4. Verify **RED** → **GREEN** transition

```bash
./scripts/run-tests.sh --lit
# lower_matmul_*.mlir: PASSED ✓
```

---

### Phase 3: REFACTOR - Optimize Without Breaking Tests

Once all tests pass:
- Optimize tiling strategy
- Improve vectorization quality
- Reduce vsetvl instructions
- **All tests still pass** ✓

---

### Functional Validation

#### 3.1 Test Data Generation

```bash
make test-data
# Generates test_data/matmul_*.bin files
```

#### 3.2 Functional Testing

```python
# tests/matmul_functional_test.py
from tests.functional_validator import FunctionalValidator
import numpy as np


def test_matmul_correctness():
    validator = FunctionalValidator(verbose=True)

    # Load test data
    A = np.fromfile("tests/test_data/matmul_small_A.bin", dtype=np.float32).reshape(4, 4)
    B = np.fromfile("tests/test_data/matmul_small_B.bin", dtype=np.float32).reshape(4, 4)
    expected_C = np.fromfile("tests/test_data/matmul_small_ref.bin", dtype=np.float32).reshape(4, 4)

    # Run compiled kernel (would capture output)
    computed_C = run_kernel(A, B)

    # Validate
    result = validator.validate_matmul(computed_C, A, B)
    assert result.passed, f"Failed with error: {result.max_error}"
```

#### 3.3 Multi-VLEN Testing

```bash
# Test with different VLEN values
./scripts/run-tests.sh --integration --qemu-vlen 128
./scripts/run-tests.sh --integration --qemu-vlen 256
./scripts/run-tests.sh --integration --qemu-vlen 512
```

---

## Edge Cases

- [ ] Empty matrices (0×0)
- [ ] Single-element matrices (1×1)
- [ ] Tall matrices (10000×1)
- [ ] Wide matrices (1×10000)
- [ ] Non-power-of-2 dimensions
- [ ] Large matrices (1024×1024)
- [ ] Dynamic shapes (known at runtime, not compile-time)
- [ ] Different element types (f32, f16, i8) in same operation

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Tiling strategy suboptimal | Medium | Medium | Profile and tune on real hardware |
| Register pressure too high | Medium | Low | Monitor LLVM output, adjust LMUL |
| Tail handling incorrect | High | Medium | Explicit tests for non-divisible dimensions |
| Performance regression | Medium | Low | Benchmark vs reference regularly |
| LLVM backend issues | High | Low | Test with LLVM 18+ only |

---

## Dependencies

- DES-001: Vector Operations Lowering (must complete first)
- LLVM 18+ with RISC-V support
- MatMul specification: `specs/kernels/matmul.md`
- Test infrastructure: Phase 1 (complete)

---

## Implementation Plan

### Week 1: RED Phase (Write Tests)
- [ ] Write all lit tests (parsing, verifier, lowering stages)
- [ ] Write all unit tests
- [ ] Confirm tests fail appropriately
- [ ] Generate test data

### Week 2: GREEN Phase (Implement)
- [ ] Create ks.matmul operation definition
- [ ] Implement verifier logic
- [ ] Implement lower-to-linalg pass
- [ ] Verify tests pass one by one

### Week 2-3: GREEN Phase Continued
- [ ] Implement tiling pass
- [ ] Implement vectorization pass
- [ ] Implement RVV lowering pass
- [ ] Run full test suite

### Week 3: REFACTOR Phase (Optimize)
- [ ] Profile on QEMU
- [ ] Optimize tiling parameters
- [ ] Reduce vsetvl instructions
- [ ] Verify all tests still pass

### Week 4: Validation & Documentation
- [ ] Multi-VLEN testing (128, 256, 512)
- [ ] Functional correctness validation
- [ ] Performance benchmarking
- [ ] Write comprehensive documentation

---

## Success Criteria

- ✅ All lit tests pass (parsing, lowering, RVV generation)
- ✅ All unit tests pass
- ✅ Functional validation passes (correctness within tolerance)
- ✅ Multi-VLEN testing passes (VLEN=128, 256, 512)
- ✅ Performance meets acceptance criteria (80%+ of reference)
- ✅ Code review approved
- ✅ Documentation complete

---

## Review History

(To be filled in during review process)

---

## Related Documents

- [DES-001: Vector Operations Lowering](DES-001-vector-operations-lowering.md)
- [MatMul Specification](../../specs/kernels/matmul.md)
- [Testing Guide](../guides/testing-guide.md)
- [TDD Development Plan](../../RISC-V_RVV_KERNEL_LIBRARY_PLAN.md)
