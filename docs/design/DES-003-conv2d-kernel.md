# DES-003: Conv2D Kernel Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-06 |
| **Phase** | Phase 4 - Extended Kernels |
| **Priority** | High |

---

## Context

### Problem Statement

2D convolution is a fundamental operation in CNNs. We need to:
1. Define `ks.conv2d` operation with stride, padding, dilation support
2. Lower efficiently to RISC-V RVV with proper data layout handling
3. Optimize for memory access patterns (NHWC format)

### Key Differences from MatMul

- **More complex indexing**: 2D spatial operations + channels
- **Data layout matters**: NHWC (channels-last) preferred for vectorization
- **Striding and padding**: Additional complexity in bounds checking
- **Dilation**: Sparse element access patterns

---

## Requirements

From specification: `specs/kernels/conv2d.md`

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Parse `ks.conv2d` with stride, padding, dilation | Must Have |
| REQ-2 | Support NHWC data layout | Must Have |
| REQ-3 | Verify operand and result shapes | Must Have |
| REQ-4 | Lower to im2col + matmul pattern | Should Have |
| REQ-5 | Direct lowering to vectorized loops | Should Have |
| REQ-6 | Support dynamic spatial dimensions | Should Have |
| REQ-7 | Performance within 75% of reference | Should Have |
| REQ-8 | Multi-VLEN testing (128, 256, 512) | Must Have |

---

## Design Overview

### Architecture

```
ks.conv2d
  ├─ Input: (N, H, W, C_in)   [NHWC format]
  ├─ Filter: (K_h, K_w, C_in, C_out)
  └─ Output: (N, H_out, W_out, C_out)
    ↓
Lower to standard form
    ↓
Option A: im2col + matmul
  - Expand receptive field into columns
  - Convert to matrix multiplication
  - Efficient but high memory overhead

Option B: Direct tiled loops
  - Output space blocking (tile height/width)
  - Channel blocking for vectorization
  - Lower memory, more register-friendly
    ↓
Tiling & Vectorization
  - Tile H/W dimensions for cache efficiency
  - Vectorize channel dimension
    ↓
RVV Lowering (DES-001)
    ↓
RISC-V Assembly
```

### MLIR Operation Definition

```mlir
%output = ks.conv2d %input, %filter
  {stride = [1, 1], padding = [1, 1], dilation = [1, 1]}
  : tensor<1x32x32x3xf32>, tensor<3x3x3x16xf32>
  -> tensor<1x32x32x16xf32>
```

### Implementation Phases

**Phase 4a: Specification & Tests** (Parallel to MatMul RED phase)
- Define ks.conv2d operation
- Write verifier tests
- Write lowering tests (basic to im2col)
- Write functional tests with reference (scipy convolve)

**Phase 4b: Lowering to im2col** (After MatMul GREEN)
- Implement ks.conv2d → im2col transformation
- Use existing matmul for multiplication
- Validate correctness

**Phase 4c: Direct Tiled Loops** (Optional optimization)
- Skip im2col for certain configurations
- Direct loop over spatial and channel dimensions
- Better register allocation

---

## Test Strategy

### Tests Structure

```
tests/lit/Dialect/Kernel/conv2d_verifier.mlir
  ├─ Valid operations (various shapes, strides, padding)
  └─ Invalid operations (dimension mismatches)

tests/lit/Transforms/lower_conv2d_*.mlir
  ├─ Lower to linalg.conv_2d
  ├─ Lower to tiled loops
  └─ Lower to RVV intrinsics

tests/unit/Conv2DTest.cpp
  ├─ Operation parsing
  ├─ Shape verification
  └─ Stride/padding computation

tests/functional_validator.py
  ├─ Correctness vs scipy reference
  ├─ Different input shapes
  └─ Various strides and padding
```

### Edge Cases

- [ ] 1×1 convolution (no spatial reduction)
- [ ] Large filter (larger than output)
- [ ] Dilation (sparse access patterns)
- [ ] Non-square filters (3×5, etc.)
- [ ] Non-square inputs
- [ ] Single-channel inputs/outputs
- [ ] Dynamic spatial dimensions

---

## Implementation Plan

### Week 1-2: Specification & Tests
- [ ] Define ks.conv2d operation
- [ ] Write all verifier tests (RED phase)
- [ ] Write lowering tests (RED phase)
- [ ] Implement operation definition (GREEN phase)

### Week 2-3: Lowering Implementation
- [ ] Implement conv2d verifier
- [ ] Implement lower to linalg (if applicable)
- [ ] Implement im2col transformation
- [ ] Implement tiling
- [ ] Implement vectorization

### Week 3: Validation
- [ ] Functional correctness testing
- [ ] Multi-VLEN validation
- [ ] Performance profiling

---

## Success Criteria

- ✅ All lit tests passing
- ✅ All unit tests passing
- ✅ Functional validation passing (correctness vs reference)
- ✅ Multi-VLEN testing (128, 256, 512)
- ✅ Performance acceptable (75%+ of reference)

---

## Dependencies

- DES-001: Vector Operations Lowering
- DES-002: MatMul Kernel (for im2col approach)
- Conv2D specification: `specs/kernels/conv2d.md`

---

## Related Documents

- [Conv2D Specification](../../specs/kernels/conv2d.md)
- [MatMul Kernel Design](DES-002-matmul-kernel.md)
- [Vector Operations Lowering](DES-001-vector-operations-lowering.md)
