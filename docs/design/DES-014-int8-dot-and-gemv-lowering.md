# DES-014: INT8 Dot and GEMV Lowering

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith team |
| **Created** | 2026-06-14 |
| **Related** | TASK-013, TASK-014, TASK-015, TASK-016, specs/kernels/quantization.md |

## Context

### Problem Statement
KernelSmith has scalar C APIs and golden validation for INT8 dot/GEMV, but the
compiler IR has no quantized compute op to lower toward vector or RVV code.
`TASK-016` needs the first compiler-visible INT8 dot path with i8 operands,
i32 accumulation, and explicit zero-point metadata.

### Background
`TASK-013` defines the public `ks_dot_i8` and `ks_matvec_i8` ABI. `TASK-014`
adds exact quantized golden cases for dot and GEMV. `TASK-015` adds RVV target
profile macros for quantized tile sizes. This design starts with rank-1 dot
lowering because it is the smallest path that proves the accumulator contract.

## Requirements

From specification: `specs/kernels/quantization.md`.

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Represent INT8 dot with i8 inputs, i32 output, and zero-point metadata. | Must Have |
| REQ-2 | Lower dot to structured MLIR with signed i32 accumulation. | Must Have |
| REQ-3 | Preserve the `ks -> linalg -> vector -> target` lowering layers. | Must Have |
| REQ-4 | Keep GEMV and RVV object integration as incremental follow-ons. | Should Have |

## Design

### Overview
Add `ks.dot_i8` as a pure KernelSmith op and lower it in
`--ks-lower-to-linalg` to a `linalg.generic` reduction. The lowering widens both
i8 operands to i32 with signed extension, subtracts the i8 zero-points in i32,
multiplies, and accumulates into an i32 scalar tensor.

### Component Design

#### `ks.dot_i8`
`ks.dot_i8` consumes two rank-1 i8 tensors with matching length and returns a
rank-0 i32 tensor. Attributes `input_zero_point` and `weight_zero_point`
default to zero and must fit in signed i8.

#### Linalg lowering
The op lowers to:

```mlir
%empty = tensor.empty() : tensor<i32>
%zero = arith.constant 0 : i32
%init = linalg.fill ins(%zero : i32) outs(%empty : tensor<i32>)
%out = linalg.generic {
  indexing_maps = [
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> (d0)>,
    affine_map<(d0) -> ()>
  ],
  iterator_types = ["reduction"]
} ins(%input, %weight : tensor<Kxi8>, tensor<Kxi8>)
  outs(%init : tensor<i32>) { ... }
```

The generic body implements:

```text
acc += (int32(input[k]) - input_zero_point) *
       (int32(weight[k]) - weight_zero_point)
```

### Interface

```mlir
func.func @dot(%input: tensor<128xi8>, %weight: tensor<128xi8>)
    -> tensor<i32> {
  %acc = ks.dot_i8 %input, %weight
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<128xi8>, tensor<128xi8> -> tensor<i32>
  return %acc : tensor<i32>
}
```

### Data Flow

```
ks.dot_i8
  -> --ks-lower-to-linalg
linalg.generic i32 reduction
  -> future tiling/vectorization/RVV lowering
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Extend `ks.matmul` for i8 dot | Reuses one op family | Confuses matrix and scalar contracts; zero-point metadata does not fit existing op. | Rejected |
| Lower directly to LLVM/RVV | Shorter path to assembly | Bypasses KernelSmith layering and loses linalg/vector testability. | Rejected |
| Add `ks.dot_i8` and lower to linalg | Small vertical slice; matches C ABI; keeps vectorizable IR. | GEMV/RVV still need follow-on work. | **Selected** |

## Test Strategy

### Unit Tests
- [ ] Keep existing C API and golden dot/GEMV tests passing.

### Lit Tests
- [ ] Parse/print `ks.dot_i8`.
- [ ] Reject invalid ranks, element types, shapes, result type, and zero-points.
- [ ] Check `--ks-lower-to-linalg` emits signed i32 accumulation.

### Edge Cases
- [ ] Dynamic K dimensions are accepted when both operands are dynamic.
- [ ] Static K mismatches are rejected.

### Integration Tests
- [ ] Full CTest keeps host quantized golden validation green.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ambiguous scalar representation | Medium | Medium | Use rank-0 tensor output consistently with tensor IR. |
| Incorrect signedness | High | Medium | Require i8 operands and use `arith.extsi` before zero-point subtraction. |
| Scope creep into W4A8 | Medium | Medium | Track W4A8 in `TASK-017`; keep this design INT8 first. |

## Open Questions

- [ ] Should `ks.matvec_i8` lower by decomposition into dot slices or as one
      rank-2 linalg reduction for better vectorization?
- [ ] Where should generated quantized RVV objects plug into `libkernelsmith.a`?

## Dependencies

- `TASK-013`: C ABI and layout contract.
- `TASK-014`: quantized golden validation.
- `TASK-015`: RVV quantized profile parameters.

## Implementation Plan

### Phase 1: INT8 dot linalg slice
- [ ] Add `ks.dot_i8` with verifier and lit coverage.
- [ ] Lower `ks.dot_i8` to `linalg.generic` with i32 accumulation.

### Phase 2: GEMV and RVV follow-ons
- [ ] Add `ks.matvec_i8` and row-wise lowering.
- [ ] Add quantized tiling/vectorization/RVV validation using profile macros.

---

## Review History

### Review 1 (pending)
**Reviewer**: TBD
**Decision**: Pending

**Feedback**:
- Pending review.

**Resolution**:
- Pending review.
