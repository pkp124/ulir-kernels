# DES-014: INT8 Dot and GEMV Lowering

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Author** | KernelSmith team |
| **Created** | 2026-06-14 |
| **Related** | TASK-013, TASK-014, TASK-015, TASK-016, TASK-018, specs/kernels/quantization.md |

> **Current status (2026-09-28):** `ks.dot_i8` and `ks.matvec_i8` lower to
> linalg, and generated INT8 RVV objects can be linked behind the C API
> (`TASK-018`). The problem statement below describes the gap this design
> closed. W4A8 object replacement remains `TASK-030`.

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
Add `ks.dot_i8` and `ks.matvec_i8` as pure KernelSmith ops and lower them in
`--ks-lower-to-linalg` to `linalg.generic` reductions. The lowering widens i8
operands to i32 with signed extension, subtracts the i8 zero-points in i32,
multiplies, and accumulates into i32 output tensors.

### Component Design

#### `ks.dot_i8`
`ks.dot_i8` consumes two rank-1 i8 tensors with matching length and returns a
rank-0 i32 tensor. Attributes `input_zero_point` and `weight_zero_point`
default to zero and must fit in signed i8.

#### `ks.matvec_i8`
`ks.matvec_i8` consumes a rank-1 i8 input tensor and a rank-2 row-major i8
weights tensor shaped `[rows, cols]`. The input length must match `cols`, and
the rank-1 i32 result length must match `rows`.

#### Linalg lowering
`ks.dot_i8` lowers to:

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

`ks.matvec_i8` lowers to one `linalg.generic` with indexing maps:

```mlir
[
  affine_map<(row, col) -> (col)>,
  affine_map<(row, col) -> (row, col)>,
  affine_map<(row, col) -> (row)>
]
```

The iterator types are `["parallel", "reduction"]`, producing one i32
accumulator per weight row.

### Interface

```mlir
func.func @dot(%input: tensor<128xi8>, %weight: tensor<128xi8>)
    -> tensor<i32> {
  %acc = ks.dot_i8 %input, %weight
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<128xi8>, tensor<128xi8> -> tensor<i32>
  return %acc : tensor<i32>
}

func.func @matvec(%input: tensor<128xi8>, %weights: tensor<4x128xi8>)
    -> tensor<4xi32> {
  %out = ks.matvec_i8 %input, %weights
      {input_zero_point = -3 : i64, weight_zero_point = 5 : i64}
      : tensor<128xi8>, tensor<4x128xi8> -> tensor<4xi32>
  return %out : tensor<4xi32>
}
```

#### Generated object integration policy

`TASK-018` uses static target-profile selection instead of runtime dispatch. The
generic profile compiles the scalar reference `ks_dot_i8` and `ks_matvec_i8`
symbols from `ks_quantized.c`. A `riscv_rvv_256` build may pass generated object
files through `KS_INT8_RVV_OBJECTS`; those objects must export the same public
symbols, and the reference definitions are suppressed to avoid duplicate linker
symbols.

Generated object names should include the target profile and public symbol, for
example:

```text
ks_dot_i8_riscv_rvv_256.o
ks_matvec_i8_riscv_rvv_256.o
```

The public headers do not change. Workspace and alignment queries remain in the
C reference source and read target-profile macros, so future generated kernels
can request scratch space without changing the ABI.

### Data Flow

```
ks.dot_i8
  -> --ks-lower-to-linalg
linalg.generic i32 reduction
ks.matvec_i8
  -> --ks-lower-to-linalg
linalg.generic parallel row / reduction column i32 accumulation
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
- [x] Parse/print `ks.dot_i8`.
- [x] Reject invalid ranks, element types, shapes, result type, and zero-points.
- [x] Check `--ks-lower-to-linalg` emits signed i32 accumulation.
- [x] Parse/print `ks.matvec_i8`.
- [x] Reject invalid GEMV ranks, element types, shapes, result type, and
      zero-points.
- [x] Check `--ks-lower-to-linalg` emits row-parallel signed i32 accumulation.

### Edge Cases
- [x] Dynamic K dimensions are accepted when operands are dynamic.
- [x] Static K mismatches are rejected.

### Integration Tests
- [x] Full CTest keeps host quantized golden validation green.
- [x] RISC-V QEMU validation runs `dot_i8_smoke` and `matvec_i8_smoke` through
      the descriptor-backed C API runner.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Ambiguous scalar representation | Medium | Medium | Use rank-0 tensor output consistently with tensor IR. |
| Incorrect signedness | High | Medium | Require i8 operands and use `arith.extsi` before zero-point subtraction. |
| Scope creep into W4A8 | Medium | Medium | Track W4A8 in `TASK-017`; keep this design INT8 first. |

## Open Questions

- [x] `ks.matvec_i8` lowers as one rank-2 linalg reduction instead of
      decomposing into dot slices.
- [x] Where should generated quantized RVV objects plug into `libkernelsmith.a`?
      Track the generated INT8 RVV object replacement separately in `TASK-018`.

## Dependencies

- `TASK-013`: C ABI and layout contract.
- `TASK-014`: quantized golden validation.
- `TASK-015`: RVV quantized profile parameters.

## Implementation Plan

### Phase 1: INT8 dot linalg slice
- [x] Add `ks.dot_i8` with verifier and lit coverage.
- [x] Lower `ks.dot_i8` to `linalg.generic` with i32 accumulation.

### Phase 2: GEMV and RVV follow-ons
- [x] Add `ks.matvec_i8` and row-wise lowering.
- [x] Add quantized RISC-V functional validation for the INT8 C API runner.
- [x] Add profile-selected external INT8 object hooks for `riscv_rvv_256`
      builds.
- [x] Wire real generated quantized RVV objects into the public C API build path
      (`TASK-018`).

---

## Review History

### Review 1 (pending)
**Reviewer**: TBD
**Decision**: Pending

**Feedback**:
- Pending review.

**Resolution**:
- Pending review.
