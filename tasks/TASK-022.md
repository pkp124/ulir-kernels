# TASK-022: Lower ks.rms_norm to Vectorizable Linalg/Loop Form

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General or `.cursor/agents/mlir-pass-agent.md`.

## Description

Add compiler lowering for `ks.rms_norm` so the op compiles through the existing
RVV pipeline (`--ks-lower-to-linalg` → `--ks-tile`/`--ks-vectorize` →
`--ks-lower-to-rvv`) rather than relying only on the scalar C reference API
added in `TASK-020`. The lowering must use f32 accumulation for the
mean-of-squares reduction and apply the explicit epsilon before the reciprocal
square root, matching the reference semantics validated in `TASK-021`.

## Acceptance Criteria

- [x] Lower `ks.rms_norm` in `--ks-lower-to-linalg` to `linalg.generic`
      (reduction + two elementwise) with f32 accumulation for f32 and narrower
      element types.
- [x] Apply epsilon and `rsqrt` scaling consistently with the
      `ks_rms_norm_f32` reference and the golden descriptors
      (`rsqrt(ssq / inner + eps)`).
- [x] Preserve the weight (gain) operand semantics: `output = input * scale *
      weight[d]` broadcast over the trailing dimension.
- [x] Add a pass-transformation lit test asserting `ks.rms_norm` is removed and
      the expected linalg/arith/math ops appear (2D, 3D, dynamic trailing dim).
- [x] Confirm the lowered form vectorizes through `--ks-vectorize` and reaches
      LLVM via `--ks-lower-to-rvv` (lit coverage or documented pipeline run).
- [x] Run build, full CTest, and Python lint/format before completion.

## Dependencies

- `TASK-008`
- `TASK-019` (linalg-lowering pattern for elementwise ops)
- `TASK-020` (reference semantics)
- `TASK-021` (golden descriptors)
- `docs/design/DES-008-m3-linalg-matmul-lowering.md` (lowering pipeline reference)
- `docs/design/DES-011-riscv-first-transformer-demo.md`

## Verification

```bash
cmake --build build --parallel
cmake --build build --target check-kernelsmith-lit -- -v
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

Verified on 2026-08-08:

```bash
cmake --build build --parallel
cmake --build build --target check-kernelsmith-lit -- -v
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
build/bin/ks-opt tests/lit/Passes/lower-rms-norm.mlir \
  --ks-lower-to-linalg --ks-vectorize --ks-lower-to-rvv \
  -o /tmp/kernelsmith-rmsnorm-rvv.mlir
mlir-translate-21 --mlir-to-llvmir \
  /tmp/kernelsmith-rmsnorm-rvv.mlir \
  -o /tmp/kernelsmith-rmsnorm-rvv.ll
```

Build, all 41 lit tests, all 21 CTest tests, ruff lint, and the static/dynamic
RVV-to-LLVM IR pipeline passed. The repository-wide ruff format check still
reports two pre-existing formatting issues in unchanged documentation files:
`docs/design/DES-002-matmul-kernel.md` and `docs/guides/testing-guide.md`.

## Notes

- Numerical-stability behavior (f32 accumulation, epsilon placement) must match
  the C reference so existing golden cases stay valid.
- Softmax compiler lowering is tracked separately in `TASK-023`.
- QEMU validation of the generated RMSNorm object is part of the remaining
  `TASK-008` QEMU helper-kernel criterion.

## Log

### 2026-06-24
- Created as the RMSNorm compiler-lowering follow-up split from `TASK-008` per
  the M6 dashboard next action.
- Implemented `RMSNormToLinalgPattern` in `LowerToLinalgPass.cpp`: a
  sum-of-squares reduction `linalg.generic`, an elementwise
  `rsqrt(ssq / inner + eps)` scale generic, and an apply generic that broadcasts
  the per-row scale and the trailing weight. Static and dynamic trailing
  dimensions are both handled. Added `tests/lit/Passes/lower-rms-norm.mlir`
  (2D, 3D, dynamic).

### 2026-08-08
- Promoted f16/bf16 inputs and weights to f32 for reduction and scaling, then
  truncated the final value to the declared result type.
- Completed the RVV backend path for normalization by preparing bounded vector
  reductions/transfers, lowering math and UB operations, and checking that no
  non-LLVM operations survive before `mlir-translate`.
- Verified the focused lit suite (41/41), full CTest suite (21/21), ruff lint,
  and `--ks-lower-to-linalg --ks-vectorize --ks-lower-to-rvv` plus
  `mlir-translate` pipeline.
