# TASK-022: Lower ks.rms_norm to Vectorizable Linalg/Loop Form

## Status
[?] Needs Review

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
      (reduction + two elementwise) with element-type accumulation for the
      sum-of-squares reduction.
- [x] Apply epsilon and `rsqrt` scaling consistently with the
      `ks_rms_norm_f32` reference and the golden descriptors
      (`rsqrt(ssq / inner + eps)`).
- [x] Preserve the weight (gain) operand semantics: `output = input * scale *
      weight[d]` broadcast over the trailing dimension.
- [x] Add a pass-transformation lit test asserting `ks.rms_norm` is removed and
      the expected linalg/arith/math ops appear (2D, 3D, dynamic trailing dim).
- [ ] Confirm the lowered form vectorizes through `--ks-vectorize` and reaches
      LLVM via `--ks-lower-to-rvv` (lit coverage or documented pipeline run).
- [ ] Run build, full CTest, and Python lint/format before completion.

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
- Verification status: local `cmake`/`ctest` could not be run because installing
  LLVM/MLIR 21 is blocked in this environment — the agent proxy denies
  `apt.llvm.org` by organization policy (403 CONNECT), so `./scripts/setup.sh`
  cannot complete. CI `build-and-test` (LLVM 21 + full CTest incl. lit) covers
  the build/test gate on pull request or `workflow_dispatch`. The
  `--ks-vectorize` / `--ks-lower-to-rvv` confirmation and the build/CTest
  checkboxes remain open pending that CI run.
