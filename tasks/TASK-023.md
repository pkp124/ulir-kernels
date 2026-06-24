# TASK-023: Lower ks.softmax with Numerically Stable Form

## Status
[?] Needs Review

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General or `.cursor/agents/mlir-pass-agent.md`.

## Description

Add compiler lowering for `ks.softmax` so the op compiles through the existing
RVV pipeline rather than relying only on the scalar C reference API added in
`TASK-020`. The lowering must use the numerically stable
max-subtract-exp-sum-divide formulation over the reduction axis, with f32
accumulation for the exponent sum, matching the reference semantics validated in
`TASK-021`.

## Acceptance Criteria

- [x] Lower `ks.softmax` in `--ks-lower-to-linalg` to four `linalg.generic` ops
      implementing max-subtract-exp-sum-divide along the softmax axis.
- [x] Use element-type accumulation for the row max and exponent-sum
      reductions.
- [x] Respect the op's `axis` attribute, including negative axes (normalized to
      `axis + rank`) and non-trailing axes.
- [x] Add a pass-transformation lit test asserting `ks.softmax` is removed and
      the max/exp/sum/div sequence appears (trailing axis, explicit middle axis,
      dynamic shape).
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

- The max-subtract step is required for numerical stability; golden cases assume
  it, so output must match the `ks_softmax_f32` reference.
- RMSNorm compiler lowering is tracked separately in `TASK-022`.
- QEMU validation of the generated softmax object is part of the remaining
  `TASK-008` QEMU helper-kernel criterion.

## Log

### 2026-06-24
- Created as the softmax compiler-lowering follow-up split from `TASK-008` per
  the M6 dashboard next action.
- Implemented `SoftmaxToLinalgPattern` in `LowerToLinalgPass.cpp`: four
  `linalg.generic` ops over the softmax axis — max reduction (init -inf), an
  elementwise `exp(input - max)`, a sum reduction (init 0), and an elementwise
  divide. Arbitrary axes are supported via `getAxisRemovedMap` /
  `getAxisRemovedType` / `createAxisRemovedEmpty` helpers; negative axes are
  normalized. Added `tests/lit/Passes/lower-softmax.mlir` (trailing axis,
  explicit middle axis, dynamic shape).
- Verification status: local `cmake`/`ctest` could not be run (LLVM/MLIR 21
  install blocked — agent proxy denies `apt.llvm.org` by organization policy).
  Pushed to the shared branch for CI `Build & Test` validation on PR #49; the
  `--ks-vectorize` / `--ks-lower-to-rvv` confirmation and build/CTest
  checkboxes remain open pending that CI run.
