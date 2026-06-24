# TASK-023: Lower ks.softmax with Numerically Stable Form

## Status
[ ] Not Started

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

- [ ] Lower `ks.softmax` in `--ks-lower-to-linalg` to linalg/loop form
      implementing max-subtract-exp-sum-divide along the softmax axis.
- [ ] Use f32 accumulation for the row max and exponent-sum reductions.
- [ ] Respect the op's reduction axis attribute (if defined) or document the
      fixed-axis assumption.
- [ ] Add a pass-transformation lit test asserting `ks.softmax` is removed and
      the max/exp/sum/div sequence appears.
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
