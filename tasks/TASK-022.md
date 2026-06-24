# TASK-022: Lower ks.rms_norm to Vectorizable Linalg/Loop Form

## Status
[ ] Not Started

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

- [ ] Lower `ks.rms_norm` in `--ks-lower-to-linalg` to `linalg.generic` /
      `linalg.reduce` (or equivalent loop form) with f32 accumulation for the
      sum-of-squares reduction.
- [ ] Apply epsilon and `rsqrt` scaling consistently with the
      `ks_rms_norm_f32` reference and the golden descriptors.
- [ ] Preserve the optional weight (gain) operand semantics if present in the op
      definition.
- [ ] Add a pass-transformation lit test asserting `ks.rms_norm` is removed and
      the expected linalg/arith/math ops appear.
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
