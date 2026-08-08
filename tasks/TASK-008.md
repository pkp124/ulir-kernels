# TASK-008: Transformer Minimum Kernel Set

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General, `.cursor/agents/mlir-pass-agent.md`, or a future transformer-kernels
agent depending on subtask.

## Description

Add the minimum non-matmul kernels and C APIs needed for a minimal
llama-style transformer block on RISC-V RVV.

## Acceptance Criteria

- [x] Add `ks.add` and `ks.mul` ops with broadcasting rules, verifiers, and lit
      tests.
- [x] Add C APIs for `ks_add_f32`, `ks_mul_f32`, `ks_rms_norm_f32`, and
      `ks_softmax_f32`.
- [x] Lower add, mul, and SiLU to vectorizable linalg or loop forms.
- [x] Lower RMSNorm with f32 accumulation and explicit epsilon behavior.
- [ ] Lower softmax using max-subtract-exp-sum-divide for numerical stability.
- [x] Add functional validator coverage for transformer helper kernels.
- [ ] Add QEMU tests for generated RVV helper kernels when toolchains exist.

## Dependencies

- TASK-007: Quantization foundation, for final transformer integration.
- `docs/design/DES-011-riscv-first-transformer-demo.md`

## Verification

```bash
ctest --test-dir build --output-on-failure
```

Add C API and NumPy validation commands as each helper kernel lands.

## Notes

- Split op definition, C API, lowering, and QEMU validation into child tasks if
  implementation spans multiple PRs.

## Log

### 2026-08-08
- Completed `TASK-022`; `ks.rms_norm` now lowers through vectorizable linalg
  with f32 accumulation for f32 and narrower inputs, explicit epsilon before
  `rsqrt`, and static/dynamic RVV pipeline validation.

### 2026-06-24
- Split the two remaining compiler-lowering criteria into PR-sized child tasks:
  `TASK-022` (lower `ks.rms_norm`) and `TASK-023` (lower `ks.softmax`). The
  QEMU helper-kernel validation criterion stays on this parent until those
  lowerings generate RVV objects to validate.

### 2026-06-15
- Started M6 by splitting the first compiler helper slice into `TASK-019` for
  `ks.add`/`ks.mul` dialect definitions, verifiers, and linalg lowering.
- Completed `TASK-019`; `ks.add` and `ks.mul` now parse, verify static
  broadcasting, and lower to vectorizable `linalg.generic`. Existing SiLU
  lowering remains covered by `--ks-lower-activations`.

### 2026-06-16
- Completed `TASK-020`; `ks_add_f32`, `ks_mul_f32`, `ks_rms_norm_f32`, and
  `ks_softmax_f32` now have public C headers, scalar reference implementations,
  C API smoke tests, and NumPy reference validation.

### 2026-06-22
- Completed `TASK-021`; transformer helper C APIs now have descriptor-backed
  golden bundles, host CTest coverage, and RVV/QEMU CI coverage at VLEN 256 and
  512. RMSNorm and softmax compiler lowerings remain separate M6 follow-ups.

### 2026-06-06
- Created from M6 roadmap tasks.
