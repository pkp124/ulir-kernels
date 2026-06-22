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
- [ ] Lower RMSNorm with f32 accumulation and explicit epsilon behavior.
- [ ] Lower softmax using max-subtract-exp-sum-divide for numerical stability.
- [ ] Add functional validator coverage for transformer helper kernels.
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
- Started `TASK-021` to add descriptor-backed host and RVV/QEMU CI coverage for
  the transformer helper C APIs. RMSNorm and softmax compiler lowerings remain
  separate M6 follow-ups.

### 2026-06-06
- Created from M6 roadmap tasks.
