# TASK-021: Add Transformer Helper Golden CI Coverage

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General.

## Description

Add descriptor-backed host and RISC-V QEMU validation for the transformer helper
C APIs so add, mul, RMSNorm, and softmax are covered by the same golden
verification infrastructure used by existing RVV smoke tests.

## Acceptance Criteria

- [ ] Add golden descriptors and generated bundles for `ks_add_f32`,
      `ks_mul_f32`, `ks_rms_norm_f32`, and `ks_softmax_f32`.
- [ ] Register transformer helper golden cases in host CTest.
- [ ] Extend the descriptor runner and verifier argument mapping for helper
      kernels.
- [ ] Add helper cases to the RVV/QEMU CI workflow at VLEN 256 and 512.
- [ ] Run focused golden tests, focused helper CTest, and full CTest.

## Dependencies

- `TASK-008`
- `TASK-020`
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `docs/design/DES-013-golden-reference-verification-infrastructure.md`

## Verification

```bash
python3 -m pytest tests/test_golden_verify.py
ctest --test-dir build -R 'kernelsmith-host-golden-(add|mul|rms_norm|softmax)_f32_smoke|capi-transformer-helpers' --output-on-failure
ctest --test-dir build --output-on-failure
```

## Notes

- This validates the scalar C helper APIs on host and through the RISC-V runner
  binary. Compiler lowering for `ks.rms_norm` and `ks.softmax` remains separate
  M6 work.

## Log

### 2026-06-22
- Started descriptor-backed CI coverage for transformer helper kernels after
  `TASK-020` added the C APIs and smoke tests.
