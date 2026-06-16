# TASK-020: Add Transformer Helper C APIs

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General.

## Description

Add scalar reference C APIs for the f32 transformer helper kernels needed by the
minimal transformer block: residual add, elementwise multiply, RMSNorm, and
softmax.

## Acceptance Criteria

- [x] Add public C headers for `ks_add_f32`, `ks_mul_f32`,
      `ks_rms_norm_f32`, and `ks_softmax_f32`.
- [x] Implement C99 reference kernels with no internal allocation.
- [x] Use f32 accumulation for RMSNorm and explicit positive finite epsilon
      validation.
- [x] Implement softmax with max-subtract-exp-sum-divide numerical stability.
- [x] Add C API smoke tests covering correctness and invalid arguments.
- [x] Add NumPy reference validation for transformer helper kernels.
- [x] Run build, focused C API tests, full CTest, and Python lint/format checks.

## Dependencies

- `TASK-008`
- `TASK-019`
- `docs/design/DES-006-kernel-library-architecture.md`
- `docs/design/DES-011-riscv-first-transformer-demo.md`

## Verification

```bash
cmake --build build --parallel
ctest --test-dir build -R 'capi-(transformer-helpers|numpy-validation)' --output-on-failure
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Verified on 2026-06-16:

```bash
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build -R 'capi-(transformer-helpers|numpy-validation)' --output-on-failure
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Notes

- This task adds scalar reference C APIs only. Generated RVV helper-object
  validation remains a separate follow-up from `TASK-008`.

## Log

### 2026-06-16
- Created as the C API/reference implementation follow-up for `TASK-008` after
  `TASK-019` completed `ks.add`/`ks.mul` dialect and lowering work.
- Added public f32 helper headers, scalar reference implementations, C API smoke
  tests, and NumPy reference validation for add, mul, RMSNorm, and softmax.
