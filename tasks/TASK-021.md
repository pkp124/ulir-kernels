# TASK-021: Add Transformer Helper Golden CI Coverage

## Status
[x] Complete

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

- [x] Add golden descriptors and generated bundles for `ks_add_f32`,
      `ks_mul_f32`, `ks_rms_norm_f32`, and `ks_softmax_f32`.
- [x] Register transformer helper golden cases in host CTest.
- [x] Extend the descriptor runner and verifier argument mapping for helper
      kernels.
- [x] Add helper cases to the RVV/QEMU CI workflow at VLEN 256 and 512.
- [x] Run focused golden tests, focused helper CTest, and full CTest.

## Dependencies

- `TASK-008`
- `TASK-020`
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `docs/design/DES-013-golden-reference-verification-infrastructure.md`

## Verification

```bash
./scripts/setup.sh
python3 -m pytest tests/test_golden_verify.py
ctest --test-dir build -R 'kernelsmith-host-golden-(add|mul|rms_norm|softmax)_f32_smoke|capi-transformer-helpers' --output-on-failure
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
python tests/verify.py --case tests/golden/cases/${case}.json --target riscv_rvv_256 --riscv-runner build-rvv/bin/riscv-golden-runner --host-runner build-rvv/bin/host-reference-runner --qemu qemu-riscv64 --vlens 256 512
```

Verified on 2026-06-22:

```bash
./scripts/setup.sh
source .venv/bin/activate && python -m pytest tests/test_golden_verify.py
source .venv/bin/activate && cmake --build build --parallel
source .venv/bin/activate && ctest --test-dir build -R 'kernelsmith-host-golden-(add|mul|rms_norm|softmax)_f32_smoke|capi-transformer-helpers' --output-on-failure
source .venv/bin/activate && ctest --test-dir build --output-on-failure
source .venv/bin/activate && ruff check .
source .venv/bin/activate && ruff format --check .
source .venv/bin/activate && for case in add_f32_smoke mul_f32_smoke rms_norm_f32_smoke softmax_f32_smoke; do python tests/verify.py --case tests/golden/cases/${case}.json --target riscv_rvv_256 --riscv-runner build-rvv/bin/riscv-golden-runner --host-runner build-rvv/bin/host-reference-runner --qemu qemu-riscv64 --vlens 256 512 --output-dir build/golden-rvv-local --report build/golden-rvv-local/${case}.json; done
```

`./scripts/docker-verify.sh` was not run because Docker is unavailable on this
VM (`docker: command not found`).

## Notes

- This validates the scalar C helper APIs on host and through the RISC-V runner
  binary. Compiler lowering for `ks.rms_norm` and `ks.softmax` remains separate
  M6 work.

## Log

### 2026-06-22
- Started descriptor-backed CI coverage for transformer helper kernels after
  `TASK-020` added the C APIs and smoke tests.
- Completed descriptor-backed host and RVV/QEMU CI coverage for add, mul,
  RMSNorm, and softmax helper C APIs. Full native CTest, focused helper CTest,
  golden pytest, ruff, and local QEMU VLEN 256/512 checks passed.
