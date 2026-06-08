# TASK-012: Compare RISC-V RVV Outputs Against Golden References

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`.cursor/agents/rvv-validation-agent.md`

## Description

Upgrade the current RISC-V smoke test into true golden-reference verification.
Generated RVV binaries should consume the same descriptor inputs as host tests
and compare simulator outputs against NumPy golden outputs and, where useful,
host/x86 outputs.

## Acceptance Criteria

- [x] Update the RISC-V runner or harness contract to consume descriptor-defined
      inputs from the golden bundle.
- [x] Capture actual RVV outputs or complete comparison metrics for at least f32
      ReLU and f32 matmul smoke cases.
- [x] Compare RVV output against NumPy golden output at VLEN 256 and 512.
- [x] Optionally compare RVV output against host output for the same case.
- [x] Preserve profile-aware VLEN validation and executable-bit repair for
      downloaded CI artifacts.
- [x] Emit stable result lines with pass/fail, max error, mismatch count, VLEN,
      and case name.
- [x] Update `.github/workflows/ci-rvv-sim.yml` to run the golden-backed RVV
      verification cases.

## Dependencies

- `TASK-010`
- `TASK-011`
- `TASK-006`
- `DES-012-riscv-simulation-verification`
- `DES-013-golden-reference-verification-infrastructure`
- `scripts/compile-rvv.sh`
- `tests/riscv_runner.py`

## Verification

```bash
python tests/verify.py --target riscv_rvv_256 --case <case> --vlens 256 512
./scripts/run-tests.sh --riscv-functional
ctest --test-dir build --output-on-failure
```

Verified on 2026-06-08:

```bash
.venv/bin/pytest tests/test_golden_verify.py
.venv/bin/ruff check .
.venv/bin/ruff format --check .
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR=/usr/lib/llvm-21/lib/cmake/llvm -DMLIR_DIR=/usr/lib/llvm-21/lib/cmake/mlir -DLIT_COMMAND=/workspace/.venv/bin/lit
cmake --build build --parallel
ctest --test-dir build --output-on-failure
./scripts/setup-rvv-sim.sh
python3 tests/verify.py --case tests/golden/cases/relu_f32_smoke.json --target riscv_rvv_256 --riscv-runner build-rvv/bin/riscv-golden-runner --host-runner build-rvv/bin/host-reference-runner --qemu qemu-riscv64 --vlens 256 512 --output-dir build/golden-rvv --report build/golden-rvv/relu_riscv_report.json
python3 tests/verify.py --case tests/golden/cases/matmul_f32_smoke.json --target riscv_rvv_256 --riscv-runner build-rvv/bin/riscv-golden-runner --host-runner build-rvv/bin/host-reference-runner --qemu qemu-riscv64 --vlens 256 512 --output-dir build/golden-rvv --report build/golden-rvv/matmul_riscv_report.json
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
```

## Notes

- This task replaces handwritten embedded expected constants with generated
  golden data for scalable validation.
- The first matmul case should be small enough for PR CI but include dimensions
  that exercise the lowering path meaningfully.

## Log

### 2026-06-07
- Created as the RISC-V execution leg of the DES-013 verification design.

### 2026-06-08
- Implemented descriptor-backed RISC-V golden verification through
  `tests/riscv_runner.py` and `tests/verify.py`, including VLEN 256/512 QEMU
  runs, NumPy golden comparison, optional host comparison, executable-bit
  repair, and stable JSON result lines.
- Updated RVV CI and `./scripts/run-tests.sh --riscv-functional` to build the
  runner artifacts and execute the f32 ReLU and f32 matmul smoke cases.
