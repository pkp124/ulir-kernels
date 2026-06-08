# TASK-012: Compare RISC-V RVV Outputs Against Golden References

## Status
[~] In Progress

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

- [ ] Update the RISC-V runner or harness contract to consume descriptor-defined
      inputs from the golden bundle.
- [ ] Capture actual RVV outputs or complete comparison metrics for at least f32
      ReLU and f32 matmul smoke cases.
- [ ] Compare RVV output against NumPy golden output at VLEN 256 and 512.
- [ ] Optionally compare RVV output against host output for the same case.
- [ ] Preserve profile-aware VLEN validation and executable-bit repair for
      downloaded CI artifacts.
- [ ] Emit stable result lines with pass/fail, max error, mismatch count, VLEN,
      and case name.
- [ ] Update `.github/workflows/ci-rvv-sim.yml` to run the golden-backed RVV
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

## Notes

- This task replaces handwritten embedded expected constants with generated
  golden data for scalable validation.
- The first matmul case should be small enough for PR CI but include dimensions
  that exercise the lowering path meaningfully.

## Log

### 2026-06-07
- Created as the RISC-V execution leg of the DES-013 verification design.
