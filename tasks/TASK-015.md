# TASK-015: Add Quantized RVV Target Profile Parameters

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
General or `.cursor/agents/rvv-validation-agent.md`.

## Description

Add target-profile parameters needed by i8 and W4A8 dot/GEMV/GEMM codegen:
vector element widths, LMUL choices, tile sizes, pack factors, group sizes,
alignment, and workspace implications.

## Acceptance Criteria
- [x] Extend `target/riscv_rvv_256.h` with i8 and W4A8 tile, pack, and
      alignment parameters.
- [x] Document each new profile macro in the target profile specification.
- [x] Ensure future pass options can be derived from the profile without
      hardcoding RVV VLEN.
- [x] Add tests or script checks that validate required profile fields exist.
- [x] Update docs to explain how quantized profile values affect dot/GEMV first
      and GEMM later.

## Dependencies
- `TASK-007`: M5 parent tracker
- `TASK-013`: ABI and layout choices
- `specs/targets/riscv-rvv.md`
- `specs/targets/system-description.md`

## Verification
```bash
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

Verified on 2026-06-13:

```bash
.venv/bin/python scripts/validate_profile.py target/riscv_rvv_256.h target/generic.h
.venv/bin/pytest tests/test_validate_profile.py -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Docker parity was not available on this VM (`docker: command not found`).

## Notes
- Do not hardcode RVV VLEN in compiler code; profile values must remain
  vector-length agnostic where the ISA requires it.
- Keep dot/GEMV parameters independent enough that W4A8 decode kernels do not
  inherit unsuitable GEMM defaults.

## Log

### 2026-06-13
- Started implementation of RVV quantized profile parameters and validation.
- Completed RVV i8/W4A8 profile macros, target-profile documentation, validator
  script, pytest coverage, and CTest integration.

### 2026-06-09
- Created as the quantized target-profile child task for M5.
