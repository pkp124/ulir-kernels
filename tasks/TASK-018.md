# TASK-018: Integrate Generated INT8 RVV Objects with the C API

## Status
[ ] Not Started

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
`.cursor/agents/rvv-validation-agent.md` or a future quantization-focused agent.

## Description

Replace or dispatch the public INT8 dot/GEMV C API symbols to generated RVV
objects for the `riscv_rvv_256` target while preserving the scalar reference
implementation for generic targets.

## Acceptance Criteria
- [ ] Define the generated-object naming, symbol, and dispatch policy for
      `ks_dot_i8` and `ks_matvec_i8`.
- [ ] Extend the build path so generated INT8 RVV objects can be linked into
      `libkernelsmith.a` without changing the public headers.
- [ ] Keep generic/profile reference implementations available as fallback
      objects.
- [ ] Add C API or golden tests proving the public symbols use the generated
      target objects when the RVV profile is selected.
- [ ] Validate generated INT8 RVV objects under QEMU at VLEN 256 and 512.

## Dependencies
- `TASK-013`: C API and layout contract
- `TASK-014`: quantized golden validation
- `TASK-015`: RVV target profile parameters
- `TASK-016`: INT8 dialect and linalg lowering
- `docs/design/DES-014-int8-dot-and-gemv-lowering.md`

## Verification
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
ruff check .
ruff format --check .
```

## Notes
- Keep generated-object integration separate from the reference C API and
  descriptor-backed QEMU validation that `TASK-016` completed.
- Do not hardcode RVV VLEN; derive tile and profile choices from
  `target/riscv_rvv_256.h` or generated pass options.

## Log

### 2026-06-14
- Created as the generated INT8 RVV object replacement follow-up after
  `TASK-016` completed INT8 dialect/linalg lowering and QEMU validation of the
  quantized C API runner.
