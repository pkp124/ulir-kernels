# TASK-016: Implement INT8 Dot/GEMV Lowering

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
`.cursor/agents/mlir-pass-agent.md` or a future quantization-focused agent.

## Description

Implement the INT8 generated-kernel path for dot and GEMV using i8 inputs,
i32 accumulation, and explicit requantization where the public API requires an
integer output.

## Acceptance Criteria
- [ ] Add or update dialect/lowering metadata for i8 accumulator and
      quantization policy.
- [ ] Lower i8 dot/GEMV to vectorizable or RVV-friendly IR with i32
      accumulation.
- [ ] Add lit tests for supported lowering patterns and diagnostics for
      unsupported metadata/layout cases.
- [ ] Add host/reference or generated C API tests using the ABI from
      `TASK-013`.
- [ ] Verify outputs against quantized golden cases from `TASK-014`.
- [ ] Add QEMU RVV validation when the generated RISC-V binary is runnable in
      CI or locally.

## Dependencies
- `TASK-013`: C API and layout contract
- `TASK-014`: quantized golden validation
- `TASK-015`: RVV target profile parameters
- `specs/kernels/quantization.md`
- `specs/targets/riscv-rvv.md`

## Verification
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

Run `PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional` once
RISC-V quantized runner support exists.

## Notes
- Keep this task focused on INT8. W4A8 fused unpack/dequantize is tracked by
  `TASK-017`.

## Log

### 2026-06-14
- Started first compiler slice: `ks.dot_i8` dialect coverage and linalg lowering
  with i32 accumulation.

### 2026-06-09
- Created as the INT8 lowering child task for M5.
