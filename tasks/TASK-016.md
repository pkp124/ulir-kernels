# TASK-016: Implement INT8 Dot/GEMV Lowering

## Status
[x] Complete

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
- [x] Add or update dialect/lowering metadata for i8 accumulator and
      quantization policy.
- [x] Lower i8 dot/GEMV to vectorizable or RVV-friendly IR with i32
      accumulation.
- [x] Add lit tests for supported lowering patterns and diagnostics for
      unsupported metadata/layout cases.
- [x] Add host/reference or generated C API tests using the ABI from
      `TASK-013`.
- [x] Verify outputs against quantized golden cases from `TASK-014`.
- [x] Add QEMU RVV validation when the generated RISC-V binary is runnable in
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

Verified on 2026-06-13:

```bash
cmake --build build --parallel
.venv/bin/lit -v build/tests/lit/Dialect/Kernel/quantized-dot-gemv.mlir build/tests/lit/Dialect/Kernel/quantized-dot-gemv-invalid.mlir build/tests/lit/Passes/lower-quantized-dot-gemv.mlir --param ks_tools_dir=/workspace/build/bin
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
```

Docker parity was not available on this VM (`docker: command not found`).

## Notes
- Keep this task focused on INT8. W4A8 fused unpack/dequantize is tracked by
  `TASK-017`.

## Log

### 2026-06-13
- Started INT8 dot/GEMV dialect and lowering implementation.
- Added `ks.dot_i8` and `ks.matvec_i8` ops, verifiers, lit tests, and
  `--ks-lower-to-linalg` lowering to i32 `linalg.generic`.
- Added INT8 dot/GEMV cases to RVV functional validation and verified them with
  QEMU at VLEN 256 and 512.

### 2026-06-09
- Created as the INT8 lowering child task for M5.
