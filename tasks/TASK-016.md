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

Partial verification for the 2026-06-14 `ks.dot_i8` linalg lowering slice:

```bash
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R kernelsmith-lit
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker parity and local `clang-format` were unavailable on this VM
(`docker: command not found`, `clang-format: command not found`).

Partial verification for the 2026-06-14 `ks.matvec_i8` linalg lowering slice:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R kernelsmith-lit
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker parity and local `clang-format` were unavailable on this VM
(`docker: command not found`, `clang-format: command not found`).

Run `PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional` once
RISC-V quantized runner support exists.

Final verification for the 2026-06-14 INT8 RVV functional validation slice:

```bash
./scripts/setup.sh
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional --qemu-vlen 256
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Notes
- Keep this task focused on INT8. W4A8 fused unpack/dequantize is tracked by
  `TASK-017`.

## Log

### 2026-06-14
- Started first compiler slice: `ks.dot_i8` dialect coverage and linalg lowering
  with i32 accumulation.
- Added `ks.dot_i8`, verifier diagnostics, `--ks-lower-to-linalg` lowering to
  a `linalg.generic` i32 reduction, quantization spec updates, and DES-014.
- Verified the slice with focused lit, full CTest, and Python lint/format
  checks. Remaining task work includes INT8 GEMV lowering, RVV validation, and
  generated-kernel C API integration.
- Added `ks.matvec_i8`, verifier diagnostics, and `--ks-lower-to-linalg`
  lowering to a row-parallel, column-reduction `linalg.generic`.
- Verified the dot+GEMV linalg path with focused lit, full CTest including
  `matvec_i8_smoke` host golden validation, and Python lint/format checks.
  Remaining task work is generated/RVV validation and C API object integration.
- Added local and CI QEMU validation for `dot_i8_smoke` and
  `matvec_i8_smoke` through the RISC-V C API runner, including the quantized C
  implementation in the cross-linked binary.
- Marked the INT8 lowering task complete after VLEN 256 and default VLEN
  256/512 RISC-V functional checks, full native CTest, and Python lint/format
  checks passed. Generated INT8 RVV object replacement is tracked separately in
  `TASK-018`.

### 2026-06-09
- Created as the INT8 lowering child task for M5.
