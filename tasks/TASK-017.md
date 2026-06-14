# TASK-017: Implement W4A8 Fused Dot/GEMV Lowering

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
`.cursor/agents/rvv-validation-agent.md` or a future quantization-focused agent.

## Description

Implement the W4A8 decode-path generated-kernel path. The kernel should unpack
INT4 weights, apply scale metadata, and accumulate against INT8 activations
without materializing dequantized weights.

## Acceptance Criteria
- [x] Implement W4A8 packed-weight loading and nibble unpacking according to the
      ABI from `TASK-013`.
- [x] Fuse unpack/dequantize with dot/GEMV compute in the generated path.
- [x] Support the selected accumulation policy, either i32 or f32, with explicit
      rounding and saturation behavior.
- [x] Add lit tests for fused lowering patterns and unsupported-layout
      diagnostics.
- [x] Validate against W4A8 golden cases from `TASK-014`.
- [x] Add QEMU RVV validation and benchmark reporting for at least one W4A8
      smoke case when toolchains are available.

## Dependencies
- `TASK-013`: C API and packed W4A8 layout
- `TASK-014`: W4A8 golden validation
- `TASK-015`: RVV W4A8 target profile parameters
- `TASK-016`: shared quantized lowering conventions where applicable
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `specs/kernels/quantization.md`

## Verification
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

W4A8 RISC-V runner support now runs through
`PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional`.

Verified on 2026-06-14 for the `ks.dot_w4a8` dialect slice:

```bash
./scripts/setup.sh
ctest --test-dir build --output-on-failure -R kernelsmith-lit
ctest --test-dir build --output-on-failure
cmake --build build --parallel
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker parity and local `clang-format` were unavailable on this VM
(`docker: command not found`, `clang-format: command not found`).

Verified on 2026-06-14 for the `ks.matvec_w4a8` GEMV/linalg and RVV golden
validation slice:

```bash
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R kernelsmith-lit
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
git diff --check
```

`./scripts/setup-rvv-sim.sh` installed QEMU and the RISC-V cross-compiler before
the RVV functional run. Docker parity and local `clang-format` were unavailable
on this VM (`docker: command not found`, `clang-format: command not found`).

Verified on 2026-06-14 for the fused `ks.dot_w4a8` linalg lowering slice:

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R kernelsmith-lit
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Docker parity and local `clang-format` were unavailable on this VM
(`docker: command not found`, `clang-format: command not found`).

## Notes
- This is the primary batch-1 transformer decode kernel track. Keep it ahead of
  broad W4A8 GEMM unless prefill work becomes the immediate product need.

## Log

### 2026-06-14
- Added `ks.matvec_w4a8` dialect, verifier, and fused linalg lowering coverage,
  including row-parallel packed nibble extraction and f32 accumulation.
- Extended W4A8 golden descriptors to `riscv_rvv_256` with VLEN 256/512
  validation, and added W4A8 dot/GEMV execution to `--riscv-functional`.
- Recorded W4A8 GEMV benchmark reporting in the generated RVV functional report.
- Added fused `ks.dot_w4a8` lowering to `linalg.generic`, including packed-byte
  extraction, signed nibble unpacking, per-group scale extraction, and f32
  accumulation.
- Tightened quantized compute verifier diagnostics to reject non-signless MLIR
  integer tensor types before lowering; the C ABI remains `int8_t`/`uint8_t`.
- Started the first W4A8 compiler-visible slice with `ks.dot_w4a8` dialect
  coverage, verifier diagnostics, and quantization spec updates.

### 2026-06-09
- Created as the W4A8 fused lowering child task for M5.
