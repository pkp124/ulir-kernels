# TASK-017: Implement W4A8 Fused Dot/GEMV Lowering

## Status
[~] In Progress

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
- [ ] Implement W4A8 packed-weight loading and nibble unpacking according to the
      ABI from `TASK-013`.
- [ ] Fuse unpack/dequantize with dot/GEMV compute in the generated path.
- [ ] Support the selected accumulation policy, either i32 or f32, with explicit
      rounding and saturation behavior.
- [ ] Add lit tests for fused lowering patterns and unsupported-layout
      diagnostics.
- [ ] Validate against W4A8 golden cases from `TASK-014`.
- [ ] Add QEMU RVV validation and benchmark reporting for at least one W4A8
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

Run `PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional` once
W4A8 RISC-V runner support exists.

## Notes
- This is the primary batch-1 transformer decode kernel track. Keep it ahead of
  broad W4A8 GEMM unless prefill work becomes the immediate product need.

## Log

### 2026-06-14
- Started the first W4A8 compiler-visible slice with `ks.dot_w4a8` dialect
  coverage, verifier diagnostics, and quantization spec updates.
- Remaining task work includes fused linalg lowering, GEMV coverage, golden
  W4A8 validation through generated paths, and QEMU RVV reporting.

### 2026-06-09
- Created as the W4A8 fused lowering child task for M5.
