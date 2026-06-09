# TASK-007: RISC-V Quantization Foundation

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
General or a future quantization-focused agent.

## Description

Parent tracker for the M5 quantization foundation. M5 defines and implements
the metadata, C APIs, layouts, validation, and lowering foundation needed for
INT8 and W4A8 RISC-V inference kernels.

## Acceptance Criteria

- [x] `ks.quantize` and `ks.dequantize` ops exist with verifier and lit
      coverage for semantic conversion tests.
- [ ] `TASK-013`: Define public C APIs and packed layouts for i8 and W4A8
      dot/GEMV.
- [ ] `TASK-014`: Add NumPy/golden validation for quantized dot, GEMV, and GEMM.
- [ ] `TASK-015`: Add RVV i8/W4A8 tile parameters and pack factors to target
      profiles.
- [ ] `TASK-016`: Implement INT8 dot/GEMV lowering with i8 inputs, i32
      accumulation, and requantization where needed.
- [ ] `TASK-017`: Implement W4A8 fused unpack/dequantize plus i32 or f32
      accumulation for RVV.

## Dependencies

- TASK-006: RVV validation baseline
- TASK-013 through TASK-017 for implementation slices
- `specs/kernels/quantization.md`
- `specs/targets/riscv-rvv.md`

## Verification

```bash
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

Add focused QEMU validation when generated RVV quantized kernels are runnable in
the local or CI environment.

## Notes

- Keep public ABI and packed layout documentation in sync with implementation.
- This file is now a parent tracker; implementation should land through the
  child tasks.

## Log

### 2026-06-06
- Created from M5 roadmap tasks.

### 2026-06-09
- Started M5 after `TASK-006` completed the RVV validation baseline.
- Split the broad parent into PR-sized child tasks: `TASK-013` ABI/layout,
  `TASK-014` quantized golden validation, `TASK-015` target-profile parameters,
  `TASK-016` INT8 lowering, and `TASK-017` W4A8 lowering.
- Recorded existing `ks.quantize`/`ks.dequantize` TableGen, verifier, and lit
  coverage as already complete M5 semantic-op work.
