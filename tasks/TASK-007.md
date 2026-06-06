# TASK-007: RISC-V Quantization Foundation

## Status
[ ] Not Started

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
General or a future quantization-focused agent.

## Description

Define and implement the metadata, C APIs, layouts, validation, and lowering
foundation needed for INT8 and W4A8 RISC-V inference kernels.

## Acceptance Criteria

- [ ] Add accumulator and quantization metadata needed for i8 and W4A8 lowering.
- [ ] Define public C APIs and packed layouts for i8 and W4A8 dot/GEMV.
- [ ] Implement INT8 lowering with i8 input, i32 accumulation, and requantization
      where needed.
- [ ] Implement W4A8 fused unpack/dequantize plus i32 or f32 accumulation.
- [ ] Add RVV i8/W4A8 tile parameters and pack factors to target profiles.
- [ ] Add NumPy validation for quantized dot, GEMV, and GEMM.
- [ ] Add lit tests for quantized lowering and diagnostics.

## Dependencies

- TASK-006: RVV validation baseline
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
- Split large subtopics into child tasks if one PR becomes too broad.

## Log

### 2026-06-06
- Created from M5 roadmap tasks.
