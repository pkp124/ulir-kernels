# TASK-013: Define Quantized Dot/GEMV C APIs and Layouts

## Status
[ ] Not Started

## Priority
P0 (Milestone Blocking)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
General or a future quantization-focused agent.

## Description

Define the public C ABI, packed weight layouts, scale metadata, and workspace
rules for the first quantized dot and GEMV kernels:

- `ks_dot_i8`
- `ks_matvec_i8`
- `ks_dot_w4a8`
- `ks_matvec_w4a8`

This task is design/API work first. Lowering and generated-kernel
implementation are tracked by later M5 child tasks.

## Acceptance Criteria
- [ ] Add or update public headers for i8 and W4A8 dot/GEMV APIs.
- [ ] Document input, weight, output, scale, zero-point, and accumulator types.
- [ ] Define W4A8 nibble packing order, group/block size policy, and scale
      layout.
- [ ] Define workspace query behavior and alignment requirements.
- [ ] Add C API smoke tests for argument validation and any reference
      implementation added in this task.
- [ ] Update docs/specs so ABI and layout decisions are discoverable before
      lowering work starts.

## Dependencies
- `TASK-006`: RVV validation baseline
- `TASK-007`: M5 parent tracker
- `docs/design/DES-006-kernel-library-architecture.md`
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `specs/kernels/quantization.md`

## Verification
```bash
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

## Notes
- Prefer dot/GEMV before large GEMM because batch-1 transformer decode is the
  first optimization target.
- Keep standalone `ks.quantize` / `ks.dequantize` as semantic test ops; the
  performance path should fuse unpack/dequantize inside dot/GEMV loops.

## Log

### 2026-06-09
- Created as the first M5 child task after `TASK-006` completed RVV validation.
