# TASK-013: Define Quantized Dot/GEMV C APIs and Layouts

## Status
[x] Complete

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
- [x] Add or update public headers for i8 and W4A8 dot/GEMV APIs.
- [x] Document input, weight, output, scale, zero-point, and accumulator types.
- [x] Define W4A8 nibble packing order, group/block size policy, and scale
      layout.
- [x] Define workspace query behavior and alignment requirements.
- [x] Add C API smoke tests for argument validation and any reference
      implementation added in this task.
- [x] Update docs/specs so ABI and layout decisions are discoverable before
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

Verified on 2026-06-10:

```bash
cmake --build build --parallel
build/tests/capi/test-quantized
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Notes
- Prefer dot/GEMV before large GEMM because batch-1 transformer decode is the
  first optimization target.
- Keep standalone `ks.quantize` / `ks.dequantize` as semantic test ops; the
  performance path should fuse unpack/dequantize inside dot/GEMV loops.

## Log

### 2026-06-09
- Created as the first M5 child task after `TASK-006` completed RVV validation.

### 2026-06-10
- Started implementation of the public quantized dot/GEMV C API and layout
  contract.
- Added `include/kernelsmith/ks_quantized.h`, scalar reference implementations,
  C API smoke tests, and documentation for i8 and W4A8 dot/GEMV layouts.
- Marked complete after build, focused C API smoke test, full CTest, and Python
  lint/format checks passed.
