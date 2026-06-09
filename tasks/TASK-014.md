# TASK-014: Add Quantized Golden Validation Cases

## Status
[ ] Not Started

## Priority
P0 (Milestone Blocking)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
General or a future quantization-focused agent.

## Description

Extend the golden-reference infrastructure so i8 and W4A8 dot/GEMV/GEMM cases
can be generated, checked, and reported with explicit rounding and saturation
policy.

## Acceptance Criteria
- [ ] Extend golden descriptor schema semantics for quantized dot, GEMV, and
      GEMM cases.
- [ ] Add deterministic NumPy reference generation for at least i8 dot/GEMV and
      W4A8 dot/GEMV smoke cases.
- [ ] Record scale, zero-point, rounding, saturation, accumulator dtype, and
      packed-layout metadata in manifests.
- [ ] Add comparator coverage for exact accumulator checks and dequantized
      allclose checks where policy allows tolerance.
- [ ] Add pytest coverage for descriptor validation, manifest hashes, passing
      comparisons, and failing diagnostics.
- [ ] Integrate host/reference quantized smoke cases into CTest once a runner is
      available.

## Dependencies
- `TASK-007`: M5 parent tracker
- `TASK-013`: C ABI and layout decisions
- `docs/design/DES-013-golden-reference-verification-infrastructure.md`
- `specs/kernels/quantization.md`

## Verification
```bash
pytest tests/test_*golden*.py tests/test_*verify*.py
ruff check .
ruff format --check .
ctest --test-dir build --output-on-failure
```

## Notes
- Golden policy must be explicit before lowering work can claim correctness.
- Keep committed golden bundles small; larger randomized cases can be generated
  in CI or nightly runs.

## Log

### 2026-06-09
- Created as the quantized validation child task for M5.
