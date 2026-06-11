# TASK-014: Add Quantized Golden Validation Cases

## Status
[x] Complete

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
- [x] Extend golden descriptor schema semantics for quantized dot, GEMV, and
      GEMM cases.
- [x] Add deterministic NumPy reference generation for at least i8 dot/GEMV and
      W4A8 dot/GEMV smoke cases.
- [x] Record scale, zero-point, rounding, saturation, accumulator dtype, and
      packed-layout metadata in manifests.
- [x] Add comparator coverage for exact accumulator checks and dequantized
      allclose checks where policy allows tolerance.
- [x] Add pytest coverage for descriptor validation, manifest hashes, passing
      comparisons, and failing diagnostics.
- [x] Integrate host/reference quantized smoke cases into CTest once a runner is
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

Verified on 2026-06-11:

```bash
.venv/bin/pytest tests/test_golden_verify.py -q
.venv/bin/ruff check .
.venv/bin/ruff format --check .
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Notes
- Golden policy must be explicit before lowering work can claim correctness.
- Keep committed golden bundles small; larger randomized cases can be generated
  in CI or nightly runs.

## Log

### 2026-06-09
- Created as the quantized validation child task for M5.

### 2026-06-11
- Completed quantized golden descriptor semantics for i8 dot/GEMV/GEMM and
  W4A8 dot/GEMV smoke cases.
- Added deterministic NumPy generation, manifest quantization/layout metadata,
  host-reference runner support, CTest integration, and pytest coverage for
  descriptor validation, manifest hashes, comparisons, and diagnostics.
