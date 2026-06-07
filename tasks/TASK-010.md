# TASK-010: Build Golden Reference Verification Infrastructure

## Status
[x] Complete

## Priority
P0 (Milestone Blocking)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`general`

## Description

Define and implement the target-neutral golden-reference data contract required
before adding more RISC-V functional cases. KernelSmith needs deterministic
NumPy-generated inputs, expected outputs, metadata, and comparison policies so
host and RISC-V executions are judged against the same source of truth.

## Acceptance Criteria

- [x] Add a descriptor schema for functional cases, including kernel name,
      inputs, outputs, target list, and compare policy.
- [x] Add a manifest format that records shape, dtype, seed, generator backend,
      tolerance, quantization policy fields, and SHA-256 hashes.
- [x] Add deterministic NumPy golden generators for at least f32 ReLU and f32
      matmul smoke cases.
- [x] Add comparator support for exact, allclose, quantized exact, and
      dequantized allclose modes.
- [x] Add focused unit tests for schema validation, hash validation, comparator
      pass/fail behavior, and diagnostic output.
- [x] Update docs to explain how to add a new golden-reference case.

## Dependencies

- `DES-013-golden-reference-verification-infrastructure`
- `tests/functional_validator.py`
- `tests/test_data_generator.py`

## Verification

```bash
ruff check .
ruff format --check .
pytest tests/test_*golden*.py tests/test_*verify*.py
ctest --test-dir build --output-on-failure
```

Verified on 2026-06-07:

```bash
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/pytest tests/test_*golden*.py tests/test_*verify*.py
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Notes

- This is now the blocking validation task for M4/M5 because quantized kernels
  need explicit rounding and saturation policy before implementation.
- Keep committed golden data small; larger randomized cases should be generated
  during CI or nightly runs.

## Log

### 2026-06-07
- Created as the P0 foundation for scalable golden-reference verification.
- Implemented descriptor/manifest helpers, deterministic f32 ReLU and matmul
  golden generation, comparator modes, pytest coverage, CTest wiring, committed
  smoke bundles, and testing-guide documentation.
- Verification passed with ruff, pytest, build, and full CTest.
