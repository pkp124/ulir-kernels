# TASK-010: Build Golden Reference Verification Infrastructure

## Status
[ ] Not Started

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

- [ ] Add a descriptor schema for functional cases, including kernel name,
      inputs, outputs, target list, and compare policy.
- [ ] Add a manifest format that records shape, dtype, seed, generator backend,
      tolerance, quantization policy fields, and SHA-256 hashes.
- [ ] Add deterministic NumPy golden generators for at least f32 ReLU and f32
      matmul smoke cases.
- [ ] Add comparator support for exact, allclose, quantized exact, and
      dequantized allclose modes.
- [ ] Add focused unit tests for schema validation, hash validation, comparator
      pass/fail behavior, and diagnostic output.
- [ ] Update docs to explain how to add a new golden-reference case.

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

## Notes

- This is now the blocking validation task for M4/M5 because quantized kernels
  need explicit rounding and saturation policy before implementation.
- Keep committed golden data small; larger randomized cases should be generated
  during CI or nightly runs.

## Log

### 2026-06-07
- Created as the P0 foundation for scalable golden-reference verification.
