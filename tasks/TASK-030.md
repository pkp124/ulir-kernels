# TASK-030: Integrate W4A8 RVV Objects Behind the C API

## Status
[ ] Not Started

## Priority
P0

## Milestone
M8 — Quantized RVV Runtime Integration

## Owner Agent
General or `.cursor/agents/mlir-pass-agent.md`.

## Description

Add a generated or target-specific RVV object path for W4A8 dot/GEMV that
replaces the scalar implementation behind the existing public symbols.

## Acceptance Criteria

- [ ] Preserve `ks_dot_w4a8` and `ks_matvec_w4a8` public signatures.
- [ ] Mirror the explicit object-selection pattern used by INT8 integration.
- [ ] Keep scalar reference implementations available.
- [ ] Validate VLEN 256 and 512 under QEMU user-mode.
- [ ] Compare outputs against existing NumPy/golden references.
- [ ] Confirm the selected implementation contains and executes RVV
      instructions.
- [ ] Document tail, alignment, group-size, and workspace behavior.

## Dependencies

- `TASK-028`
- `TASK-024`

## Verification

- Run W4A8 C API, golden, and QEMU tests.
- Run `ctest --test-dir build --output-on-failure`.

## Log
