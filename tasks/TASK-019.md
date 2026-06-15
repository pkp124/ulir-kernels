# TASK-019: Add Elementwise Add/Mul Ops and Linalg Lowering

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M6 — Transformer Minimum Kernel Set

## Owner Agent
General or `.cursor/agents/mlir-pass-agent.md`.

## Description

Add the first transformer helper compiler slice: `ks.add` and `ks.mul`
elementwise float tensor ops with verifier coverage, broadcasting semantics, and
lowering to vectorizable `linalg.generic` forms.

## Acceptance Criteria

- [x] Define `ks.add` and `ks.mul` ops in TableGen.
- [x] Verify operands/results are ranked floating-point tensors with matching
      element types.
- [x] Support NumPy-style static broadcasting for leading dimensions and
      dimensions equal to `1`, while allowing dynamic dimensions when they do
      not contradict static shapes.
- [x] Add parse/print lit tests.
- [x] Add invalid verifier diagnostic lit tests.
- [x] Lower add and mul to `linalg.generic` with `arith.addf`/`arith.mulf`.
- [x] Run CTest and lint before completion.

## Dependencies

- `TASK-008`
- `docs/design/DES-011-riscv-first-transformer-demo.md`

## Verification

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

Verified on 2026-06-15:

```bash
cmake --build build --parallel
cmake --build build --target check-kernelsmith-lit -- -v
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

## Notes

- This task intentionally stops before public C APIs and QEMU helper-kernel
  validation; those should land as follow-up child tasks.

## Log

### 2026-06-15
- Created as the first PR-sized child task for `TASK-008`.
- Added `ks.add`/`ks.mul` TableGen definitions, verifier coverage for ranked
  float tensors and static broadcasting, and `--ks-lower-to-linalg`
  `linalg.generic` lowering.
- Added parse/print, verifier diagnostic, and lowering lit tests. Full CTest and
  Python lint/format checks passed.
