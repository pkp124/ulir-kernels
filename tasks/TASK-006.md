# TASK-006: Validate RVV Correctness and Benchmark Path

## Status
[~] In Progress

## Priority
P1 (High)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`.cursor/agents/rvv-validation-agent.md`

## Description

Close the remaining RISC-V RVV milestone validation gaps after the pack,
vectorize, and lower-to-RVV compiler pipeline is implemented.

## Acceptance Criteria

- [ ] Generated RVV matmul binary runs under QEMU for at least one supported VLEN.
- [ ] QEMU correctness compares generated output against the C or NumPy reference.
- [ ] Benchmark reports generic/reference vs RVV path results in a reproducible format.
- [ ] Non-divisible `N` or tail behavior is either supported with tests or rejected
      with clear diagnostics.
- [ ] Documentation records required local tooling when QEMU/cross-compiler is not
      available by default.

## Dependencies

- TASK-004: Tiling and vectorization
- TASK-005: RVV lowering pipeline
- `specs/targets/riscv-rvv.md`
- `docs/design/DES-009-m4-rvv-lowering.md`

## Verification

```bash
ctest --test-dir build --output-on-failure
./scripts/compile-rvv.sh <input.mlir> <output.o>
python tests/qemu_runner.py --help
```

Run QEMU execution tests when `qemu-riscv64` and a RISC-V cross toolchain are
available. If unavailable, record the missing tools in the task log and PR body.

## Notes

- Keep this task focused on validation and measurement, not new quantized
  lowering work.
- If tail handling requires new compiler behavior, split that implementation
  into a separate task and link it here.

## Log

### 2026-06-06
- Created from the remaining open M4 roadmap items: QEMU correctness,
  benchmarking, and tail/stride hardening.
