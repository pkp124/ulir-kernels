# TASK-006: Validate RVV Correctness and Benchmark Path

## Status
[~] In Progress

## Priority
P0 (Milestone Blocking)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`.cursor/agents/rvv-validation-agent.md`

## Description

Close the remaining RISC-V RVV milestone validation gaps after the pack,
vectorize, and lower-to-RVV compiler pipeline is implemented.

## Acceptance Criteria

- [ ] Golden-reference infrastructure exists for deterministic NumPy/framework
      inputs, expected outputs, metadata, and comparison policy.
- [ ] Host/x86 output is verified against the golden reference for at least one
      representative kernel.
- [ ] Generated RVV matmul binary runs under QEMU for at least one supported VLEN.
- [ ] QEMU correctness compares generated RVV output against the NumPy/framework
      golden reference or a golden-validated host output.
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
- `docs/design/DES-012-riscv-simulation-verification.md`
- `docs/design/DES-013-golden-reference-verification-infrastructure.md`
- `TASK-010`: Build golden reference verification infrastructure
- `TASK-011`: Add host reference execution comparator
- `TASK-012`: Compare RISC-V RVV outputs against golden references

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
- Follow DES-012 for simulator platform tiers: QEMU user-mode first, Spike as
  optional reference validation, and gem5/Renode only for later analysis or
  board-level scenarios.
- If tail handling requires new compiler behavior, split that implementation
  into a separate task and link it here.

## Log

### 2026-06-06
- Created from the remaining open M4 roadmap items: QEMU correctness,
  benchmarking, and tail/stride hardening.
- Added the first descriptor-backed RISC-V functional verification path:
  `tests/riscv_runner.py` builds generated RVV objects, links them with a C
  max-error harness, rejects VLENs below the `riscv_rvv_256` baseline, and runs
  QEMU user-mode cases that print `PASS`, `MAX_ABS_ERROR`, and `TIME_NS`.

### 2026-06-07
- Raised priority to P0 because M4/M5 correctness depends on golden-reference
  validation before adding more RVV or fixed-point kernels.
- Split scalable verification work into `TASK-010`, `TASK-011`, and `TASK-012`
  following `DES-013`.
