# TASK-032: Validate the Quantized Runtime Under QEMU User-Mode

## Status
[ ] Not Started

## Priority
P0

## Milestone
M8 — Quantized RVV Runtime Integration

## Owner Agent
General or `.cursor/agents/rvv-validation-agent.md`.

## Description

Cross-build the pinned llama.cpp integration and run its deterministic
KernelSmith quantized decode path under RISC-V QEMU user-mode.

## Acceptance Criteria

- [ ] Produce a reproducible static or self-contained RISC-V Linux build.
- [ ] Run the pinned model workload at VLEN 256 and 512.
- [ ] Match the approved host token/error policy.
- [ ] Prove the W4A8 RVV path executes through counters and instruction
      evidence.
- [ ] Upload machine-readable correctness and benchmark reports.
- [ ] Label QEMU timing as simulation data rather than hardware performance.

## Dependencies

- `TASK-031`
- `docs/design/DES-012-riscv-simulation-verification.md`

## Verification

- Run the cross-build script from a clean tree.
- Run QEMU at VLEN 256 and 512.
- Run `ctest --test-dir build --output-on-failure`.

## Log
