# TASK-009: Plan RISC-V Simulation Verification

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`general`

## Description

Create the design and execution plan for incorporating RISC-V simulation
platforms into KernelSmith verification. This task scopes how QEMU, Spike,
QEMU system-mode, gem5, and Renode should be introduced without delaying the
current M4 generated-kernel correctness work.

## Acceptance Criteria

- [x] Add a design document for RISC-V simulation verification.
- [x] Define simulator platform tiers and their CI policy.
- [x] Document why QEMU user-mode is the first required verification platform.
- [x] Document how Spike, gem5, Renode, and QEMU system-mode fit into later
      phases.
- [x] Record follow-up implementation phases for `TASK-006`.
- [x] Update task indexes and the milestone dashboard.

## Dependencies

- `TASK-006`
- `docs/design/DES-009-m4-rvv-lowering.md`
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `specs/targets/riscv-rvv.md`
- `tests/qemu_runner.py`
- `scripts/setup-rvv-sim.sh`

## Verification

```bash
git status --short --branch
ctest --test-dir build --output-on-failure
```

Verified on 2026-06-09:

```bash
ctest --test-dir build --output-on-failure
```

## Notes

- This is a planning task. Implementation work for the first simulator runner
  changes should be tracked under `TASK-006` or a follow-up child task.
- The design uses `https://github.com/pkp124/riscv` as a reference for platform
  taxonomy and CI patterns, not as code to vendor into KernelSmith.

## Log

### 2026-06-06
- Added `DES-012` with the simulation verification architecture.
- Marked this task as needs review because the planning artifact is ready for
  design review before runner implementation begins.

### 2026-06-09
- Reviewed `DES-012` after `TASK-006`, `TASK-010`, `TASK-011`, and `TASK-012`
  landed the QEMU user-mode, golden-reference, and benchmark-reporting paths.
- Accepted the QEMU-first simulator tiering as the M4 verification policy.
- Left Spike, QEMU system-mode, gem5, and Renode as future follow-up tasks only
  when their validation scope becomes concrete.
