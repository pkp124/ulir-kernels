# TASK-033: Bring Up a Configurable QEMU RVV System Node

## Status
[ ] Not Started

## Priority
P1

## Milestone
M9 — Configurable RISC-V System Simulation

## Owner Agent
General.

## Description

Define a simulator-neutral system configuration and boot one RV64GCV node in
QEMU system-mode with the stable runtime workload from M8.

## Acceptance Criteria

- [ ] Add a design document for the system configuration and simulator adapter
      boundary.
- [ ] Configure one hart, ISA/extensions, VLEN/ELEN, memory, and required
      devices without hardcoding them in the launcher.
- [ ] Distinguish kernel target profiles from system topology configuration.
- [ ] Boot a pinned Linux/initramfs or justified alternative.
- [ ] Run an RVV probe and the stable KernelSmith runtime workload.
- [ ] Report unsupported QEMU properties instead of silently ignoring them.
- [ ] Keep this test scheduled/manual until its cost is understood.

## Dependencies

- `TASK-032`
- `docs/design/DES-012-riscv-simulation-verification.md`
- `https://github.com/pkp124/riscv` as a reference, not vendored code

## Verification

- Rebuild the image and system configuration from pinned inputs.
- Boot QEMU system-mode and validate deterministic workload output.

## Log
