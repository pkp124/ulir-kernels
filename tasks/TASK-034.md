# TASK-034: Map the Runtime Workload to gem5 Full-System

## Status
[ ] Not Started

## Priority
P1

## Milestone
M9 — Configurable RISC-V System Simulation

## Owner Agent
General.

## Description

Map the validated single-node system configuration and runtime workload from
QEMU system-mode to gem5 full-system for cache and microarchitectural analysis.

## Acceptance Criteria

- [ ] Pin a gem5 revision with sufficient RVV support for the workload.
- [ ] Translate the common system configuration through a gem5 adapter.
- [ ] Boot the same pinned image and execute the same deterministic workload.
- [ ] Compare functional output with QEMU system-mode.
- [ ] Export cache, memory, instruction, and cycle statistics in a stable report.
- [ ] Document unsupported or non-equivalent configuration fields.
- [ ] Keep gem5 out of required pull-request checks.

## Dependencies

- `TASK-033`

## Verification

- Rebuild gem5 and the system image from pinned inputs.
- Run the deterministic workload and compare outputs with QEMU.
- Validate the machine-readable statistics report.

## Notes

- Additional harts, accelerators, and heterogeneous nodes are follow-up tasks.

## Log
