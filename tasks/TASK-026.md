# TASK-026: Pin llama.cpp Runtime and Model Baseline

## Status
[ ] Not Started

## Priority
P0

## Milestone
M7 — Known-Runtime Transformer Integration Smoke

## Owner Agent
General.

## Description

Establish a reproducible, unmodified llama.cpp baseline and identify the
narrowest candidate operation seam for a KernelSmith host integration.

## Acceptance Criteria

- [ ] Pin a llama.cpp commit and record its source URL and license.
- [ ] Select a licensed, CI-sized model; record its URL, checksum, architecture,
      and artifact policy.
- [ ] Add a reproducible host build/run command for deterministic greedy output.
- [ ] Record baseline tokens and relevant numerical output.
- [ ] Map candidate runtime operations to existing KernelSmith public C APIs.
- [ ] Recommend one seam, with stop criteria if no semantic match exists.
- [ ] Do not modify runtime execution to call KernelSmith in this task.

## Dependencies

- `TASK-025`
- `docs/design/DES-016-known-runtime-first-transformer-integration.md`

## Verification

- Rebuild the pinned runtime from a clean source checkout.
- Verify the model checksum.
- Run the deterministic baseline twice and compare tokens.

## Notes

- A moving llama.cpp branch or an unlicensed model is not acceptable.
- Model size must be practical for host CI and later RISC-V QEMU validation.

## Log
