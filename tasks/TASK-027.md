# TASK-027: Route a Host Runtime Operation Through KernelSmith

## Status
[ ] Not Started

## Priority
P0

## Milestone
M7 — Known-Runtime Transformer Integration Smoke

## Owner Agent
General.

## Description

Add a build-time opt-in that routes one semantically matching f32 llama.cpp
operation through the existing KernelSmith public C API.

## Acceptance Criteria

- [ ] Keep the unmodified runtime baseline available for comparison.
- [ ] Route exactly one reviewed operation through `libkernelsmith`.
- [ ] Report a non-zero KernelSmith invocation count.
- [ ] Compare adapter and baseline operation outputs within a documented policy.
- [ ] Preserve deterministic greedy tokens from `TASK-026`.
- [ ] Document whether the seam should become a backend or remain a CPU adapter.
- [ ] Make no quantized or RVV acceleration claim.

## Dependencies

- `TASK-026`

## Verification

- Build baseline and KS-enabled host configurations.
- Run operation-level comparison and deterministic token validation.
- Run `ctest --test-dir build --output-on-failure`.

## Notes

- Return to design review if the seam requires broad GGML buffer, scheduler, or
  graph ownership.

## Log
