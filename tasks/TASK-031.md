# TASK-031: Route llama.cpp Quantized Decode Through KernelSmith

## Status
[ ] Not Started

## Priority
P0

## Milestone
M8 — Quantized RVV Runtime Integration

## Owner Agent
General.

## Description

Use the approved conversion path and W4A8 implementation to route a pinned
llama.cpp decode dot/GEMV operation through KernelSmith.

## Acceptance Criteria

- [ ] Convert the selected runtime weights through `TASK-029`.
- [ ] Invoke `ks_dot_w4a8` or `ks_matvec_w4a8` on a reviewed decode path.
- [ ] Report conversion metadata and a non-zero quantized KS invocation count.
- [ ] Compare operation outputs, deterministic tokens, and model-level error
      against the pinned runtime baseline.
- [ ] Keep unsupported prefill/GEMM paths on the runtime baseline.
- [ ] Document the integration as a conversion adapter, not native GGML
      compatibility.

## Dependencies

- `TASK-029`
- `TASK-030`

## Verification

- Run host baseline and KS-enabled deterministic inference.
- Run operation-level numerical comparisons and model-level token checks.
- Run `ctest --test-dir build --output-on-failure`.

## Log
