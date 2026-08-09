# TASK-029: Convert GGML Q4_0 Weights to Native W4A8

## Status
[ ] Not Started

## Priority
P0

## Milestone
M8 — Quantized RVV Runtime Integration

## Owner Agent
General.

## Description

Implement an offline or load-time conversion that decodes GGML `Q4_0` blocks
to a numerical reference and requantizes them to the approved KernelSmith W4A8
layout.

## Acceptance Criteria

- [ ] Pin the referenced GGML `Q4_0` decoding behavior to a source commit.
- [ ] Test positive, negative, and zero GGML block scales.
- [ ] Requantize to positive native scales without direct nibble reuse.
- [ ] Validate packing, group boundaries, odd lengths, and row strides.
- [ ] Report reconstruction error against the decoded f32 reference.
- [ ] Reject unsupported GGML formats and layout versions clearly.

## Dependencies

- `TASK-028`

## Verification

- Run converter unit and numerical reference tests.
- Run `ruff check .` and `ruff format --check .`.
- Run `ctest --test-dir build --output-on-failure`.

## Notes

- Conversion is lossy and must not be described as GGML format compatibility.

## Log
