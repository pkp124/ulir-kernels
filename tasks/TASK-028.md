# TASK-028: Ratify and Version the Native Quantized Layout

## Status
[ ] Not Started

## Priority
P0

## Milestone
M8 — Quantized RVV Runtime Integration

## Owner Agent
General.

## Description

Complete design review of `DES-015`, settle the native W4A8 contract needed by
runtime conversion, and add explicit layout versioning with ABI tests.

## Acceptance Criteria

- [ ] Approve or revise the native signed-INT4 packing and scale contract.
- [ ] Resolve the initial group-size recommendation.
- [ ] Add `KS_QUANT_LAYOUT_VERSION` only after design approval.
- [ ] Add tests for nibble order, signed decoding, scale indexing, strides, and
      unsupported versions.
- [ ] Update public quantization specifications and API documentation.
- [ ] Preserve existing M5 golden cases.

## Dependencies

- `TASK-027`
- `docs/design/DES-015-quantized-layout-abi-and-integration-conversion-paths.md`

## Verification

- Run quantized C API and golden-reference tests.
- Run `ctest --test-dir build --output-on-failure`.

## Notes

- This task cannot self-approve `DES-015`; implementation follows recorded
  design approval.

## Log
