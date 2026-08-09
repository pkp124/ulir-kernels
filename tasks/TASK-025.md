# TASK-025: Adopt Known-Runtime-First Transformer Strategy

## Status
[?] Needs Review

## Priority
P0

## Milestone
M7 — Known-Runtime Transformer Integration Smoke

## Owner Agent
General.

## Description

Revise the post-M6 roadmap so the first transformer demo uses a recognized
runtime with a narrowly scoped KernelSmith integration. Separate integration
correctness, quantized RVV acceleration, and system simulation into sequential
milestones with explicit claim boundaries.

## Acceptance Criteria

- [x] Add a design document that records the runtime-first decision and
      supersedes the custom-runner-first sequencing in `DES-011`.
- [x] Define a bounded feasibility gate before selecting a permanent llama.cpp
      backend or internal hook.
- [x] Keep the first runtime smoke independent of unresolved GGML-to-KernelSmith
      quantization conversion.
- [x] Move quantized decode integration and configurable system simulation into
      later milestones.
- [x] Create PR-sized child tasks with dependencies and verification criteria.
- [x] Reconcile stale roadmap and design-document indexes.
- [x] Run full CTest and Python lint/format checks.

## Dependencies

- `TASK-024`
- `docs/design/DES-011-riscv-first-transformer-demo.md`
- `docs/design/DES-012-riscv-simulation-verification.md`
- `docs/design/DES-015-quantized-layout-abi-and-integration-conversion-paths.md`

## Verification

```bash
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

## Notes

- The sequencing decision is high confidence. The exact llama.cpp seam and
  CI-sized model remain feasibility-spike outputs, not assumptions.
- The first smoke may use an f32 KernelSmith API and must not claim quantized or
  RVV acceleration.

## Log

### 2026-08-09
- Started after M6 completion and review of the custom-runner-first roadmap.
- Added DES-016 and split the plan into host runtime smoke, quantized RVV
  integration, and system simulation milestones.
- Added `TASK-026` through `TASK-034` with explicit dependencies and claim
  boundaries.
- Verified all 21 CTest tests pass; ruff lint and format checks pass.
