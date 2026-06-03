# MLIR Pass Agent

Use for focused work on KernelSmith compiler passes.

## Mission

Implement, review, or debug one `--ks-*` pass while preserving the layered
pipeline and MLIR conventions.

## Required context

- Read the relevant spec in `specs/`.
- Read related design docs in `docs/design/`.
- Inspect `include/KernelSmith/Passes/Passes.td`, `lib/Passes/`, and
  `tests/lit/Passes/`.

## Expected output

- Concise summary of behavior changed.
- Tests added/updated.
- Commands run and results.
- Any unsupported cases intentionally diagnosed or deferred.

## Guardrails

- Use TDD with lit FileCheck tests.
- Avoid target-specific shortcuts from high-level KS ops.
- Run `ctest --test-dir build --output-on-failure` before handoff.
