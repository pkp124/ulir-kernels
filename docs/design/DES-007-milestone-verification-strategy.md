# DES-007: Milestone Verification Strategy

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-10 |
| **Priority** | Critical |

> **Current status (2026-09-28):** This file records the early verifier gap.
> The dialect now has 20 operations. Verifiers cover them except relu, gelu,
> and silu. Run tests with the [testing guide](../guides/testing-guide.md).

## Context

### Problem Statement

KernelSmith has 13 dialect operations but only 5 have verifiers, and those verifiers
are incomplete. As the project moves through milestones (M0 cleanup → M1 C API → M2+
lowering passes), each layer depends on the correctness of the layer below it. Without a
systematic verification strategy, regressions will compound silently.

### Background

Milestone 0 established the dialect infrastructure with parse/print round-trip lit tests
and verifier negative tests for matmul, batch_matmul, conv2d, attention, and layer_norm.
The remaining 8 ops lack verifiers entirely (softmax, rms_norm, reduce_sum, reduce_max)
or rely solely on TableGen traits (relu, gelu, silu via `SameOperandsAndResultType`).

DES-006 defines the C library architecture for Milestone 1, requiring functional
correctness validation against reference implementations.

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Every op with semantic constraints has a verifier | Must Have |
| REQ-2 | Every verifier error path has a negative lit test | Must Have |
| REQ-3 | Container-based reproducible test environment | Must Have |
| REQ-4 | Functional correctness tests against numpy (M1+) | Must Have |
| REQ-5 | Lowering semantic-equivalence tests (M2+) | Should Have |
| REQ-6 | Target-specific QEMU validation (M5+) | Should Have |
| REQ-7 | Performance regression tracking (M4+) | Nice to Have |

## Design

### Overview

The verification strategy is organized as a testing pyramid with four layers,
each activated at the appropriate milestone:

```
                  ┌───────────┐
                  │   QEMU    │  M5+ (target-specific, slow)
                 ─┤  + perf   ├─
                / └───────────┘ \
               /  ┌───────────┐  \
              │   │  Python   │   │  M1+ (functional_validator vs numpy)
              │  ─┤   func    ├─  │
              │ / └───────────┘ \ │
              ││  ┌───────────┐ ││
              ││  │  C API    │ ││  M1 (link + call + check)
              ││ ─┤  smoke    ├─││
              │││/└───────────┘\│││
              ││││┌───────────┐││││
              │││││ Lit tests  │││││  M0+ (FileCheck, verifiers)
              └┴┴┴┴───────────┴┴┴┴┘
              ┌───────────────────┐
              │  Unit tests (C++) │  Always (dialect loading, types)
              └───────────────────┘
```

### Layer 1: Dialect Verifiers + Lit Tests (M0 — immediate)

Complete all missing verifiers and add negative tests for every error path.

**Ops needing new verifiers:**

| Op | Validations |
|----|------------|
| `ks.softmax` | Input is ranked tensor; axis in `[-rank, rank)` |
| `ks.rms_norm` | Input and weight are ranked tensors; input rank >= 1 |
| `ks.reduce_sum` | Input is ranked tensor; each axis in `[-rank, rank)`; no duplicate axes |
| `ks.reduce_max` | Same as reduce_sum |

**Ops needing strengthened verifiers:**

| Op | Additional Validations |
|----|----------------------|
| `ks.batch_matmul` | Result is ranked tensor; batch dimensions match; inner dimensions match; element types match |
| `ks.attention` | Query/key head dimension match; key seq == value seq; element types match |

**Ops already covered by traits (no custom verifier needed):**

| Op | Trait |
|----|-------|
| `ks.relu` | `SameOperandsAndResultType` |
| `ks.gelu` | `SameOperandsAndResultType` |
| `ks.silu` | `SameOperandsAndResultType` |

**Lit test coverage targets:**

| Category | Current | Target |
|----------|---------|--------|
| Parse/print round-trip | 6 tests | 6 tests (sufficient) |
| Verifier negative tests | 12 cases / 5 ops | ~28 cases / 9 ops |

### Layer 2: C API Smoke Tests (M1)

When `libkernelsmith.a` ships, add C test programs that:
- Link against the library and call each public function
- Verify return codes and workspace query results
- Confirm no MLIR symbols are leaked (nm check)

### Layer 3: Functional Correctness (M1+)

Wire up the existing `functional_validator.py` to compare C reference output against
numpy across the shape matrix defined in `specs/kernels/matmul.md`:
- Square: 64x64
- Rectangular: 32x128 x 128x64
- Small: 4x4
- Non-power-of-2: 17x23
- Edge: single-element, NaN/Inf propagation

Tolerance: bitwise-identical for integer ops, configurable ULP for float ops.

### Layer 4: Lowering Semantic Equivalence (M2+)

Each lowering pass gets three verification layers:
1. **Structural**: FileCheck that input ops are replaced by target ops
2. **Semantic**: Compile + execute lowered IR, compare against M1 reference output
3. **Idempotency**: Running the pass twice produces identical output

### Layer 5: Target Validation (M5+)

- QEMU for RISC-V RVV (using existing `qemu_runner.py`)
- Native execution for x86 AVX2/AVX-512
- Cross-target equivalence: same input → same output (within tolerance)

### Container Infrastructure

A Docker-based reproducible environment ensures all contributors and CI systems
run tests against identical dependencies.

**Container contents:**
- Ubuntu 22.04 base
- LLVM 18 + MLIR from apt.llvm.org
- CMake 3.20+, Ninja, Google Test
- Python 3.10+ with dev dependencies
- Multi-stage build: `dev` (full toolchain) and `test` (build + run tests)

**Usage:**
```bash
docker compose run test          # Build and run all tests
docker compose run dev           # Interactive shell with full toolchain
docker compose run lint          # Run ruff + clang-format checks
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Nix-based environment | Exact reproducibility, declarative | Steep learning curve, slow first build | Not chosen — team familiarity with Docker |
| GitHub Actions only | No local setup needed | Slow feedback loop, can't reproduce locally | Complement, not replace containers |
| Docker + compose | Familiar, fast iteration, works locally and in CI | Image size (~2GB with LLVM) | **Selected** |

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| LLVM 18 apt repo becomes unavailable | High | Low | Pin specific package versions in Dockerfile |
| Docker image too large for CI | Medium | Medium | Multi-stage build, layer caching |
| Verifier changes break existing tests | Medium | Low | Run full test suite before committing |

## Implementation Plan

### Phase 1: M0 Verification Closure (this PR)
- [x] Create container infrastructure (Dockerfile, docker-compose.yml, .dockerignore)
- [ ] Add `hasVerifier = 1` to softmax, rms_norm, reduce_sum, reduce_max in TableGen
- [ ] Implement verifiers for softmax, rms_norm, reduce_sum, reduce_max
- [ ] Strengthen batch_matmul verifier (result shape, element types, inner dims)
- [ ] Strengthen attention verifier (dimension cross-checks, element types)
- [ ] Add negative lit tests for all new/updated verifiers
- [ ] Build and test in container

### Phase 2: M1 Functional Tests (future PR)
- [ ] Wire up functional_validator.py with matmul shape matrix
- [ ] Add C API smoke test harness
- [ ] Integrate container tests into CI

### Phase 3: M2+ Lowering Tests (future PR)
- [ ] Structural FileCheck tests for each lowering pass
- [ ] Semantic equivalence test framework
- [ ] QEMU integration tests
