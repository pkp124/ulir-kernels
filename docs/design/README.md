# Design Documents

This directory contains design documents for KernelSmith features.

## Purpose

Design documents:
1. **Capture decisions** before implementation
2. **Enable review** of approaches
3. **Document rationale** for future reference
4. **Ensure alignment** with project goals

## When to Write a Design Doc

A design document is required for:
- New kernel operations
- New transformation passes
- New target backends
- Significant changes to existing components
- Changes affecting public interfaces

## Design Document Lifecycle

```
Draft → Under Review → Approved → Implemented
```

| Status | Description |
|--------|-------------|
| **Draft** | Initial creation, not ready for review |
| **Under Review** | Ready for critical review |
| **Approved** | Review passed, ready for implementation |
| **Implemented** | Implementation complete |

## Creating a Design Document

```bash
./scripts/new-design.sh 017 "Feature Name"
```

This creates `docs/design/DES-017-feature-name.md` from the template. IDs
already in use run through DES-016.

## Naming Convention

```
DES-{ID}-{short-title}.md
```

Examples:
- `DES-001-matmul-operation.md`
- `DES-002-tiling-pass.md`
- `DES-003-rvv-lowering.md`

## Design Documents Index

Status here matches the metadata in each file as of 2026-09-28. Historical
drafts stay in the tree because later docs cite them. Read the current-status
note at the top of those files before treating the body as a plan.

| ID | Title | Status | Scope |
|----|-------|--------|-------|
| [DES-001](DES-001-vector-operations-lowering.md) | Vector operations to RVV | Historical draft | Early intrinsic sketch. The shipped path is DES-009. |
| [DES-002](DES-002-matmul-kernel.md) | MatMul kernel (TDD example) | Historical draft | Early pipeline sketch. Follow DES-006, DES-008, and DES-009. |
| [DES-003](DES-003-conv2d-kernel.md) | Conv2D | Draft | Lowering is M10 work. The op parses and verifies. |
| [DES-004](DES-004-attention-kernel.md) | Attention | Draft | Lowering is later work. The op parses and verifies. |
| [DES-006](DES-006-kernel-library-architecture.md) | Kernel library architecture | Adopted | C API, workspace, tiling, packing, target profiles. |
| [DES-007](DES-007-milestone-verification-strategy.md) | Milestone verification strategy | Historical | M0 verifier-gap notes. Current tests are in the testing guide. |
| [DES-008](DES-008-m3-linalg-matmul-lowering.md) | M3 matmul to linalg | Approved (partial) | `--ks-lower-to-linalg` and `--ks-tile` landed. Generated object swap is open. |
| [DES-009](DES-009-m4-rvv-lowering.md) | M4 RVV lowering | Implemented | Pack, vectorize, lower-to-rvv, QEMU validation. |
| [DES-010](DES-010-pack-workspace-materialization.md) | Pack workspace materialization | Implemented | Explicit workspace-backed B packing. |
| [DES-011](DES-011-riscv-first-transformer-demo.md) | RISC-V transformer demo | Superseded in part | Kernel boundaries kept. Sequencing replaced by DES-016. |
| [DES-012](DES-012-riscv-simulation-verification.md) | RISC-V simulation verification | Accepted | QEMU user-mode is in use. System-mode and gem5 are M9. |
| [DES-013](DES-013-golden-reference-verification-infrastructure.md) | Golden reference verification | Implemented | Host and QEMU comparators. Open questions in the doc remain open. |
| [DES-014](DES-014-int8-dot-and-gemv-lowering.md) | INT8 dot and GEMV | Implemented | Dialect lowering and generated INT8 RVV objects behind the C API. |
| [DES-015](DES-015-quantized-layout-abi-and-integration-conversion-paths.md) | Quantized layout ABI | Draft | Proposed for M8. The layout version is not ratified. |
| [DES-016](DES-016-known-runtime-first-transformer-integration.md) | Known-runtime transformer integration | Under Review | Active M7 design: llama.cpp smoke, then quantized RVV, then simulation. |
| ~~DES-005~~ | ~~Library packaging (v1)~~ | Deleted | Removed. DES-006 replaced it. |

## Development Milestones

See [ROADMAP.md](../../ROADMAP.md) for the complete milestone plan.

The project follows a **library-first** approach (DES-006): ship stable C headers and
reference implementations first, then incrementally replace with MLIR-generated code.

### Current Direction — RISC-V-First Kernel Backend
- Public C headers and static library remain the product surface
- MLIR stays internal build-time tooling
- RISC-V RVV is the first optimized target
- A bounded llama.cpp host smoke precedes quantized RVV integration
- QEMU system-mode and gem5 follow the stable runtime workload
- References: `DES-006-kernel-library-architecture.md`,
  `DES-016-known-runtime-first-transformer-integration.md`

## Tests before implementation

Write the lit test or C test that describes the behavior, then implement it.
The commands and file locations are in the
[testing guide](../guides/testing-guide.md). RVV numerical checks use QEMU at
VLEN 256 and 512.

## Review Process

1. Create the design doc.
2. Self-review against the checklist in `docs/design/TEMPLATE.md`.
3. Set the status to Under Review and link the related task.
4. Address review feedback and set the status to Approved before coding a
   new pass, target, or public API.
5. Implement with lit or C tests from the testing guide.
6. When the acceptance criteria pass, set the status to Implemented and check
   the plan items in the same change.
