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
make new-design ID=001 TITLE="Feature Name"
```

This creates `docs/design/DES-001-feature-name.md` from the template.

## Naming Convention

```
DES-{ID}-{short-title}.md
```

Examples:
- `DES-001-matmul-operation.md`
- `DES-002-tiling-pass.md`
- `DES-003-rvv-lowering.md`

## Design Documents Index

| ID | Title | Status | Scope | File |
|----|-------|--------|-------|------|
| [DES-006](DES-006-kernel-library-architecture.md) | **Kernel Library Architecture** | Draft | C API, memory mgmt, tiling, packing, target profiles | `DES-006-kernel-library-architecture.md` |
| [DES-009](DES-009-m4-rvv-lowering.md) | **M4 RVV Lowering** | Draft | RVV lowering pipeline and vector preparation | `DES-009-m4-rvv-lowering.md` |
| [DES-010](DES-010-pack-workspace-materialization.md) | **Pack Workspace Materialization** | Implemented | Explicit workspace-backed B packing | `DES-010-pack-workspace-materialization.md` |
| [DES-011](DES-011-riscv-first-transformer-demo.md) | **RISC-V First Transformer Demo Strategy** | Superseded in part | Original custom-runner-first strategy; kernel boundaries retained | `DES-011-riscv-first-transformer-demo.md` |
| [DES-012](DES-012-riscv-simulation-verification.md) | **RISC-V Simulation Verification** | Accepted | QEMU user-mode first; system-mode and gem5 follow-ons | `DES-012-riscv-simulation-verification.md` |
| [DES-014](DES-014-int8-dot-and-gemv-lowering.md) | **INT8 Dot and GEMV Lowering** | Draft | Quantized lowering and generated-object integration | `DES-014-int8-dot-and-gemv-lowering.md` |
| [DES-015](DES-015-quantized-layout-abi-and-integration-conversion-paths.md) | **Quantized Layout ABI and Integration Conversion Paths** | Draft | Proposed native layout versioning and GGML requantization | `DES-015-quantized-layout-abi-and-integration-conversion-paths.md` |
| [DES-016](DES-016-known-runtime-first-transformer-integration.md) | **Known-Runtime-First Transformer Integration** | Under Review | llama.cpp smoke, quantized RVV path, then system simulation | `DES-016-known-runtime-first-transformer-integration.md` |
| [DES-001](DES-001-vector-operations-lowering.md) | Vector Operations Lowering to RISC-V RVV | Draft | RVV intrinsic mapping | `DES-001-vector-operations-lowering.md` |
| [DES-002](DES-002-matmul-kernel.md) | MatMul Kernel (TDD Example) | Draft | MLIR pipeline (partially superseded by DES-006) | `DES-002-matmul-kernel.md` |
| [DES-003](DES-003-conv2d-kernel.md) | Conv2D Kernel Implementation | Draft | Conv2D lowering | `DES-003-conv2d-kernel.md` |
| [DES-004](DES-004-attention-kernel.md) | Attention Kernel Implementation | Draft | Attention lowering | `DES-004-attention-kernel.md` |
| ~~DES-005~~ | ~~Library Packaging (v1)~~ | Deleted | Superseded by DES-006, removed from repo | — |

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

## TDD Development Approach

Each design document includes a complete TDD cycle:

```
RED Phase: Write comprehensive tests
  ├─ Lit tests (MLIR parsing, lowering, verification)
  ├─ Unit tests (C++ functionality)
  ├─ Functional tests (correctness validation)
  └─ Tests fail ✗

GREEN Phase: Implement to pass tests
  ├─ Create operation definitions
  ├─ Implement lowering passes
  └─ Tests pass ✓

REFACTOR Phase: Optimize without breaking tests
  ├─ Profile performance
  ├─ Optimize hot paths
  └─ Tests still pass ✓

VALIDATION Phase: Multi-VLEN testing
  ├─ Test on QEMU (VLEN=128, 256, 512)
  ├─ Functional correctness verification
  ├─ Performance measurement
  └─ All tests pass ✓
```

## Review Process

See `.agents/workflows/design-review.md` for the full review process.

Quick summary:
1. Create design doc (Author)
2. Self-review against checklist
3. Mark as "Under Review"
4. Critical review (Reviewer)
5. Address feedback
6. Get approval
7. Implement following TDD approach
8. Update design doc with implementation learnings
9. Mark as "Implemented"
