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
| [DES-001](DES-001-vector-operations-lowering.md) | Vector Operations Lowering to RISC-V RVV | Draft | RVV intrinsic mapping | `DES-001-vector-operations-lowering.md` |
| [DES-002](DES-002-matmul-kernel.md) | MatMul Kernel (TDD Example) | Draft | MLIR pipeline (partially superseded by DES-006) | `DES-002-matmul-kernel.md` |
| [DES-003](DES-003-conv2d-kernel.md) | Conv2D Kernel Implementation | Draft | Conv2D lowering | `DES-003-conv2d-kernel.md` |
| [DES-004](DES-004-attention-kernel.md) | Attention Kernel Implementation | Draft | Attention lowering | `DES-004-attention-kernel.md` |
| ~~DES-005~~ | ~~Library Packaging (v1)~~ | Deleted | Superseded by DES-006, removed from repo | — |

## Development Milestones

See [ROADMAP.md](../../ROADMAP.md) for the complete milestone plan.

The project follows a **library-first** approach (DES-006): ship stable C headers and
reference implementations first, then incrementally replace with MLIR-generated code.

### Current: Milestone 1 — C API + Reference Library
- Public C headers (ks_matmul.h, ks_activations.h, ks_common.h)
- Target profile system (target/generic.h)
- Handwritten reference implementations
- Reference: `DES-006-kernel-library-architecture.md`

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
