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

| ID | Title | Status | Phase | File |
|----|-------|--------|-------|------|
| [DES-001](DES-001-vector-operations-lowering.md) | Vector Operations Lowering to RISC-V RVV | Draft | Phase 2 | `DES-001-vector-operations-lowering.md` |
| [DES-002](DES-002-matmul-kernel.md) | MatMul Kernel Implementation (TDD Example) | Draft | Phase 3 | `DES-002-matmul-kernel.md` |
| [DES-003](DES-003-conv2d-kernel.md) | Conv2D Kernel Implementation | Draft | Phase 4 | `DES-003-conv2d-kernel.md` |
| [DES-004](DES-004-attention-kernel.md) | Attention Kernel Implementation | Draft | Phase 5 | `DES-004-attention-kernel.md` |

## Development Phases

The KernelSmith project is organized into phases with corresponding design documents:

### Phase 1: Test Infrastructure ✅ (Complete)
- Test framework setup
- QEMU multi-VLEN configuration
- Test data generators
- Functional validation harness
- Reference: `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md`

### Phase 2: Vector Operations Foundation 📋 (In Planning)
- Lowering vector operations to RISC-V RVV
- Type conversion system
- Vsetvl optimization
- Reference: `DES-001-vector-operations-lowering.md`

### Phase 3: MatMul Kernel 📋 (In Planning)
- Complete example of TDD development
- Full lowering pipeline
- Reference implementation
- Reference: `DES-002-matmul-kernel.md`

### Phase 4: Conv2D Kernel 📋 (In Planning)
- 2D convolution support
- Memory layout optimization
- Reference: `DES-003-conv2d-kernel.md`

### Phase 5: Attention Kernel 📋 (In Planning)
- Scaled dot-product attention
- Transformer support
- Reference: `DES-004-attention-kernel.md`

### Phase 6+: Extended Kernels & Optimization 📋 (Planned)
- Additional operations (activations, normalizations, reductions)
- Performance optimization
- IREE integration

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
