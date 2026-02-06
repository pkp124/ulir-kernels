# KernelSmith RISC-V RVV Development Summary

## Project Overview

**Goal**: Build a comprehensive library of ML kernels for RISC-V Vector Extension (RVV), optimized for performance and developed with Test-Driven Development (TDD) principles.

**Target**: RISC-V with RVV (Scalable Vector Extension)
**Focus**: MatMul, Conv2D, Attention - then extended to additional operations
**Learning Goal**: Master MLIR for eventual IREE contribution

---

## What's Complete ✅

### 1. Comprehensive Development Plan
**File**: `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md`

- 8-phase development strategy from foundation to IREE integration
- TDD approach (RED-GREEN-REFACTOR) for each kernel
- Test pyramid structure (spec → unit → functional → integration)
- Multi-VLEN testing strategy (128, 256, 512 bits)
- Learning outcomes mapped to MLIR expertise and IREE contribution

**Status**: ✅ Complete, committed, pushed

---

### 2. Phase 1: Test Infrastructure ✅
**Location**: `scripts/`, `tests/`, `docs/guides/testing-guide.md`

#### Test Runner
- `scripts/run-tests.sh` - Unified test orchestration
- Categories: lit (MLIR), unit (C++), integration (full pipeline)
- Multi-VLEN QEMU execution support
- Verbose output and error handling

#### Test Data Generation
- `tests/test_data_generator.py` - Generate test tensors
- MatMul: Small (4×4), Medium (64×64), Large (256×256), Rectangular, Float16
- Conv2D: Multiple configurations
- Attention: Different sequence lengths
- Produces: Binary data + metadata + reference results

#### Functional Validation
- `tests/functional_validator.py` - Compare vs reference implementations
- Supports: MatMul, Conv2D, Attention, Activations
- Error metrics: max, mean, relative error
- JSON validation reports

#### QEMU Multi-VLEN Testing
- `tests/qemu_runner.py` - Execute on QEMU with configurable VLEN
- Standard VLENs: 128, 256, 512 bits
- Consistency validation across different vector lengths
- Benchmarking support

#### Documentation
- `docs/guides/testing-guide.md` - Comprehensive testing patterns
- Lit test examples (parsing, verifier, lowering)
- C++ unit test patterns with Google Test
- TDD workflow documentation
- Debugging and profiling guides

**Status**: ✅ Complete, committed, pushed

---

### 3. Comprehensive Design Documents 📋

#### DES-001: Vector Operations Lowering (Phase 2)
**File**: `docs/design/DES-001-vector-operations-lowering.md`

Foundation for all kernels. Covers:
- Type conversion (MLIR vectors → RISC-V RVV parameters)
- Operation lowering patterns:
  - `vector.load` → `vle{SEW}.v`
  - `vector.store` → `vse{SEW}.v`
  - `vector.fma` → `vfmacc.vv`
  - `vector.reduction<add>` → `vfredusum.vs`
  - `vector.reduction<max>` → `vfredmax.vs`
- Vsetvl optimization guidance
- VLA code generation
- Test strategy: unit, lit, integration, multi-VLEN
- Implementation plan: 5 stages, ~11-12 days

**Status**: 📋 Draft, ready for review

---

#### DES-002: MatMul Kernel Implementation (Phase 3)
**File**: `docs/design/DES-002-matmul-kernel.md`

Reference TDD example. Covers:
- Complete pipeline: `ks.matmul` → `linalg.matmul` → tiled loops → vectorized → RVV
- Operation definition and verifier
- Lowering stages with MLIR examples
- Tiling strategy (M=64, N=64, K=32)
- Full TDD cycle documentation:
  - RED phase: Comprehensive test specifications
  - GREEN phase: Implementation approach
  - REFACTOR phase: Optimization strategy
- Edge cases: Various matrix shapes, data types
- Implementation plan: 4 weeks (RED-GREEN-REFACTOR-Validation)

**Status**: 📋 Draft, ready for implementation planning

---

#### DES-003: Conv2D Kernel Implementation (Phase 4)
**File**: `docs/design/DES-003-conv2d-kernel.md`

Extended kernel. Covers:
- 2D convolution with NHWC format
- Stride, padding, dilation support
- Two implementation strategies:
  - im2col + matmul (uses MatMul kernel)
  - Direct tiled loops (more efficient)
- Memory access pattern optimization
- Test structure and edge cases
- Implementation plan: ~10-12 days

**Status**: 📋 Draft, planned after MatMul

---

#### DES-004: Attention Kernel Implementation (Phase 5)
**File**: `docs/design/DES-004-attention-kernel.md`

Advanced kernel. Covers:
- Scaled Dot-Product Attention (SDPA) decomposition
- Numerically stable softmax
- Masking support (causal, custom)
- Memory bandwidth optimization
- Three-stage computation: Q·K^T → softmax → apply to V
- Test strategy with numerical validation
- Implementation plan: ~12-14 days

**Status**: 📋 Draft, planned after Conv2D

---

#### Design Documentation Index
**File**: `docs/design/README.md`

Updated with:
- Design document index table
- Development phases overview
- TDD approach explanation
- Cross-references between documents

**Status**: ✅ Updated, committed, pushed

---

## Development Phases at a Glance

```
Phase 1 (Week 0-1) ✅ COMPLETE
├─ Test infrastructure
├─ Test runners
├─ Data generation
└─ Functional validation

Phase 2 (Week 2-4) 📋 PLANNED
├─ Vector operations lowering (DES-001)
├─ Type conversion system
├─ Vsetvl optimization
└─ Foundation for all kernels

Phase 3 (Week 4-8) 📋 PLANNED
├─ MatMul kernel (DES-002)
├─ Full TDD cycle example
├─ Performance baseline
└─ Reference implementation

Phase 4 (Week 8-10) 📋 PLANNED
├─ Conv2D kernel (DES-003)
├─ CNN support
└─ Memory optimization

Phase 5 (Week 10-12) 📋 PLANNED
├─ Attention kernel (DES-004)
├─ Transformer support
└─ Advanced numeric techniques

Phase 6+ (Week 12+) 📋 PLANNED
├─ Extended operations
├─ Activations, normalizations, reductions
├─ Performance optimization
└─ IREE integration
```

---

## Key Design Principles

### 1. Test-Driven Development
```
RED: Write failing tests
  ├─ Lit tests (parsing, lowering)
  ├─ Unit tests (functionality)
  └─ Functional tests (correctness)

GREEN: Implement to pass tests
  └─ All tests pass ✓

REFACTOR: Optimize without breaking tests
  └─ All tests still pass ✓

VALIDATE: Multi-VLEN testing
  └─ Correctness verified ✓
```

### 2. Multi-Level Lowering Pipeline
```
Kernel Dialect (ks.matmul)
  ↓ Lower to Linalg
Linalg Operations (linalg.matmul)
  ↓ Add Tiling
SCF Loops (nested for loops)
  ↓ Vectorize
Vector Operations (vector.load, fma, store)
  ↓ RVV Lowering (DES-001)
LLVM RVV IR (llvm.call @llvm.riscv.*)
  ↓ LLVM Codegen
RISC-V Assembly
```

### 3. Vector Length Agnostic (VLA) Approach
- Single code path works with VLEN = 128, 256, 512, ...
- Runtime vector length discovery via `vsetvl`
- Validated across multiple VLEN values on QEMU

### 4. Specification-First Development
- Write specs before implementation
- Design documents capture decisions
- Tests verify against specifications

---

## Documentation Hierarchy

```
RISC-V_RVV_KERNEL_LIBRARY_PLAN.md (Top-level)
  ├─ Overview of 8 phases
  ├─ TDD approach
  ├─ Test strategy
  └─ Learning outcomes

├─ docs/guides/testing-guide.md
│   ├─ How to run tests
│   ├─ Test patterns
│   ├─ Multi-VLEN testing
│   └─ Debugging guide

└─ docs/design/ (Detailed design per component)
    ├─ DES-001: Vector operations (foundation)
    ├─ DES-002: MatMul (reference pattern)
    ├─ DES-003: Conv2D (extended kernel)
    ├─ DES-004: Attention (advanced kernel)
    └─ README.md (Index and phases)

Test Infrastructure (Ready to use)
├─ scripts/run-tests.sh
├─ tests/test_data_generator.py
├─ tests/qemu_runner.py
└─ tests/functional_validator.py

Specifications (From project)
├─ specs/targets/riscv-rvv.md
├─ specs/kernels/matmul.md
├─ specs/kernels/conv2d.md
└─ specs/kernels/attention.md
```

---

## How to Use This Documentation

### For Understanding the Project
1. Start: `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md` (overview)
2. Then: `docs/design/README.md` (phases and documents)
3. Deep dive: Specific DES-XXX document for that phase

### For Implementation
1. Read the relevant DES-XXX document completely
2. Follow test strategy exactly (RED → GREEN → REFACTOR)
3. Use `docs/guides/testing-guide.md` for test patterns
4. Run tests continuously with `./scripts/run-tests.sh`

### For Testing
1. Generate test data: `make test-data`
2. Run by category: `./scripts/run-tests.sh --lit` (or `--unit`, `--integration`)
3. Validate multi-VLEN: `./scripts/run-tests.sh --integration --qemu-vlen 256`
4. Functional validation: `make test-validate`

### For Contributing Additional Kernels
1. Copy `docs/design/TEMPLATE.md` → `DES-XXX-name.md`
2. Fill in requirements from spec
3. Design using DES-002 (MatMul) as pattern reference
4. Follow same TDD approach
5. Update `docs/design/README.md` index

---

## Files Created/Modified

### Documentation
- ✅ `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md` (new)
- ✅ `docs/guides/testing-guide.md` (new)
- ✅ `docs/design/DES-001-vector-operations-lowering.md` (new)
- ✅ `docs/design/DES-002-matmul-kernel.md` (new)
- ✅ `docs/design/DES-003-conv2d-kernel.md` (new)
- ✅ `docs/design/DES-004-attention-kernel.md` (new)
- ✅ `docs/design/README.md` (updated)
- ✅ `DEVELOPMENT_SUMMARY.md` (this file)

### Test Infrastructure
- ✅ `scripts/run-tests.sh` (new, executable)
- ✅ `tests/test_data_generator.py` (new, executable)
- ✅ `tests/qemu_runner.py` (new, executable)
- ✅ `tests/functional_validator.py` (new, executable)
- ✅ `Makefile` (updated with test targets)

### Git
- 3 commits pushed to `claude/mlir-riscv-rvv-kernels-Xi6CW`
  1. Development plan commit
  2. Phase 1 test infrastructure commit
  3. Design documents commit

---

## Getting Started with Phase 2

To begin implementing Phase 2 (Vector Operations Lowering):

1. **Read the Design**
   ```bash
   cat docs/design/DES-001-vector-operations-lowering.md
   ```

2. **Review Test Strategy**
   - See "Test Strategy" section in DES-001
   - Understand RED-GREEN-REFACTOR cycle

3. **Start RED Phase**
   - Create `tests/lit/Transforms/lower-to-rvv.mlir`
   - Create `tests/unit/RVVLoweringTest.cpp`
   - Write tests based on DES-001 test strategy
   - Confirm tests fail

4. **Then GREEN Phase**
   - Create lowering patterns
   - Implement type conversion
   - Get tests to pass

5. **Then REFACTOR Phase**
   - Implement vsetvl optimization
   - Profile and optimize
   - Ensure tests still pass

6. **Finally VALIDATE Phase**
   - Run: `make test-data`
   - Run: `make test-validate`
   - Run: `./scripts/run-tests.sh --integration --qemu-vlen 256`

---

## Success Metrics

### Per Phase
- ✅ **Phase 1**: Test infrastructure complete and working
- ⏳ **Phase 2**: Vector operations lowering complete (11-12 days)
- ⏳ **Phase 3**: MatMul kernel complete (4 weeks)
- ⏳ **Phase 4**: Conv2D kernel complete (2 weeks)
- ⏳ **Phase 5**: Attention kernel complete (2 weeks)
- ⏳ **Phase 6+**: Extended operations complete

### Overall Success Criteria
- ✅ Comprehensive design documentation created
- ⏳ 3 core kernels fully implemented (MatMul, Conv2D, Attention)
- ⏳ 10+ extended operations (activations, normalizations, reductions)
- ⏳ 100% test pass rate across all categories (lit, unit, functional, integration)
- ⏳ Multi-VLEN validation complete (VLEN=128, 256, 512)
- ⏳ Performance characterized and documented
- ⏳ IREE integration roadmap defined
- ⏳ Ready for contribution to upstream IREE project

---

## Next Steps

### Immediate (Next Session)
1. Review all design documents
2. Provide feedback on approach
3. Begin Phase 2 (Vector Operations) implementation
4. Start RED phase: Write tests for vector operations

### Short Term (Week 1-2)
- Complete Phase 2 vector operations lowering
- All tests passing (RED → GREEN → REFACTOR)
- Multi-VLEN validation successful

### Medium Term (Week 2-8)
- Implement Phase 3 (MatMul) with comprehensive TDD
- Establish performance baseline
- Validate against reference implementations

### Longer Term (Week 8+)
- Continue with Conv2D and Attention
- Optimize all kernels
- Plan IREE integration

---

## Key Resources

- **RISC-V RVV Spec**: `specs/targets/riscv-rvv.md`
- **Testing Guide**: `docs/guides/testing-guide.md`
- **Test Infrastructure**: `scripts/run-tests.sh` and tools
- **Design Templates**: `docs/design/TEMPLATE.md`
- **MLIR Docs**: https://mlir.llvm.org/
- **RISC-V RVV**: https://riscv.org/technical/specifications/

---

## Questions?

Refer to the specific design document for the phase you're working on. Each DES-XXX document has comprehensive sections on:
- Problem statement and scope
- Detailed technical design
- Test strategy with examples
- Implementation plan
- Success criteria

---

**Status**: Ready for Phase 2 implementation
**Last Updated**: 2026-02-06
**Branch**: `claude/mlir-riscv-rvv-kernels-Xi6CW`
