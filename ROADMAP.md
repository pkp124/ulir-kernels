# KernelSmith Roadmap

This document defines the project milestones and long-term vision. The Manager Agent prioritizes work that advances milestone completion.

## Vision

KernelSmith will be the go-to framework for generating optimized AI kernels across multiple architectures, starting with RISC-V RVV and expanding to other targets.

---

## Milestones

### M1: Foundation ✅
**Goal:** Project infrastructure and basic dialect

**Status:** Complete

**Deliverables:**
- [x] CMake build system with CTest
- [x] Agent-based development workflow
- [x] Kernel dialect definition (TableGen)
- [x] Basic operations defined
- [x] ks-opt tool structure

**Success Criteria:**
- [x] Project builds with MLIR
- [x] Development workflow documented

---

### M2: Core Dialect
**Goal:** Working kernel dialect with basic operations

**Status:** Not Started

**Target:** First working ks-opt that can parse and verify kernels

**Deliverables:**
- [ ] ks.matmul operation (parse, print, verify)
- [ ] ks.relu, ks.gelu, ks.softmax operations
- [ ] ks.attention operation
- [ ] Comprehensive lit tests
- [ ] Verifiers catch all invalid inputs

**Success Criteria:**
- [ ] `ks-opt input.mlir` parses all defined operations
- [ ] Round-trip (parse → print → parse) works
- [ ] Invalid inputs produce clear error messages
- [ ] All lit tests pass

**Tasks:**
| Task | Status | Required |
|------|--------|----------|
| TASK-002 | ⬜ | Yes |
| TASK-003 | ⬜ | Yes |
| TASK-004 | ⬜ | Yes |
| TASK-005 | ⬜ | Yes |

---

### M3: Lowering Pipeline
**Goal:** Lower ks operations to linalg/vector dialects

**Status:** Not Started

**Target:** Complete lowering from ks.* to vectorized code

**Deliverables:**
- [ ] `--ks-lower-to-linalg` pass
- [ ] `--ks-tile` pass with configurable tile sizes
- [ ] `--ks-vectorize` pass
- [ ] End-to-end pipeline from ks.matmul to vector ops

**Success Criteria:**
- [ ] ks.matmul → linalg.matmul lowering works
- [ ] Tiling produces correct scf.for loops
- [ ] Vectorization produces vector.* operations
- [ ] Lowered code is functionally equivalent

**Depends On:** M2

**Tasks:**
| Task | Status | Required |
|------|--------|----------|
| TASK-010 | ⬜ | Yes |
| TASK-011 | ⬜ | Yes |
| TASK-012 | ⬜ | Yes |

---

### M4: RISC-V RVV Backend
**Goal:** Generate RISC-V RVV assembly from kernels

**Status:** Not Started

**Target:** Complete pipeline to RVV assembly that runs on QEMU

**Deliverables:**
- [ ] `--ks-lower-to-rvv` pass
- [ ] RVV intrinsic mapping
- [ ] Proper vsetvl insertion
- [ ] Tail handling with masking
- [ ] End-to-end: ks.matmul → RVV assembly

**Success Criteria:**
- [ ] Generated assembly is valid RISC-V
- [ ] Runs correctly on QEMU with RVV
- [ ] Works with multiple VLEN values (128, 256, 512)
- [ ] Performance is reasonable (within 2x of reference)

**Depends On:** M3

**Tasks:**
| Task | Status | Required |
|------|--------|----------|
| TASK-020 | ⬜ | Yes |
| TASK-021 | ⬜ | Yes |
| TASK-022 | ⬜ | Yes |

---

### M5: Optimization & Performance
**Goal:** Optimized kernels competitive with hand-tuned code

**Status:** Not Started

**Target:** Performance within 80% of hand-optimized implementations

**Deliverables:**
- [ ] Performance benchmarking infrastructure
- [ ] Tiling optimization for cache hierarchy
- [ ] Register allocation optimization
- [ ] Prefetching for memory-bound kernels
- [ ] LMUL tuning for RVV

**Success Criteria:**
- [ ] GEMM achieves 80%+ of theoretical peak
- [ ] Benchmarks against reference implementations
- [ ] Performance regression tests

**Depends On:** M4

---

### M6: Additional Targets (Future)
**Goal:** Support additional architectures

**Status:** Planning

**Potential Targets:**
- ARM SVE/SVE2
- x86 AVX-512
- GPU backends (CUDA, ROCm)

**Depends On:** M4

---

## Milestone Timeline

```
M1 ✅ ──→ M2 ──→ M3 ──→ M4 ──→ M5 ──→ M6
Foundation   Core    Lowering   RVV    Optimize  More
             Dialect Pipeline  Backend           Targets
```

---

## Current Focus

**Active Milestone:** M2 (Core Dialect)

**Rationale:** M2 is the foundation for all subsequent work. Without a working dialect, we cannot build lowering passes or backends.

**Next Actions:**
1. Complete TASK-002 (Kernel Dialect Core)
2. Complete TASK-003 (ks.matmul)
3. Complete TASK-004 (Activation operations)
4. Complete TASK-005 (ks.attention)

---

## Milestone Prioritization Rules

The Manager Agent follows these rules:

1. **Milestone-aligned work takes priority** over other work
2. **Required tasks** for current milestone come before optional tasks
3. **Blocking tasks** (dependencies for others) come first
4. **Do not start next milestone** until current one is complete
5. **Technical debt** can be addressed if it blocks milestone progress

---

## Milestone Review

At milestone completion:

1. Verify all success criteria met
2. Document lessons learned
3. Update roadmap with actuals
4. Plan next milestone in detail
5. Celebrate! 🎉
