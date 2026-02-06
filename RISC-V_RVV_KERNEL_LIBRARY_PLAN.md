# RISC-V RVV Kernel Library Development Plan

## Project Vision

Build a comprehensive, well-tested library of ML kernels for RISC-V Vector Extension (RVV), following Test-Driven Development (TDD) principles. This library will serve as both a learning resource for MLIR and a foundation for contributing to IREE.

## Development Approach: TDD

**Core Principle**: Test-First Development
1. Write specification tests (MLIR lit tests for parsing/printing)
2. Write functional tests (C++ unit tests for correctness)
3. Implement kernel logic to make tests pass
4. Add optimization passes incrementally

---

## Phase 1: Foundation & Core Infrastructure

### 1.1 Setup Development Environment
- [ ] Create feature branch: `claude/mlir-riscv-rvv-kernels-Xi6CW`
- [ ] Set up test execution framework
- [ ] Configure QEMU for multi-VLEN testing (128, 256, 512 bits)
- [ ] Create validation harness for functional testing

### 1.2 Testing Infrastructure
- [ ] Set up lit tests for each kernel (parsing, lowering, verification)
- [ ] Set up C++ functional tests with reference implementations
- [ ] Set up integration tests (full pipeline: IR → RISC-V)
- [ ] Create test data generators for various tensor shapes

**Deliverables**:
- Test runners working for matmul, conv2d, attention
- Documentation on running tests with different VLEN values
- Reference test data

---

## Phase 2: Vector Operations Foundation (Prerequisite)

Before implementing kernels, establish core vector operation lowering to RISC-V RVV.

### 2.1 Vector Operation Lowering Patterns
**Operations to implement**:
- `vector.load` → `vle{SEW}.v`
- `vector.store` → `vse{SEW}.v`
- `vector.fma` → `vfmacc.vv`
- `vector.reduction<add>` → `vfredusum.vs`
- `vector.reduction<max>` → `vfredmax.vs`

### 2.2 Type Conversion
- Implement MLIR vector type → RISC-V RVV type mapping
- Handle LMUL selection for register grouping
- Support multiple element widths (f32, f16, i32, i8)

**Deliverables**:
- Operating lowering patterns in place
- Tests verifying vector ops lower correctly to RVV intrinsics

---

## Phase 3: MatMul Kernel - TDD Example (Start Here!)

### 3.1 Specification Phase
- [x] Specification already exists: `specs/kernels/matmul.md`
- [ ] Review and validate against RISC-V RVV constraints

### 3.2 Test-First Development

#### Step 1: Define Tests (Before Implementation)

**3.2.1 Lit Test: Parsing/Printing** (`tests/lit/Dialect/Kernel/matmul_basic.mlir`)
```mlir
// Test: Can parse and print ks.matmul without errors
// Test: Verify attributes (tile sizes, etc.)
// Test: Check type relationships (M×K × K×N → M×N)
```

**3.2.2 Lit Test: Verifier** (`tests/lit/Dialect/Kernel/matmul_verifier.mlir`)
```mlir
// Test: Rejects mismatched dimensions
// Test: Rejects incompatible element types
// Test: Accepts valid configurations
```

**3.2.3 Lit Test: Lowering Pipeline** (`tests/lit/Transforms/lower_matmul_to_rvv.mlir`)
```mlir
// Step 1: ks.matmul → linalg.matmul
// Step 2: Add tiling loops (scf.for)
// Step 3: Vectorize inner operations
// Step 4: Lower to RVV intrinsics (vfmacc.vv, etc.)
```

**3.2.4 C++ Functional Tests** (`tests/unit/MatMulTest.cpp`)
```cpp
// Test multiple configurations:
// - Small (4×4 × 4×4)
// - Medium (64×64 × 64×64)
// - Large (256×256 × 256×256)
// - Dynamic shapes
// - Various element types (f32, f16)

// Validation: Compare against reference implementation (Eigen/BLAS)
```

#### Step 2: Implement Kernel Lowering

**3.3.1 Kernel to Linalg Lowering**
- Convert `ks.matmul` to `linalg.matmul` + `tensor.empty`
- Verify operation structure

**3.3.2 Add Tiling Pass**
- Tile for L1 cache efficiency (M=64, N=64, K=32)
- Use `scf.for` loops around `linalg.matmul`

**3.3.3 Vectorization Pass**
- Convert tiled matmul to vector operations
- Generate `vector.load`, `vector.fma`, `vector.store`
- Handle edge cases (non-multiple-of-VL dimensions)

**3.3.4 RVV Lowering Pass**
- Map `vector.*` operations to `llvm.call @llvm.riscv.*`
- Manage `vsetvl` instructions
- Handle masking for tail elements

**3.3.5 Testing**
- Run each pass in isolation, verify tests pass
- Run full pipeline, compare output against reference

#### Step 3: Functional Validation

**3.3.6 Multi-VLEN Testing**
```bash
# Test with different VLEN values
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./test_matmul
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./test_matmul
qemu-riscv64 -cpu rv64,v=true,vlen=512 ./test_matmul
```

**3.3.7 Correctness Validation**
- Compute results on multiple VLEN configurations
- Compare against Eigen/BLAS reference
- Validate numerical accuracy (accounting for rounding)

#### Step 4: Performance Analysis

- Measure execution time vs. reference implementation
- Profile to identify bottlenecks
- Document performance characteristics

**Deliverables for Phase 3**:
- ✅ All MatMul tests passing
- ✅ Functional validation on QEMU (multiple VLEN)
- ✅ Performance baseline established
- ✅ Documentation of implementation approach

---

## Phase 4: Conv2D Kernel

Repeat TDD approach from Phase 3:
1. Define comprehensive tests
2. Implement lowering pipeline
3. Functional validation on QEMU
4. Performance analysis

**Considerations**:
- More complex tiling strategy (NHWC format)
- Handling padding and strides
- Memory access patterns (importance of data layout)

---

## Phase 5: Attention Kernel

### Scaled Dot-Product Attention (SDPA)
Repeat TDD approach with added complexity:
- Matrix multiply (Q·K^T)
- Softmax (row-wise reduction + exp + normalization)
- Final multiply (attention_weights · V)

**Complexity**: Multi-stage computation with reductions

---

## Phase 6: Additional Kernels & Operations

### Activation Functions
- ReLU
- GELU (approximate)
- SiLU

### Normalization
- Layer Normalization
- RMS Normalization

### Reductions
- Sum, Max, Mean reductions

**Development Model**: Each follows same TDD pattern as Phases 3-5

---

## Phase 7: Optimization & Tuning

### 7.1 Kernel Fusion
- Combine operations (e.g., MatMul → Add → ReLU)
- Reduce memory bandwidth

### 7.2 Register Allocation
- Fine-tune LMUL and tile sizes
- Optimize for register pressure

### 7.3 Instruction Scheduling
- Reorder operations to hide latency
- Pipeline memory operations

### 7.4 Performance Profiling
- Measure compute vs. memory bound
- Roofline model analysis

---

## Phase 8: Integration & Documentation

### 8.1 Build System Integration
- Create unified `make kernel-library` target
- Ensure tests run in CI/CD

### 8.2 Example Programs
- Create standalone examples for each kernel
- Include usage patterns and best practices

### 8.3 Documentation
- API reference for all kernels
- Performance characteristics by architecture
- Tuning guidelines

### 8.4 IREE Integration Planning
- Identify what can be upstreamed
- Prepare specifications for contribution
- Design interfaces for IREE integration

---

## Test Strategy Summary

### Test Pyramid

```
                    △ Integration Tests (Full pipeline)
                   △△ Functional Tests (Correctness)
                  △△△ Unit Tests (Pass isolation)
                 △△△△ Lit Tests (Parsing/Lowering)
                △△△△△ Specification Tests (Types)
```

### Test Coverage Per Kernel

| Test Type | Count | Scope |
|-----------|-------|-------|
| Specification | 3-5 | Type checking, verifier |
| Parsing/Lowering | 5-8 | Each transformation pass |
| Functional | 10-15 | Various shapes, dtypes, edge cases |
| Integration | 3-5 | Full pipeline, different VLEN |
| Performance | 2-3 | Baseline, profiling |

---

## Metrics & Success Criteria

### Phase 3 (MatMul) Success Criteria
- ✅ 100% test pass rate
- ✅ Functional correctness on QEMU (all VLEN values)
- ✅ Performance within 80% of reference BLAS
- ✅ Clean, documented code
- ✅ Comprehensive test coverage

### Overall Library Success Criteria
- ✅ All 3 core kernels (MatMul, Conv2D, Attention) working
- ✅ 10+ secondary operations (activations, normalizations)
- ✅ Multi-VLEN functional validation for all
- ✅ Clear performance characteristics documented
- ✅ Ready for IREE integration

---

## Git Workflow

### Branch Strategy
- Feature branch: `claude/mlir-riscv-rvv-kernels-Xi6CW`
- Commit after each completed test/implementation pair
- Clear commit messages following conventional commits

### Commit Structure (TDD Pattern)
```
feat: Add MatMul kernel tests (parsing, verifier, lowering)
- Define lit tests for ks.matmul syntax
- Add C++ functional tests with test data
- Tests currently fail (Red phase)

feat: Implement MatMul lowering to linalg
- Tests now pass (Green phase)

refactor: Optimize MatMul tiling strategy
- Improve performance without changing behavior

feat: Add RVV lowering for MatMul
- Full pipeline now works end-to-end
```

---

## Timeline & Milestones

**Estimated breakdown** (actual pace depends on learning curve):

- **Phase 1-2**: Foundation (2-3 days)
- **Phase 3**: MatMul TDD (3-5 days) - Most learning happens here
- **Phase 4**: Conv2D (2-3 days)
- **Phase 5**: Attention (3-4 days)
- **Phase 6**: Additional kernels (3-5 days)
- **Phase 7-8**: Optimization & integration (2-3 days)

**Total**: ~16-23 days of focused development

---

## Learning Outcomes

By completing this plan, you will have gained expertise in:

1. **MLIR Fundamentals**
   - Dialects and operations
   - Type system and verification
   - Control flow (scf.for, etc.)

2. **Compiler Passes**
   - Pattern matching and rewriting
   - IR transformation strategies
   - Lowering between abstraction levels

3. **Vector Architecture**
   - RISC-V RVV ISA semantics
   - Scalable vector code generation
   - Memory access optimization

4. **TDD in Systems Programming**
   - Test-first design approach
   - Incremental validation
   - Performance measurement

5. **IREE Integration**
   - How kernels fit into compiler pipelines
   - Specification-driven development
   - Contribution readiness

---

## Resources

- RISC-V RVV Specification: https://riscv.org/technical/specifications/
- MLIR Documentation: https://mlir.llvm.org/
- eMBARc MLIR: https://github.com/embarc-research/embarc-mlir (reference)
- IREE Project: https://github.com/openxla/iree
- Lit Testing: https://llvm.org/docs/CommandGuide/lit/
