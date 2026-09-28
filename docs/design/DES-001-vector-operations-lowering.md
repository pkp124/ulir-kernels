# DES-001: Vector Operations Lowering to RISC-V RVV

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-06 |
| **Phase** | Phase 2 - Foundation |
| **Priority** | Critical |

> **Current status (2026-09-28):** Historical draft. The RVV pipeline that
> shipped is [DES-009](DES-009-m4-rvv-lowering.md). Use that document for the
> pass sequence and validation story.

---

## Context

### Problem Statement

KernelSmith kernels are initially expressed using MLIR's vector dialect (`vector.*` operations). To generate efficient RISC-V code for the RVV extension, these high-level vector operations must be lowered to RISC-V RVV intrinsic calls and instructions.

**Key Challenge**: RISC-V RVV uses **Vector Length Agnostic (VLA)** semantics where vector length is runtime-determined. Code must work correctly regardless of VLEN (128-16384 bits).

### Background

- RISC-V RVV is KernelSmith's primary target architecture
- VLEN is not known at compile time (unlike AVX-512 which is always 512 bits)
- Operations must use `vsetvl` to configure vector parameters (SEW, LMUL, VL)
- Masking is needed for non-power-of-2 dimensions

### Scope

This design covers lowering the core vector operations needed by KernelSmith kernels:
- Load/Store operations
- Arithmetic operations (add, mul, fma)
- Reductions (sum, max, min)
- Conversions and type operations

---

## Requirements

From specification: `specs/targets/riscv-rvv.md`

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Implement pass to lower `vector.load` → `vle{SEW}.v` | Must Have |
| REQ-2 | Implement pass to lower `vector.store` → `vse{SEW}.v` | Must Have |
| REQ-3 | Implement pass to lower `vector.fma` → `vfmacc.vv` | Must Have |
| REQ-4 | Implement pass to lower `vector.reduction<add>` → `vfredusum.vs` | Must Have |
| REQ-5 | Implement pass to lower `vector.reduction<max>` → `vfredmax.vs` | Must Have |
| REQ-6 | Type conversion: MLIR vector types → RISC-V RVV types | Must Have |
| REQ-7 | Support dynamic VLEN via `vsetvl` generation | Must Have |
| REQ-8 | Support masking for tail elements | Should Have |
| REQ-9 | Support LMUL register grouping for larger vectors | Should Have |
| REQ-10 | Preserve correctness through lit tests + functional validation | Must Have |

---

## Design

### Overview

The vector operations lowering consists of three main components:

1. **Type Conversion Layer**: Maps MLIR vector types to RISC-V RVV type parameters (SEW, LMUL, VL)
2. **Operation Lowering Patterns**: Implements rewrite rules for each vector operation
3. **VLA Code Generation**: Generates runtime `vsetvl` calls and handles dynamic vector lengths

### Architecture

```
MLIR Vector IR
  ├─ vector.load [...]
  ├─ vector.fma [...]
  ├─ vector.store [...]
  └─ vector.reduction [...]
        ↓
    [Type Inference Pass]
        ↓
    [Lowering Patterns]
        ↓
LLVM IR with RVV Intrinsics
  ├─ llvm.call @llvm.riscv.vle32.v [...]
  ├─ llvm.call @llvm.riscv.vfmacc.vv [...]
  ├─ llvm.call @llvm.riscv.vse32.v [...]
  └─ llvm.call @llvm.riscv.vfredusum.vs [...]
        ↓
    [LLVM → ASM]
        ↓
RISC-V Assembly
```

### Component Design

#### 1. Type Conversion System

**Goal**: Map abstract vector types to concrete RVV parameters

**Input**: MLIR vector type (e.g., `vector<8xf32>`)
**Output**: RVV parameters (SEW=32, LMUL, VL)

```cpp
struct RVVTypeInfo {
  unsigned sew;           // Selected Element Width (8, 16, 32, 64)
  float lmul;             // Length Multiplier (1/8, 1/4, 1/2, 1, 2, 4, 8)
  unsigned vlmax;         // Max elements = VLEN × LMUL / SEW (runtime)
  llvm::Type* elementType; // LLVM element type
};

// Type conversion function
std::optional<RVVTypeInfo> convertMLIRVectorToRVV(VectorType vectorType);
```

**Conversion Table**:
```
MLIR Type         → SEW  | LMUL Selection Strategy
─────────────────────────┼──────────────────────────
vector<8xf32>     → 32   | LMUL = 1 (8 elements × 32 bits = 256 bits)
vector<16xf32>    → 32   | LMUL = 2 (16 elements × 32 bits = 512 bits)
vector<32xf32>    → 32   | LMUL = 4 (32 elements × 32 bits = 1024 bits)
vector<4xf64>     → 64   | LMUL = 2 (4 elements × 64 bits = 256 bits)
vector<Nxf32>     → 32   | LMUL auto-selected for best performance
```

#### 2. VsetvlGuidance Pass

**Purpose**: Minimize `vsetvl` instructions by identifying where they can be reused

Strategy:
- Insert `vsetvl` at function entry and after control flow merges
- Reuse `vsetvl` results in sequential operations
- Track active configuration to avoid redundant calls

```mlir
// Example: Before optimization
llvm.call @llvm.riscv.vsetvli(8, 32, ...) → %vl1
llvm.call @llvm.riscv.vle32.v(..., %vl1)
llvm.call @llvm.riscv.vsetvli(8, 32, ...) → %vl2  // Redundant!
llvm.call @llvm.riscv.vfmacc.vv(..., %vl2)

// After optimization
llvm.call @llvm.riscv.vsetvli(8, 32, ...) → %vl
llvm.call @llvm.riscv.vle32.v(..., %vl)
llvm.call @llvm.riscv.vfmacc.vv(..., %vl)        // Reuse %vl
```

#### 3. Operation Lowering Patterns

**Pattern Structure** (using MLIR's RewritePattern framework):

```cpp
class VectorLoadPattern : public ConversionPattern {
  matchAndRewrite(vector::LoadOp op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) {
    // 1. Get type info for vector type
    auto typeInfo = convertMLIRVectorToRVV(op.getVectorType());

    // 2. Generate vsetvl call
    Value vl = rewriter.create<llvm::CallOp>(
      getFunctionDeclaration(module, "llvm.riscv.vsetvli"),
      ...
    );

    // 3. Generate vle{SEW}.v intrinsic
    rewriter.replaceOpWithNewOp<llvm::CallOp>(
      op, getFunctionDeclaration(module,
        llvm::Twine("llvm.riscv.vle") + typeInfo.sewStr() + ".v"),
      ArrayRef<Value>{/*ptr*/, vl}
    );
  }
};
```

**Per-Operation Patterns**:

| MLIR Op | RVV Intrinsic | Pattern Notes |
|---------|---------------|---------------|
| `vector.load` | `vle{SEW}.v` | Indexed by address |
| `vector.store` | `vse{SEW}.v` | Requires address + data + VL |
| `arith.addf` | `vfadd.vv` | Element-wise, needs VL |
| `arith.mulf` | `vfmul.vv` | Element-wise, needs VL |
| `vector.fma` | `vfmacc.vv` | Three-operand, accumulates |
| `vector.reduction<add>` | `vfredusum.vs` | Produces scalar result |
| `vector.reduction<max>` | `vfredmax.vs` | Produces scalar result |

#### 4. Tail Element Handling

**Challenge**: When vector length is not a multiple of element count

**Solution - Masking**:
```mlir
// Original: vector<10xf32>
// VLEN=256 → 8 elements per operation
// Need 2 operations: one for 8 elements, one for 2 elements

// Stage 1: Set VL=8 for full vectors
llvm.call @llvm.riscv.vsetvli(8, 32, ...) → %vl1
llvm.call @llvm.riscv.vle32.v(..., %vl1)

// Stage 2: Set VL=2 for remaining elements
llvm.call @llvm.riscv.vsetvli(2, 32, ...) → %vl2
llvm.call @llvm.riscv.vle32.v(...+offset, %vl2)
```

---

## Interface Design

### Pass Interface

```cpp
/// Create RVV lowering pass
std::unique_ptr<Pass> createLowerToRVVPass();

// Usage in pipeline:
// pm.addPass(createLowerToRVVPass());
```

### Transformation Sequence

```
Input:  vector.load, vector.fma, vector.store (MLIR Vector IR)
  ↓
Type Inference Pass
  ↓
VsetvlGuidance Pass (optional optimization)
  ↓
Operation Lowering Pass
  ↓
Output: llvm.call @llvm.riscv.vle*, vfmacc, etc. (LLVM RVV IR)
```

### MLIR Example

```mlir
// Input: Vector operations
func.func @matmul_tile(%A: memref<64xf32>, %B: memref<64xf32>, %C: memref<64xf32>) {
  %c0 = arith.constant 0 : index
  %cstride = arith.constant 8 : index

  scf.for %i = %c0 to %c64 step %cstride {
    %v_a = vector.load %A[%i] : memref<64xf32>, vector<8xf32>
    %v_b = vector.load %B[%i] : memref<64xf32>, vector<8xf32>
    %v_c = vector.fma %v_a, %v_b, %v_c : vector<8xf32>
    vector.store %v_c, %C[%i] : memref<64xf32>, vector<8xf32>
  }
  return
}

// Output: RISC-V RVV IR
func.func @matmul_tile(%A: i64, %B: i64, %C: i64) {
  %vl = llvm.call @llvm.riscv.vsetvli(8, 32, 0) : () -> i64

  scf.for %i = ... {
    %v_a = llvm.call @llvm.riscv.vle32.v(%A, %vl) : ...
    %v_b = llvm.call @llvm.riscv.vle32.v(%B, %vl) : ...
    %v_c = llvm.call @llvm.riscv.vfmacc.vv(%v_a, %v_b, %v_c) : ...
    llvm.call @llvm.riscv.vse32.v(%C, %v_c, %vl) : ...
  }
  return
}
```

---

## Data Flow

```
Kernel MLIR (ks.matmul)
  ↓
Lower to Linalg (ks.matmul → linalg.matmul)
  ↓
Add Tiling (scf.for loops + smaller matmul)
  ↓
Vectorize (vector.load, vector.fma, vector.store)
  ↓ [Phase 2: VECTOR OPERATIONS LOWERING]
  ├─ Type Conversion
  ├─ Vsetvl Guidance
  └─ Operation Lowering
  ↓
LLVM RVV IR (llvm.call @llvm.riscv.*)
  ↓
LLVM Codegen → ASM
```

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Template all operations at compile-time | Fast codegen | Requires known VLEN | Rejected |
| Use RISC-V C intrinsics | Standard API | Less control | Rejected |
| Generate vsetvl for each operation | Simple | Excessive vsetvl instructions | Rejected |
| **Vsetvl guidance + pattern matching** | Efficient, flexible, scalable | More complex | **Selected** |

### Rationale

The selected approach balances several concerns:
1. **Efficiency**: Minimizes `vsetvl` instructions which are expensive
2. **Flexibility**: Works with any VLEN at runtime
3. **Maintainability**: Centralized pattern definitions
4. **Scalability**: Easy to add new operations

---

## Test Strategy

### Unit Tests (`tests/unit/RVVLoweringTest.cpp`)

- [ ] Type conversion: `vector<8xf32>` → (SEW=32, LMUL=1)
- [ ] Type conversion: `vector<16xf32>` → (SEW=32, LMUL=2)
- [ ] Type conversion: `vector<4xf64>` → (SEW=64, LMUL=1)
- [ ] Invalid type rejection: `vector<3xf32>` → error
- [ ] Vsetvl guidance: Reuse across operations
- [ ] Vsetvl guidance: After control flow split

### Lit Tests (`tests/lit/Transforms/lower-to-rvv.mlir`)

- [ ] `vector.load` → `llvm.call @llvm.riscv.vle*.v`
- [ ] `vector.store` → `llvm.call @llvm.riscv.vse*.v`
- [ ] `vector.fma` → `llvm.call @llvm.riscv.vfmacc.vv`
- [ ] `vector.reduction<add>` → `llvm.call @llvm.riscv.vfredusum.vs`
- [ ] `vector.reduction<max>` → `llvm.call @llvm.riscv.vfredmax.vs`
- [ ] Correct vsetvl placement
- [ ] Type matches operation (f32 → vle32.v, f64 → vle64.v)

### Integration Tests (`tests/integration/rvv_lowering_integration.py`)

- [ ] End-to-end: Vector IR → LLVM RVV IR → RISC-V Assembly
- [ ] QEMU execution with VLEN=128
- [ ] QEMU execution with VLEN=256
- [ ] QEMU execution with VLEN=512
- [ ] Numerical correctness (vs reference C implementation)
- [ ] Dynamic shapes and tail handling

### Edge Cases

- [ ] Empty vectors (if applicable)
- [ ] Single-element vectors
- [ ] Non-aligned memory access
- [ ] Mixed element types in same function
- [ ] VL greater than natural vector size (with LMUL > 1)

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| VLEN dependency | High | Medium | Runtime parameter discovery via vsetvli |
| Inefficient vsetvl usage | Medium | Medium | Implement vsetvl guidance pass |
| Type mismatch bugs | High | Low | Comprehensive unit tests |
| Tail element correctness | Medium | Medium | Explicit test cases for non-divisible lengths |
| LLVM intrinsic availability | High | Low | Verify with LLVM 18+ documentation |

---

## Open Questions

- [ ] Should we support integer vector operations (vi, vi) in Phase 2, or defer to Phase 3+?
- [ ] How to handle vector gather/scatter efficiently?
- [ ] Should masking be automatic or explicit in IR?
- [ ] Performance profiling: What's acceptable overhead vs scalar code?

---

## Dependencies

- LLVM 18+ with RISC-V support
- MLIR Vector dialect
- MLIR LLVM dialect
- RISC-V RVV specification understanding

---

## Implementation Plan

### Stage 1: Type Conversion (2-3 days)
- [ ] Implement `RVVTypeInfo` structure
- [ ] Create `convertMLIRVectorToRVV()` function
- [ ] Add type conversion tests
- [ ] Document type mapping table

### Stage 2: Basic Operation Lowering (3-4 days)
- [ ] Implement `LowerVectorLoadToRVV` pattern
- [ ] Implement `LowerVectorStoreToRVV` pattern
- [ ] Implement `LowerVectorFmaToRVV` pattern
- [ ] Add lit tests for each operation
- [ ] Create integration test framework

### Stage 3: Reductions & Advanced Ops (2-3 days)
- [ ] Implement `LowerVectorReductionToRVV` pattern
- [ ] Add reduction-specific tests
- [ ] Handle mask generation for reductions

### Stage 4: Vsetvl Optimization (2-3 days)
- [ ] Implement `VsetvlGuidancePass`
- [ ] Add tests for vsetvl reuse
- [ ] Measure codegen efficiency improvement

### Stage 5: Validation & Hardening (2-3 days)
- [ ] QEMU multi-VLEN testing
- [ ] Performance profiling
- [ ] Edge case hardening
- [ ] Documentation

---

## Success Criteria

- ✅ All lit tests pass (parsing, lowering, type verification)
- ✅ All unit tests pass (type conversion, patterns)
- ✅ Integration tests pass on QEMU (VLEN=128, 256, 512)
- ✅ Numerical correctness verified against reference C code
- ✅ Code review completed
- ✅ Documentation complete

---

## Review History

(To be filled in during review process)

---

## Related Documents

- [RISC-V RVV Specification](specs/targets/riscv-rvv.md)
- [MatMul Kernel Design](DES-002-matmul-kernel.md)
- [Testing Guide](../guides/testing-guide.md)
