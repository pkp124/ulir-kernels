# Matrix Multiplication (matmul) Specification

## Overview

General matrix multiplication operation: `C = A × B`

This is a foundational operation for neural networks, used in:
- Fully connected layers
- Attention mechanisms (QK^T, attention × V)
- Embedding lookups with projection

## Operation Definition

```mlir
%C = kernel.matmul %A, %B : tensor<M×K×dtype>, tensor<K×N×dtype> -> tensor<M×N×dtype>
```

## Mathematical Definition

```
C[i, j] = Σ(k=0 to K-1) A[i, k] × B[k, j]
```

## Input/Output Specification

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `A` | `tensor<M×K×dtype>` | Left-hand side matrix |
| `B` | `tensor<K×N×dtype>` | Right-hand side matrix |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `C` | `tensor<M×N×dtype>` | Result matrix |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tile_sizes` | `array<i64>` | `[]` | Optional tiling hint `[tile_M, tile_N, tile_K]` |

### Type Constraints

- `dtype` must be a floating-point or integer type
- `A` and `B` must have the same element type
- `C` has the same element type as inputs
- Inner dimension of `A` (K) must match outer dimension of `B`

## Verification Rules

1. `A.rank == 2` (must be 2D matrix)
2. `B.rank == 2` (must be 2D matrix)
3. `A.shape[1] == B.shape[0]` (K dimensions match)
4. `C.shape == [A.shape[0], B.shape[1]]` (output shape)
5. `A.dtype == B.dtype == C.dtype` (type consistency)

## Lowering Strategy

### 1. Kernel → Linalg

```mlir
// Before
%C = kernel.matmul %A, %B

// After
%C_init = tensor.empty() : tensor<M×N×dtype>
%C_zero = linalg.fill ins(%zero) outs(%C_init)
%C = linalg.matmul ins(%A, %B) outs(%C_zero)
```

### 2. Tiling

Apply tiling transformation based on target:

```mlir
// Tiled version (example: 64×64×32 tiles)
scf.for %i = 0 to M step 64 {
  scf.for %j = 0 to N step 64 {
    scf.for %k = 0 to K step 32 {
      // Tiled matmul on 64×32 × 32×64 → 64×64
    }
  }
}
```

**Tiling considerations:**
- Tile M, N for L1 cache (typically 32-128 elements)
- Tile K for register reuse
- Consider vector length for vectorization

### 3. Vectorization

```mlir
// Inner loop vectorized
%a_vec = vector.load %A[%i, %k] : vector<VL×dtype>
%c_acc = scf.for %j = ... iter_args(%acc = %c_init) {
  %b_val = memref.load %B[%k, %j] : dtype
  %b_vec = vector.broadcast %b_val : vector<VL×dtype>
  %fma = vector.fma %a_vec, %b_vec, %acc
  scf.yield %fma
}
```

### 4. Target-Specific (RVV)

```mlir
// RVV lowering
%vl = rvv.setvl %len, e32, m4  // Set vector length
%a = rvv.vle32 %A_ptr          // Vector load
%c = rvv.vfmacc %c, %a, %b     // Fused multiply-accumulate
```

## Performance Model

### Arithmetic Intensity

- Operations: `2 × M × N × K` (multiply + add)
- Memory: `M×K + K×N + M×N` elements
- Intensity: `2×M×N×K / (M×K + K×N + M×N)`

For large square matrices (M = N = K):
- Intensity ≈ `2×N / 3` (compute-bound for large N)

### Optimal Tiling

For RISC-V with VLEN=256, f32:
- Tile M: 64 (multiple of VL=8)
- Tile N: 64
- Tile K: 32 (for register pressure)

## Test Cases

### Basic Cases

1. **Square matrices**: 64×64 × 64×64
2. **Rectangular**: 32×128 × 128×64
3. **Small**: 4×4 × 4×4
4. **Large**: 1024×1024 × 1024×1024

### Edge Cases

1. **Vector-matrix**: 1×K × K×N
2. **Matrix-vector**: M×K × K×1
3. **Single element**: 1×1 × 1×1
4. **Non-power-of-2**: 17×23 × 23×31

### Numerical Cases

1. **Identity**: A × I = A
2. **Zero**: A × 0 = 0
3. **Precision**: Check for accumulation errors

### Performance Cases

1. **Cache-resident**: Fits in L1
2. **Memory-bound**: Large matrices
3. **Various shapes**: Wide, tall, square

## References

- BLAS GEMM specification
- [GotoBLAS paper](https://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf)
- LLVM/MLIR Linalg documentation
