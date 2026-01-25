# ks.matmul - Matrix Multiplication Specification

## Overview

General matrix multiplication operation: `C = A × B`

## Operation Definition

```mlir
%C = ks.matmul %A, %B : tensor<M×K×dtype>, tensor<K×N×dtype> -> tensor<M×N×dtype>
```

## Mathematical Definition

```
C[i, j] = Σ(k=0 to K-1) A[i, k] × B[k, j]
```

## Input/Output Specification

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `lhs` | `tensor<M×K×dtype>` | Left-hand side matrix |
| `rhs` | `tensor<K×N×dtype>` | Right-hand side matrix |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `result` | `tensor<M×N×dtype>` | Result matrix |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `tile_sizes` | `array<i64>` | `[]` | Optional tiling hint |

### Type Constraints

- `dtype` must be floating-point or integer
- Both operands must have same element type
- Inner dimensions must match (lhs.dim[1] == rhs.dim[0])

## Verification Rules

1. Both operands must be 2D ranked tensors
2. Inner dimensions must be compatible
3. Result shape must be [M, N]
4. Element types must match

## Lowering Strategy

### 1. ks.matmul → linalg.matmul

```mlir
// Before
%C = ks.matmul %A, %B

// After
%C_init = tensor.empty() : tensor<M×N×dtype>
%C_zero = linalg.fill ins(%zero) outs(%C_init)
%C = linalg.matmul ins(%A, %B) outs(%C_zero)
```

### 2. Tiling

```mlir
scf.for %i = 0 to M step tile_M {
  scf.for %j = 0 to N step tile_N {
    scf.for %k = 0 to K step tile_K {
      // Tiled matmul
    }
  }
}
```

### 3. Vectorization → Target Lowering

See `specs/targets/riscv-rvv.md` for RVV-specific lowering.

## Test Cases

1. Square matrices: 64×64 × 64×64
2. Rectangular: 32×128 × 128×64
3. Small: 4×4 × 4×4
4. Non-power-of-2: 17×23 × 23×31
5. Different dtypes: f32, f16, bf16
