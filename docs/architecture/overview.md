# Architecture Overview

This document describes the high-level architecture of the MLIR Kernel Generation project.

## Design Goals

1. **Portability**: Write kernels once, target multiple architectures
2. **Performance**: Generate code competitive with hand-tuned implementations
3. **Extensibility**: Easy to add new kernels and targets
4. **Testability**: Every component can be tested in isolation

## Compilation Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     High-Level IR                                │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Kernel Dialect                                          │    │
│  │  - kernel.matmul, kernel.conv2d, kernel.attention        │    │
│  │  - High-level semantics, no tiling decisions            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Tiling Pass
┌─────────────────────────────────────────────────────────────────┐
│                     Tiled IR                                     │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Linalg + SCF                                            │    │
│  │  - scf.for loops with tiled operations                  │    │
│  │  - Tile sizes chosen for target architecture            │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Vectorization Pass
┌─────────────────────────────────────────────────────────────────┐
│                     Vector IR                                    │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  Vector Dialect                                          │    │
│  │  - vector.load, vector.store, vector.fma                │    │
│  │  - Architecture-agnostic vector operations              │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ Target Lowering Pass
┌─────────────────────────────────────────────────────────────────┐
│                     Target-Specific IR                           │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  RVV Intrinsics / ARM SVE / x86 AVX                      │    │
│  │  - Target-specific vector operations                     │    │
│  │  - Optimal instruction selection                         │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼ LLVM Lowering
┌─────────────────────────────────────────────────────────────────┐
│                     LLVM IR                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  LLVM Dialect → LLVM IR → Target Assembly                │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Kernel Dialect

The `kernel` dialect provides high-level operations for AI workloads:

```mlir
// Matrix multiplication
%C = kernel.matmul %A, %B : tensor<M×K×f32>, tensor<K×N×f32> -> tensor<M×N×f32>

// 2D Convolution
%out = kernel.conv2d %input, %filter {strides = [1,1], padding = [1,1,1,1]}
       : tensor<N×H×W×C×f32>, tensor<KH×KW×C×OC×f32> -> tensor<N×OH×OW×OC×f32>

// Attention
%out = kernel.scaled_dot_product_attention %Q, %K, %V
       : tensor<B×L×D×f32>, tensor<B×L×D×f32>, tensor<B×L×D×f32> -> tensor<B×L×D×f32>
```

Key characteristics:
- **Pure operations**: No side effects, easy to reason about
- **Shape inference**: Output shapes derived from inputs
- **Verification**: Strong type checking and shape validation

### 2. Transformation Passes

#### Tiling Pass
Converts high-level operations into tiled loops:

```mlir
// Before: kernel.matmul %A, %B
// After:
scf.for %i = 0 to M step tile_M {
  scf.for %j = 0 to N step tile_N {
    scf.for %k = 0 to K step tile_K {
      // Tiled matmul on tile_M × tile_K × tile_N
    }
  }
}
```

Tiling decisions based on:
- Target vector register size
- Cache hierarchy (L1, L2)
- Memory bandwidth

#### Vectorization Pass
Converts scalar operations to vector operations:

```mlir
// Before: linalg.matmul
// After:
%a = vector.load %A[%i, %k] : vector<VL×f32>
%b = vector.broadcast %B[%k, %j] : f32 -> vector<VL×f32>
%c = vector.fma %a, %b, %acc : vector<VL×f32>
```

### 3. Target Backends

Each target implements:
1. **Type conversion**: MLIR types → target types
2. **Operation lowering**: vector ops → target intrinsics
3. **Optimization patterns**: Target-specific optimizations

#### RISC-V RVV Backend

```mlir
// vector.load → vle32.v
// vector.store → vse32.v
// vector.fma → vfmacc.vv
```

Key considerations:
- **Scalable vectors**: VLEN not known at compile time
- **LMUL configuration**: Register grouping for larger vectors
- **Masking**: Handle non-power-of-2 dimensions

### 4. Runtime Library

Provides:
- Memory allocation (aligned for vector access)
- Kernel invocation API
- Profiling and debugging support

## Design Decisions

### Why MLIR?

1. **Multi-level IR**: Natural fit for progressive lowering
2. **Dialects**: Extensible, can mix different abstractions
3. **Infrastructure**: TableGen, passes, pattern rewriting
4. **LLVM integration**: Direct path to optimized code

### Why Tiled + Vectorized Approach?

1. **Cache efficiency**: Tiling keeps data in cache
2. **Register utilization**: Vectorization uses SIMD registers
3. **Target flexibility**: Same tiling, different vector widths

### Why Specification-Driven?

1. **Clarity**: Clear contracts before implementation
2. **Testing**: Spec defines expected behavior
3. **Documentation**: Specs are living documentation

## Extension Points

### Adding a New Kernel

1. Define operation in `kernel` dialect
2. Add lowering pattern to Linalg/Vector
3. Existing target backends work automatically

### Adding a New Target

1. Create target-specific lowering passes
2. Map vector operations to target intrinsics
3. Add target-specific optimizations

## Performance Model

For each kernel, we track:
- **Arithmetic intensity**: FLOPS / bytes moved
- **Memory access pattern**: Sequential, strided, random
- **Parallelism**: Available at each loop level

This informs:
- Tiling decisions
- Prefetching strategies
- Parallelization approach
