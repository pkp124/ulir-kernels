# RISC-V RVV Target Specification

## Overview

RISC-V Vector Extension (RVV) is the first target architecture for kernel generation. RVV provides a scalable vector ISA where the vector length is not fixed at compile time.

## Architecture Characteristics

### Vector Length Agnostic (VLA)

Unlike fixed-width SIMD (AVX, NEON), RVV:
- Vector length (VLEN) ranges from 128 to 16384 bits
- Code works correctly regardless of VLEN
- Use `vsetvl` to query and set vector length

### Key Parameters

| Parameter | Description | Values |
|-----------|-------------|--------|
| VLEN | Vector register width | 128, 256, 512, ... bits |
| SEW | Selected Element Width | 8, 16, 32, 64 bits |
| LMUL | Length Multiplier | 1/8, 1/4, 1/2, 1, 2, 4, 8 |
| VLMAX | Max elements per op | VLEN × LMUL / SEW |

### Register File

- 32 vector registers (v0-v31)
- v0 often reserved for masks
- Registers can be grouped (LMUL > 1)

## Element Types

| SEW | MLIR Type | RVV Type | Use Case |
|-----|-----------|----------|----------|
| 8 | i8/f8 | e8 | Quantized models |
| 16 | i16/f16/bf16 | e16 | Mixed precision |
| 32 | i32/f32 | e32 | Standard precision |
| 64 | i64/f64 | e64 | High precision |

## Instruction Mapping

### Memory Operations

| MLIR | RVV | Description |
|------|-----|-------------|
| `vector.load` | `vle{SEW}.v` | Unit-stride load |
| `vector.store` | `vse{SEW}.v` | Unit-stride store |
| `vector.gather` | `vluxei{SEW}.v` | Indexed load |
| `vector.scatter` | `vsuxei{SEW}.v` | Indexed store |
| `vector.maskedload` | `vle{SEW}.v, v0.t` | Masked load |

### Arithmetic Operations

| MLIR | RVV | Description |
|------|-----|-------------|
| `arith.addf` | `vfadd.vv` | Vector add |
| `arith.mulf` | `vfmul.vv` | Vector multiply |
| `vector.fma` | `vfmacc.vv` | Fused multiply-add |
| `arith.maxf` | `vfmax.vv` | Element-wise max |
| `arith.cmpf` | `vmf{lt,le,eq,...}` | Compare |

### Reduction Operations

| MLIR | RVV | Description |
|------|-----|-------------|
| `vector.reduction<add>` | `vfredusum.vs` | Sum reduction |
| `vector.reduction<max>` | `vfredmax.vs` | Max reduction |

### Special Operations

| MLIR | RVV | Description |
|------|-----|-------------|
| `vector.broadcast` | `vfmv.v.f` | Broadcast scalar |
| `vector.splat` | `vmv.v.x` | Splat integer |
| `vector.shuffle` | `vrgather.vv` | Permute elements |

## Lowering Patterns

### Pattern 1: Simple Element-wise

```mlir
// MLIR Vector
%c = arith.addf %a, %b : vector<VL×f32>

// RVV
vsetvli t0, a0, e32, m1  // Set VL
vle32.v v1, (a1)         // Load a
vle32.v v2, (a2)         // Load b
vfadd.vv v3, v1, v2      // Add
vse32.v v3, (a3)         // Store c
```

### Pattern 2: Reduction

```mlir
// MLIR Vector
%sum = vector.reduction<add> %vec : vector<VL×f32> to f32

// RVV
vsetvli t0, a0, e32, m1
vle32.v v1, (a1)
vmv.s.x v2, zero         // Initialize accumulator
vfredusum.vs v2, v1, v2  // Reduce
vfmv.f.s fa0, v2         // Extract scalar
```

### Pattern 3: Fused Multiply-Accumulate

```mlir
// MLIR Vector
%d = vector.fma %a, %b, %c : vector<VL×f32>

// RVV
vfmacc.vv v3, v1, v2     // v3 = v3 + v1 * v2
```

### Pattern 4: Masked Operations

```mlir
// MLIR Vector
%result = arith.select %mask, %a, %b : vector<VL×i1>, vector<VL×f32>

// RVV
vmand.mm v0, v4, v4      // Set mask register
vmerge.vvm v3, v2, v1, v0  // Select based on mask
```

## Tiling Strategy

### For GEMM (Matrix Multiplication)

Optimal tiling depends on:
1. **VLEN**: Determines inner tile width
2. **L1 cache**: Outer tile sizes
3. **Register count**: Accumulator tiles

Example for VLEN=256, f32:
```
VLMAX = 256 / 32 = 8 elements

Tile sizes:
- M_tile = 8 (one vector)
- N_tile = 8 (unroll for accumulators)
- K_tile = 32 (for register reuse)
```

### Register Allocation

With LMUL=4 (8 register groups):
- 2 groups for A tiles
- 4 groups for C accumulators (unrolled)
- 2 groups for B loads (rotated)

## Tail Handling

For non-multiple-of-VL dimensions:

```asm
# Process full vectors
loop:
    vsetvli t0, a0, e32, m1
    # ... process t0 elements
    sub a0, a0, t0
    bnez a0, loop
```

The `vsetvli` automatically handles the tail by returning fewer elements.

## Performance Optimization

### 1. Minimize vsetvl Instructions

```asm
# Bad: vsetvl in every iteration
loop:
    vsetvli t0, a0, e32, m1
    vle32.v v1, (a1)
    ...

# Good: vsetvl once, reuse VL
    vsetvli t0, a0, e32, m1
loop:
    vle32.v v1, (a1)
    ...
    sub a0, a0, t0
    bnez a0, loop
```

### 2. Use LMUL for Compute-Bound Kernels

Higher LMUL = more elements per instruction = better throughput.

```asm
# LMUL=1: 8 elements (VLEN=256, f32)
vsetvli t0, a0, e32, m1

# LMUL=4: 32 elements
vsetvli t0, a0, e32, m4
```

### 3. Overlap Load and Compute

```asm
# Pipeline: load next while computing current
vle32.v v2, (a1)     # Load next
vfmacc.vv v8, v1, v4  # Compute current
```

### 4. Use Widening/Narrowing for Mixed Precision

```asm
# FP16 input, FP32 accumulator
vle16.v v1, (a1)          # Load fp16
vfwcvt.f.f.v v2, v1       # Widen to fp32
vfmacc.vv v8, v2, v4      # Accumulate in fp32
```

## Testing Strategy

### Functional Testing

1. **QEMU with RVV**: Fast functional simulation
   ```bash
   qemu-riscv64 -cpu rv64,v=true ./kernel_test
   ```

2. **Spike**: Reference simulator
   ```bash
   spike --isa=rv64gcv pk ./kernel_test
   ```

### Testing Multiple VLENs

Test with different VLEN to ensure VLA correctness:
```bash
# QEMU with specific VLEN
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./test
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./test
qemu-riscv64 -cpu rv64,v=true,vlen=512 ./test
```

## References

- [RISC-V V Extension Specification](https://github.com/riscv/riscv-v-spec)
- [RISC-V Vector Programming Manual](https://github.com/riscv/riscv-v-spec/blob/master/v-spec.adoc)
- [LLVM RISC-V Vector Support](https://llvm.org/docs/RISCVUsage.html#risc-v-vector-extension-rvv)
