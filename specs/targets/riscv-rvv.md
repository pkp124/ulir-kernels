# RISC-V RVV Target Specification

## Overview

RISC-V Vector Extension (RVV) is the primary target for KernelSmith. RVV provides a scalable vector ISA where vector length is runtime-determined.

## Key Concepts

### Vector Length Agnostic (VLA)

- VLEN ranges from 128 to 16384 bits
- Code works regardless of VLEN
- Use `vsetvl` to set/query vector length

### Configuration Parameters

| Parameter | Description | Range |
|-----------|-------------|-------|
| VLEN | Vector register width | 128-16384 bits |
| SEW | Selected Element Width | 8, 16, 32, 64 bits |
| LMUL | Length Multiplier | 1/8, 1/4, 1/2, 1, 2, 4, 8 |
| VLMAX | Max elements/op | VLEN × LMUL / SEW |

## Instruction Mapping

### Memory Operations

| KernelSmith/Vector | RVV | Description |
|-------------------|-----|-------------|
| `vector.load` | `vle{SEW}.v` | Unit-stride load |
| `vector.store` | `vse{SEW}.v` | Unit-stride store |
| `vector.gather` | `vluxei{SEW}.v` | Indexed load |

### Arithmetic

| KernelSmith/Vector | RVV | Description |
|-------------------|-----|-------------|
| `arith.addf` | `vfadd.vv` | Vector add |
| `arith.mulf` | `vfmul.vv` | Vector multiply |
| `vector.fma` | `vfmacc.vv` | Fused multiply-add |
| `arith.maxf` | `vfmax.vv` | Element-wise max |

### Reductions

| KernelSmith/Vector | RVV | Description |
|-------------------|-----|-------------|
| `vector.reduction<add>` | `vfredusum.vs` | Sum reduction |
| `vector.reduction<max>` | `vfredmax.vs` | Max reduction |

## Lowering Pipeline

```
ks.matmul
    ↓ --ks-lower-to-linalg
linalg.matmul
    ↓ --ks-tile
scf.for (tiled)
    ↓ --ks-vectorize
vector.load/fma/store
    ↓ --ks-lower-to-rvv
llvm.call @llvm.riscv.*
    ↓ mlir-translate + llc
RISC-V Assembly
```

## Performance Guidelines

1. **Minimize vsetvl**: Set once, reuse in loops
2. **Use LMUL > 1**: For compute-bound kernels
3. **Pipeline memory access**: Overlap load and compute
4. **Handle tails properly**: Use masking

## Testing

Test with multiple VLEN values:
```bash
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./test
qemu-riscv64 -cpu rv64,v=true,vlen=256 ./test
qemu-riscv64 -cpu rv64,v=true,vlen=512 ./test
```

## Tiling Strategy for GEMM

For VLEN=256, f32 (VL=8 elements):
- Tile M: 64 (multiple of VL)
- Tile N: 64
- Tile K: 32 (for register pressure)
