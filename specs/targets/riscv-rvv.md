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

## Quantized Profile Parameters

The RVV profile exposes quantized tiles through `KS_QUANT_*` macros so lowering
passes and build scripts can derive pass options from the target profile. Do not
hardcode RVV VLEN in compiler code.

For `target/riscv_rvv_256.h`:

| Kernel path | Profile values | Rationale |
|-------------|----------------|-----------|
| INT8 dot | `KS_QUANT_DOT_I8_TILE_K = 128` | One LMUL=4 i8 vector group at the minimum 256-bit VLEN. |
| INT8 GEMV | `KS_QUANT_MATVEC_I8_TILE_ROWS = 4`, `KS_QUANT_MATVEC_I8_TILE_COLS = 128` | Batch-1 decode keeps several i32 accumulators live while streaming one K tile. |
| W4A8 dot | `KS_QUANT_DOT_W4A8_TILE_K = 64` | One default scale group and 32 packed weight bytes per tile. |
| W4A8 GEMV | `KS_QUANT_MATVEC_W4A8_TILE_ROWS = 4`, `KS_QUANT_MATVEC_W4A8_TILE_COLS = 64` | Fused unpack/dequantize works one scale group at a time. |
| Quantized GEMM | `KS_QUANT_MATMUL_*` | Separate defaults for later prefill/small-batch work; do not reuse for decode without retuning. |

W4A8 packed weights remain low-nibble first as defined by the quantization ABI.
The profile default group size is 64 weights per f32 scale. Current quantized RVV
workspace macros return `0u`; future packers must update those macros before
using caller-provided scratch memory.
