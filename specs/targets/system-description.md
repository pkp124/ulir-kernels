# System Description Interface — Target Profile Specification

## Overview

A target profile is a static description of the hardware that a KernelSmith library is
compiled for. One profile produces one library. The profile drives every target-dependent
decision: tile sizes, vector widths, packing strategy, alignment, and SIMD feature flags.

## Design Principles

1. **Static and build-time only.** No runtime detection, no cpuid, no /proc.
2. **C preprocessor format.** Plain `#define` values in a header file.
3. **Overridable.** Users can `-DKS_MATMUL_TILE_M_L1=16` to override any value.
4. **Freestanding.** No includes beyond stdint.h semantics. Works on bare metal.
5. **One profile, one library.** Cross-compile for each target separately.

## Profile Header Structure

Every profile header defines the following sections. All values are required unless
marked optional.

### Architecture ID

```c
#define KS_TARGET_NAME    "human-readable-name"   /* e.g. "x86-avx2" */
#define KS_TARGET_ARCH    KS_ARCH_xxx             /* architecture enum */
```

Valid `KS_TARGET_ARCH` values:

| Value | Meaning |
|-------|---------|
| `KS_ARCH_GENERIC` | Portable C, no SIMD assumptions |
| `KS_ARCH_X86_64` | x86-64 (AVX2, AVX-512, etc.) |
| `KS_ARCH_AARCH64` | ARMv8-A (NEON, SVE) |
| `KS_ARCH_RISCV64` | RISC-V 64-bit (RVV) |
| `KS_ARCH_CORTEX_M` | ARM Cortex-M (no vector unit) |

### SIMD / Vector Unit

```c
#define KS_SIMD_WIDTH_BITS   256   /* Total vector register width in bits */
#define KS_SIMD_WIDTH_F32    8     /* Elements per vector register (32-bit float) */
#define KS_SIMD_WIDTH_F16    16    /* Elements per vector register (16-bit float) */
#define KS_SIMD_WIDTH_I8     32    /* Elements per vector register (8-bit int) */
```

For `KS_ARCH_GENERIC` and `KS_ARCH_CORTEX_M`, set all to 1 (scalar).

For RVV (vector-length agnostic), set to the *minimum guaranteed* VLEN divided by
element width. The kernel will work on wider implementations via `vsetvl`.

### Cache Hierarchy

```c
#define KS_L1D_SIZE_KB       32    /* L1 data cache in KiB */
#define KS_L2_SIZE_KB        256   /* L2 cache in KiB (0 if none) */
#define KS_L3_SIZE_KB        8192  /* L3 cache in KiB (0 if none) */
#define KS_CACHELINE_BYTES   64    /* Cache line size in bytes */
```

For MCUs with only scratchpad/TCM, set `KS_L1D_SIZE_KB` to the TCM size and
`KS_L2_SIZE_KB = 0`.

### Memory Alignment

```c
#define KS_PREFERRED_ALIGN   32    /* Optimal alignment for vector loads (bytes) */
#define KS_REQUIRED_ALIGN    1     /* Minimum required alignment (1 = any) */
```

`KS_PREFERRED_ALIGN` is documented to users as the recommended pointer alignment.
`KS_REQUIRED_ALIGN` is a hard requirement — pointers below this alignment cause
undefined behavior (relevant for architectures without unaligned access support).

### MatMul Tile Sizes

```c
/* L2-level tiles (outer tiling loop) */
#define KS_MATMUL_TILE_M_L2   128
#define KS_MATMUL_TILE_N_L2   256
#define KS_MATMUL_TILE_K_L2   512

/* L1-level tiles (inner tiling loop) */
#define KS_MATMUL_TILE_M_L1   32
#define KS_MATMUL_TILE_N_L1   64
#define KS_MATMUL_TILE_K_L1   128

/* Register tile / micro-kernel dimensions */
#define KS_MATMUL_MR          6    /* Rows per micro-kernel invocation */
#define KS_MATMUL_NR          16   /* Cols per micro-kernel invocation */
```

#### Tile Size Constraints

The following invariants must hold:

```
L2 tiles:
  KS_MATMUL_TILE_M_L2 * KS_MATMUL_TILE_K_L2 * 4  <=  KS_L2_SIZE_KB * 1024 / 2

L1 tiles:
  KS_MATMUL_TILE_M_L1 * KS_MATMUL_TILE_N_L1 * 4  <=  KS_L1D_SIZE_KB * 1024 / 2

Register tile:
  KS_MATMUL_MR * KS_MATMUL_NR  <=  available_vector_regs * KS_SIMD_WIDTH_F32
  KS_MATMUL_NR must be a multiple of KS_SIMD_WIDTH_F32
```

(The `* 4` assumes f32; for other types scale by element size.)

For `KS_ARCH_GENERIC`, set L2/L1 tiles to modest values (e.g., 64/32) and
MR=NR=4. Performance is not the goal; correctness is.

### Packing Flags

```c
#define KS_MATMUL_PACK_B       1   /* 1 = pack B operand for cache locality */
#define KS_MATMUL_PACK_A       0   /* 1 = also pack A operand */
```

Packing consumes workspace memory. When both are 0, workspace for matmul is 0 bytes.

### Quantized Dot/GEMV/GEMM Parameters

Profiles that support optimized quantized kernels define the following build-time
values. The `KS_QUANT_*` macros are the stable interface for scripts and pass
option generation; target-specific helper macros such as `KS_RVV_*` may derive
them from VLEN, SEW, LMUL, cache, or ABI choices.

#### INT8 dot and GEMV

```c
#define KS_QUANT_DOT_I8_TILE_K          128  /* K elements per dot tile */
#define KS_QUANT_MATVEC_I8_TILE_ROWS    4    /* output rows per GEMV tile */
#define KS_QUANT_MATVEC_I8_TILE_COLS    128  /* K cols per GEMV tile */
#define KS_QUANT_I8_PACK_FACTOR         128  /* i8 elements per pack group */
#define KS_QUANT_I8_PACK_ALIGN          64   /* packed i8 alignment in bytes */
```

`KS_QUANT_DOT_I8_TILE_K` and `KS_QUANT_MATVEC_I8_TILE_COLS` should be multiples
of the profile's i8 vector width. On RVV, derive them from minimum VLEN, SEW,
and LMUL instead of hardcoding a physical vector length in compiler passes.

#### W4A8 dot and GEMV

```c
#define KS_QUANT_DOT_W4A8_TILE_K        64  /* logical weights per dot tile */
#define KS_QUANT_MATVEC_W4A8_TILE_ROWS  4   /* output rows per GEMV tile */
#define KS_QUANT_MATVEC_W4A8_TILE_COLS  64  /* K cols per GEMV tile */
#define KS_QUANT_W4A8_GROUP_SIZE        64  /* default weights per scale */
#define KS_QUANT_W4A8_PACK_FACTOR       64  /* logical weights per pack group */
#define KS_QUANT_W4A8_PACKED_TILE_BYTES 32  /* packed bytes per W4A8 tile */
#define KS_QUANT_W4A8_PACK_ALIGN        64  /* packed weight alignment */
#define KS_QUANT_W4A8_SCALE_ALIGN       4   /* f32 scale alignment */
```

W4A8 dot/GEMV values are intentionally independent from GEMM values because
batch-1 transformer decode is the first optimized path. The default group size
must match the public C ABI's scale layout for generated kernels that assume a
profile default; callers may still pass explicit group sizes to generic APIs.

#### Quantized GEMM defaults

```c
#define KS_QUANT_MATMUL_I8_TILE_M_L2     128
#define KS_QUANT_MATMUL_I8_TILE_N_L2     128
#define KS_QUANT_MATMUL_I8_TILE_K_L2     512
#define KS_QUANT_MATMUL_I8_TILE_M_L1     8
#define KS_QUANT_MATMUL_I8_TILE_N_L1     16
#define KS_QUANT_MATMUL_I8_TILE_K_L1     128
#define KS_QUANT_MATMUL_W4A8_TILE_M_L2   64
#define KS_QUANT_MATMUL_W4A8_TILE_N_L2   128
#define KS_QUANT_MATMUL_W4A8_TILE_K_L2   256
#define KS_QUANT_MATMUL_W4A8_TILE_M_L1   4
#define KS_QUANT_MATMUL_W4A8_TILE_N_L1   16
#define KS_QUANT_MATMUL_W4A8_TILE_K_L1   64
```

GEMM parameters are profile defaults for later prefill and small-batch kernels.
They must not be reused blindly for dot/GEMV lowering because decode kernels have
different register-pressure and memory-bandwidth constraints.

#### Quantized workspace declarations

```c
#define KS_QUANT_DOT_I8_WORKSPACE_BYTES(K) 0u
#define KS_QUANT_MATVEC_I8_WORKSPACE_BYTES(ROWS, COLS) 0u
#define KS_QUANT_DOT_W4A8_WORKSPACE_BYTES(K, GROUP_SIZE) 0u
#define KS_QUANT_MATVEC_W4A8_WORKSPACE_BYTES(ROWS, COLS, GROUP_SIZE) 0u
```

Workspace macros make scratch-buffer requirements explicit at build time. A
profile that introduces a quantized packer must update these before generated
kernels start consuming caller-provided workspace.

### Feature Flags (optional, target-specific)

```c
/* x86-specific */
#define KS_HAS_FMA           1     /* FMA3 instructions available */
#define KS_HAS_F16C          1     /* f16 <-> f32 conversion */
#define KS_HAS_VNNI          0     /* VNNI int8 dot product (AVX-512) */
#define KS_HAS_AVX512        0     /* AVX-512F base */

/* ARM-specific */
#define KS_HAS_SVE           0     /* Scalable Vector Extension */
#define KS_HAS_DOTPROD       0     /* SDOT/UDOT for int8 */

/* RISC-V-specific */
#define KS_HAS_RVV           1     /* Vector extension */
#define KS_RVV_VLEN_MIN      128   /* Minimum VLEN in bits */

/* Double buffering (optimization, optional) */
#define KS_MATMUL_DOUBLE_BUFFER  0  /* 1 = overlap pack + compute */
```

Feature flags are used by the build script to select MLIR pass options and by the
codegen to gate target-specific intrinsics.

### Activation / Element-wise Tile Sizes (optional)

```c
/* For element-wise ops (relu, gelu, etc.) — how many elements per iteration */
#define KS_ELEMENTWISE_TILE  256   /* Process 256 elements per loop iteration */
```

Default: `KS_SIMD_WIDTH_F32 * 8` (8 vector loads per iteration for software pipelining).

## Profile Validation

A build-time script validates the profile before compilation:

```bash
python scripts/validate_profile.py target/x86_avx2.h
```

Checks:
- All required `#define` values present
- Tile size constraints satisfied (L2 tiles fit in L2, etc.)
- NR is a multiple of SIMD width
- PREFERRED_ALIGN is a power of 2
- PREFERRED_ALIGN >= REQUIRED_ALIGN
- RVV profiles define the required `KS_QUANT_*` i8/W4A8 tile, pack,
  alignment, and workspace macros

Validation errors are build failures, not runtime errors.

## Provided Profiles

| File | Target | SIMD | L1 | L2 | Pack B |
|------|--------|------|-----|-----|--------|
| `generic.h` | Portable C | 1 (scalar) | 16KB | 0 | No |
| `x86_avx2.h` | x86-64 AVX2+FMA | 8 (256b) | 32KB | 256KB | Yes |
| `x86_avx512.h` | x86-64 AVX-512F | 16 (512b) | 32KB | 1MB | Yes |
| `arm_neon.h` | ARMv8-A NEON | 4 (128b) | 32KB | 256KB | Yes |
| `arm_sve_256.h` | ARM SVE 256-bit | 8 (256b) | 64KB | 512KB | Yes |
| `riscv_rvv_256.h` | RISC-V RVV VLEN=256 | 8 (256b) | 16KB | 128KB | Yes |
| `riscv_rvv_512.h` | RISC-V RVV VLEN=512 | 16 (512b) | 32KB | 256KB | Yes |
| `cortex_m7.h` | ARM Cortex-M7 | 1 (scalar) | 16KB | 0 | No |

Users may create custom profiles for their specific hardware by copying the nearest
profile and adjusting cache sizes and tile parameters.

## Adding Kernel Tile Sizes to a Profile

When a new kernel is added (e.g., conv2d), new `#define` sections are added to each
profile:

```c
/* Conv2D tile sizes */
#define KS_CONV2D_TILE_OH      8
#define KS_CONV2D_TILE_OW      16
#define KS_CONV2D_TILE_OC      32
#define KS_CONV2D_TILE_IC      32
#define KS_CONV2D_PACK_FILTER  1
```

Each kernel documents its tile size parameters and constraints in its own spec.
Profiles that predate the new kernel simply omit the section — the build script
uses fallback defaults from the kernel spec.

## Relationship to MLIR Passes

The build script translates profile `#define` values into MLIR pass options:

```
KS_MATMUL_TILE_M_L2=128  -->  --ks-tile-l2="tile-m=128,tile-n=256,tile-k=512"
KS_MATMUL_PACK_B=1       -->  --ks-pack="pack-b=true"
KS_SIMD_WIDTH_F32=8      -->  --ks-vectorize="width=8"
KS_HAS_AVX512=0          -->  --convert-to-llvm="target-triple=x86_64-...,target-features=+avx2,+fma"
```

MLIR passes are target-agnostic. All target knowledge is in the profile header.
