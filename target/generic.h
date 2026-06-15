/*
 * KernelSmith Target Profile: Generic (Portable C)
 *
 * Produces a portable C library with no SIMD or platform assumptions.
 * Works on any target with a C99 compiler. Performance is not optimized —
 * this profile exists for correctness validation and as a fallback.
 *
 * Workspace: 0 bytes (no packing)
 * SIMD: none (scalar)
 * Tiling: single-level, conservative sizes
 */

#ifndef KS_TARGET_PROFILE_H
#define KS_TARGET_PROFILE_H

/* ── Architecture ────────────────────────────────────────────────────── */

#define KS_ARCH_GENERIC   0
#define KS_ARCH_X86_64    1
#define KS_ARCH_AARCH64   2
#define KS_ARCH_RISCV64   3
#define KS_ARCH_CORTEX_M  4

#define KS_TARGET_NAME    "generic"
#define KS_TARGET_ARCH    KS_ARCH_GENERIC

/* ── Vector Unit ─────────────────────────────────────────────────────── */

#define KS_SIMD_WIDTH_BITS   0      /* No vector unit */
#define KS_SIMD_WIDTH_F32    1      /* Scalar */
#define KS_SIMD_WIDTH_F16    1
#define KS_SIMD_WIDTH_I8     1

/* ── Cache Hierarchy ─────────────────────────────────────────────────── */

#define KS_L1D_SIZE_KB       16     /* Assume small cache */
#define KS_L2_SIZE_KB        0      /* No L2 assumption */
#define KS_L3_SIZE_KB        0
#define KS_CACHELINE_BYTES   64

/* ── Memory Alignment ────────────────────────────────────────────────── */

#define KS_PREFERRED_ALIGN   8      /* Natural alignment for double/int64 */
#define KS_REQUIRED_ALIGN    1      /* No hard alignment requirement */

/* ── MatMul Tile Sizes ───────────────────────────────────────────────── */

/*
 * Single-level tiling only (no L2-level outer loop).
 * Tiles chosen so that one A-tile + one B-tile fits in 16KB L1:
 *   64 * 32 * 4 = 8KB  (A tile, f32)
 *   32 * 64 * 4 = 8KB  (B tile, f32)
 *   Total = 16KB
 */
#define KS_MATMUL_TILE_M_L2   0     /* 0 = skip L2 tiling */
#define KS_MATMUL_TILE_N_L2   0
#define KS_MATMUL_TILE_K_L2   0

#define KS_MATMUL_TILE_M_L1   64
#define KS_MATMUL_TILE_N_L1   64
#define KS_MATMUL_TILE_K_L1   32

#define KS_MATMUL_MR          4     /* Scalar: 4x4 micro-kernel */
#define KS_MATMUL_NR          4

/* ── Packing ─────────────────────────────────────────────────────────── */

#define KS_MATMUL_PACK_B      0     /* No packing — workspace = 0 */
#define KS_MATMUL_PACK_A      0

/* ── Feature Flags ───────────────────────────────────────────────────── */

#define KS_HAS_FMA            0
#define KS_HAS_F16C           0
#define KS_HAS_VNNI           0
#define KS_HAS_AVX512         0
#define KS_HAS_SVE            0
#define KS_HAS_DOTPROD        0
#define KS_HAS_RVV            0

/* ── Optimization Flags ──────────────────────────────────────────────── */

#define KS_MATMUL_DOUBLE_BUFFER  0

/* ── Element-wise Operations ─────────────────────────────────────────── */

#define KS_ELEMENTWISE_TILE   64    /* Elements per loop iteration */

/* ── Quantized Workspace ─────────────────────────────────────────────── */

#define KS_QUANT_DOT_I8_WORKSPACE_BYTES(K) \
  0u
#define KS_QUANT_MATVEC_I8_WORKSPACE_BYTES(ROWS, COLS) \
  0u
#define KS_QUANT_DOT_W4A8_WORKSPACE_BYTES(K, GROUP_SIZE) \
  0u
#define KS_QUANT_MATVEC_W4A8_WORKSPACE_BYTES(ROWS, COLS, GROUP_SIZE) \
  0u

#endif /* KS_TARGET_PROFILE_H */
