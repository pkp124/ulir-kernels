/*
 * KernelSmith Target Profile: x86-64 AVX2 + FMA
 *
 * Targets x86-64 processors with AVX2 and FMA3 (Haswell and later).
 * This covers the vast majority of x86 hardware from 2013 onwards.
 *
 * Workspace: ~64KB typical (B packing buffer)
 * SIMD: 256-bit (8x f32, 16x f16, 32x i8)
 * Tiling: two-level (L2 + L1), B packing enabled
 *
 * Reference: Intel Optimization Manual, Chapter 15 (Matrix Multiply)
 */

#ifndef KS_TARGET_PROFILE_H
#define KS_TARGET_PROFILE_H

/* ── Architecture ────────────────────────────────────────────────────── */

#define KS_ARCH_GENERIC   0
#define KS_ARCH_X86_64    1
#define KS_ARCH_AARCH64   2
#define KS_ARCH_RISCV64   3
#define KS_ARCH_CORTEX_M  4

#define KS_TARGET_NAME    "x86-avx2"
#define KS_TARGET_ARCH    KS_ARCH_X86_64

/* ── Vector Unit ─────────────────────────────────────────────────────── */

#define KS_SIMD_WIDTH_BITS   256
#define KS_SIMD_WIDTH_F32    8      /* 256 / 32 */
#define KS_SIMD_WIDTH_F16    16     /* 256 / 16 (with F16C conversion) */
#define KS_SIMD_WIDTH_I8     32     /* 256 / 8 */

/* ── Cache Hierarchy ─────────────────────────────────────────────────── */

/*
 * Conservative values based on common x86 parts:
 * - Haswell/Skylake: 32KB L1D, 256KB L2
 * - Zen2/Zen3: 32KB L1D, 512KB L2
 * - Alder Lake: 48KB L1D (P-core), 1.25MB L2
 *
 * We use the smallest common denominator.
 */
#define KS_L1D_SIZE_KB       32
#define KS_L2_SIZE_KB        256
#define KS_L3_SIZE_KB        8192   /* Not used for tiling decisions */
#define KS_CACHELINE_BYTES   64

/* ── Memory Alignment ────────────────────────────────────────────────── */

#define KS_PREFERRED_ALIGN   32     /* AVX2: 32-byte aligned vmovaps */
#define KS_REQUIRED_ALIGN    1      /* Unaligned access supported */

/* ── MatMul Tile Sizes ───────────────────────────────────────────────── */

/*
 * L2 tiles (outer loop):
 *   A panel: 128 * 512 * 4 = 256KB  (fits in L2)
 *   B panel: 512 * 48  * 4 =  96KB  (packed, accessed sequentially)
 */
#define KS_MATMUL_TILE_M_L2   128
#define KS_MATMUL_TILE_N_L2   48     /* Multiple of NR (48 = 3 * 16) */
#define KS_MATMUL_TILE_K_L2   512

/*
 * L1 tiles (inner loop):
 *   A block: 6  * 512 * 4 =  12KB  (MR rows of A panel)
 *   C block: 6  * 16  * 4 =  384B  (in registers)
 *
 * The L1 tile effectively IS the micro-kernel for this profile:
 * we tile M by MR and N by NR within the L2 tile.
 */
#define KS_MATMUL_TILE_M_L1   6     /* == MR for this profile */
#define KS_MATMUL_TILE_N_L1   16    /* == NR for this profile */
#define KS_MATMUL_TILE_K_L1   512   /* Full K panel (same as L2 K) */

/*
 * Register tile / micro-kernel:
 *   6 rows x 16 cols = 96 f32 values
 *   16 cols / 8 per YMM = 2 YMM registers per row
 *   6 rows * 2 = 12 YMM accumulators
 *   + 1 YMM for A broadcast + 2 YMM for B loads = 15 YMM total
 *   (16 YMM registers available in AVX2)
 */
#define KS_MATMUL_MR          6
#define KS_MATMUL_NR          16

/* ── Packing ─────────────────────────────────────────────────────────── */

/*
 * Pack B: rearrange B[K,N] into tile-contiguous panels of width NR.
 * This eliminates TLB misses and enables sequential cache line access.
 *
 * Pack workspace: KS_MATMUL_TILE_K_L2 * KS_MATMUL_NR * 4 = 512 * 16 * 4 = 32KB
 *
 * Pack A: not needed at this tile size — A rows are accessed sequentially
 * and MR is small enough to stay in L1.
 */
#define KS_MATMUL_PACK_B      1
#define KS_MATMUL_PACK_A      0

/* ── Feature Flags ───────────────────────────────────────────────────── */

#define KS_HAS_FMA            1     /* FMA3: vfmadd231ps */
#define KS_HAS_F16C           1     /* vcvtph2ps / vcvtps2ph */
#define KS_HAS_VNNI           0     /* Not in AVX2 baseline */
#define KS_HAS_AVX512         0
#define KS_HAS_SVE            0
#define KS_HAS_DOTPROD        0
#define KS_HAS_RVV            0

/* ── Optimization Flags ──────────────────────────────────────────────── */

/*
 * Double buffering: not critical on x86 because the hardware prefetcher
 * handles sequential access patterns well. Software prefetch hints
 * (_mm_prefetch) are more effective here.
 */
#define KS_MATMUL_DOUBLE_BUFFER  0

/* ── Element-wise Operations ─────────────────────────────────────────── */

/*
 * Process 64 f32 elements per iteration = 8 YMM loads.
 * This amortizes loop overhead and enables instruction-level parallelism.
 */
#define KS_ELEMENTWISE_TILE   64

#endif /* KS_TARGET_PROFILE_H */
