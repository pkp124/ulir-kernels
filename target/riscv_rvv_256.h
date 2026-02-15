/*
 * KernelSmith Target Profile: RISC-V RVV (VLEN=256)
 *
 * Targets RISC-V 64-bit processors with Vector Extension 1.0 and
 * VLEN >= 256 bits (e.g., SiFive P670/P870, SpacemiT K1).
 *
 * The code generated from this profile is Vector Length Agnostic (VLA) —
 * it will run correctly on any VLEN >= 256, but tile sizes are tuned for
 * VLEN=256. Wider implementations (512+) will utilize extra vector
 * bandwidth automatically through vsetvl.
 *
 * Workspace: ~32KB typical (B packing buffer)
 * SIMD: 256-bit RVV (8x f32, 16x f16, 32x i8 at LMUL=1)
 * Tiling: two-level (L2 + L1), B packing enabled
 *
 * Reference: RISC-V "V" Vector Extension v1.0
 *            DES-001 (Vector Operations Lowering to RVV)
 */

#ifndef KS_TARGET_PROFILE_H
#define KS_TARGET_PROFILE_H

/* ── Architecture ────────────────────────────────────────────────────── */

#define KS_ARCH_GENERIC   0
#define KS_ARCH_X86_64    1
#define KS_ARCH_AARCH64   2
#define KS_ARCH_RISCV64   3
#define KS_ARCH_CORTEX_M  4

#define KS_TARGET_NAME    "riscv-rvv-256"
#define KS_TARGET_ARCH    KS_ARCH_RISCV64

/* ── Vector Unit ─────────────────────────────────────────────────────── */

/*
 * RVV VLEN=256 at LMUL=1:
 *   f32: 256 / 32 = 8 elements
 *   f16: 256 / 16 = 16 elements
 *   i8:  256 / 8  = 32 elements
 *
 * At LMUL=2 (default for compute-bound kernels):
 *   f32: 16 elements per register group
 *   f16: 32 elements per register group
 *
 * KS_VECTOR_IS_SCALABLE indicates VLA semantics — the actual vector
 * length is determined at runtime via vsetvli. Tile sizes below assume
 * VLEN >= KS_RVV_VLEN_MIN but will work for larger VLENs.
 */
#define KS_SIMD_WIDTH_BITS   256
#define KS_SIMD_WIDTH_F32    8      /* 256 / 32, LMUL=1 */
#define KS_SIMD_WIDTH_F16    16     /* 256 / 16, LMUL=1 */
#define KS_SIMD_WIDTH_I8     32     /* 256 / 8,  LMUL=1 */
#define KS_VECTOR_IS_SCALABLE 1     /* VLA: VLEN may be larger at runtime */
#define KS_RVV_VLEN_MIN      256    /* Minimum VLEN this profile targets */
#define KS_RVV_LMUL_DEFAULT  2      /* LMUL for compute-bound micro-kernels */

/* ── Cache Hierarchy ─────────────────────────────────────────────────── */

/*
 * Typical dual-issue rv64gcv core (SiFive P670-class):
 *   L1D: 32KB, 4-way
 *   L2:  256KB-512KB per core cluster
 *   L3:  varies (shared, or absent on simpler cores)
 *
 * Conservative values for broad RVV hardware compatibility.
 */
#define KS_L1D_SIZE_KB       32
#define KS_L2_SIZE_KB        256
#define KS_L3_SIZE_KB        0       /* Not assumed — many RV64 cores lack L3 */
#define KS_CACHELINE_BYTES   64

/* ── Memory Alignment ────────────────────────────────────────────────── */

/*
 * RVV supports unaligned vector loads/stores (vle/vse are alignment-
 * agnostic), but aligned access avoids crossing cache line boundaries
 * and is faster on most implementations.
 *
 * Preferred alignment = VLEN/8 = 32 bytes.
 */
#define KS_PREFERRED_ALIGN   32     /* VLEN/8 = 32 bytes */
#define KS_REQUIRED_ALIGN    1      /* RVV handles unaligned access */

/* ── MatMul Tile Sizes ───────────────────────────────────────────────── */

/*
 * L2 tiles (outer loop):
 *   A panel: 64 * 256 * 4 = 64KB   (fits in L2/2 = 128KB)
 *   B panel: 256 * 64 * 4 = 64KB   (packed into workspace)
 *   Total L2 working set ~128KB, fits in 256KB L2
 *
 * Tiles are multiples of VL (8 for f32 at LMUL=1) for clean vector
 * loop bounds.
 */
#define KS_MATMUL_TILE_M_L2   64
#define KS_MATMUL_TILE_N_L2   64    /* Multiple of NR */
#define KS_MATMUL_TILE_K_L2   256

/*
 * L1 tiles (inner loop):
 *   A block: 4 * 256 * 4 = 4KB     (MR rows, fits in L1)
 *   C block: 4 * 16  * 4 = 256B    (in vector registers)
 *
 * The L1 N-tile equals NR: micro-kernel processes one NR-wide panel
 * at a time. K is the full L2 K-tile (no L1 K-tiling needed).
 */
#define KS_MATMUL_TILE_M_L1   4     /* == MR */
#define KS_MATMUL_TILE_N_L1   16    /* == NR */
#define KS_MATMUL_TILE_K_L1   256   /* Full K panel (same as L2 K) */

/*
 * Register tile / micro-kernel:
 *
 * With LMUL=2, each register group holds 16 f32 values (2 × v-regs).
 * MR=4, NR=16: 4 × 16 = 64 accumulator values in 4 register groups.
 *
 * Register budget (32 v-regs at LMUL=2 = 16 groups):
 *   4 groups: C accumulators (4 rows × 1 NR-wide group each)
 *   1 group:  A element broadcast
 *   1 group:  B panel load
 *   ─────────
 *   6 groups used, 10 groups free for software pipelining / prefetch
 *
 * This is conservative — leaves headroom for the compiler to schedule
 * loads and stores without register spills.
 */
#define KS_MATMUL_MR          4
#define KS_MATMUL_NR          16    /* VLMAX at LMUL=2, SEW=32 */

/* ── Packing ─────────────────────────────────────────────────────────── */

/*
 * Pack B: rearrange B[K,N] into NR-wide contiguous panels.
 * Critical for RVV: strided loads on in-order cores are expensive,
 * and RVV does not have register-rename depth to hide latency.
 *
 * Pack workspace: KS_MATMUL_TILE_K_L2 * KS_MATMUL_NR * 4
 *               = 256 * 16 * 4 = 16KB
 *
 * Pack A: not needed — MR=4 rows are accessed sequentially and fit
 * in L1 cache. A packing would add overhead without measurable benefit
 * at this tile size.
 */
#define KS_MATMUL_PACK_B      1
#define KS_MATMUL_PACK_A      0

/* ── Feature Flags ───────────────────────────────────────────────────── */

#define KS_HAS_FMA            1     /* vfmacc.vv — fused multiply-add */
#define KS_HAS_F16C           1     /* vfwcvt.f.f.v (f16→f32 widening) */
#define KS_HAS_VNNI           0     /* No vector dot-product extension yet */
#define KS_HAS_AVX512         0
#define KS_HAS_SVE            0
#define KS_HAS_DOTPROD        0
#define KS_HAS_RVV            1

/* ── Optimization Flags ──────────────────────────────────────────────── */

/*
 * Double buffering: pack next B tile while computing current one.
 * Potentially high benefit on in-order RV64 cores (30-50% speedup
 * per DES-006 §8). Deferred to Milestone 8 — requires doubling the
 * pack workspace and a dedicated optimization pass.
 */
#define KS_MATMUL_DOUBLE_BUFFER  0

/* ── Element-wise Operations ─────────────────────────────────────────── */

/*
 * Process 64 f32 elements per iteration at LMUL=1 = 8 vector loads.
 * At LMUL=2 this becomes 4 register-group loads — good ILP on
 * dual-issue cores.
 */
#define KS_ELEMENTWISE_TILE   64

#endif /* KS_TARGET_PROFILE_H */
