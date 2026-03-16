// KernelSmith — RISC-V RVV Target Profile (VLEN=256 baseline)
//
// This header defines compile-time constants for the RISC-V Vector Extension
// target profile. It is consumed by the pass pipeline (via ks-opt options)
// and by the C kernel library build system.
//
// Primary target: RISC-V RVV (Vector Length Agnostic, VLEN=256 baseline)
// Example SoCs: T-Head C908/C910, SiFive X280, Kendryte K230
//
// Vector parameters (VLEN=256, f32, LMUL=4):
//   VLMAX = VLEN * LMUL / SEW = 256 * 4 / 32 = 32 elements
//   MR (register tile rows)  = 16
//   NR (register tile cols)  = 32
//
// Two-level tiling strategy:
//   L2 tile  (fits in L2 cache, ~512KB):  M=128, N=128, K=256
//   Reg tile (fits in vector registers):  M=16,  N=32,  K=1 (inner via SIMD)
//
// Usage in pass pipeline:
//   ks-opt input.mlir \
//     --ks-lower-to-linalg \
//     "--ks-tile=tile-size-m=128 tile-size-n=128 tile-size-k=256" \
//     --ks-pack \
//     "--ks-tile=tile-size-m=16 tile-size-n=32 tile-size-k=0" \
//     --ks-vectorize \
//     --ks-lower-to-rvv \
//     -o output.mlir

#ifndef KS_TARGET_RISCV_RVV_256_H
#define KS_TARGET_RISCV_RVV_256_H

// ===--- Architecture ID -------------------------------------------------===//

#define KS_TARGET_ARCH        "riscv64"
#define KS_TARGET_ABI         "lp64d"
#define KS_TARGET_MARCH       "rv64gcv"
#define KS_TARGET_FEATURES    "+v,+zve64d,+zvl256b"
#define KS_TARGET_TRIPLE      "riscv64-unknown-linux-gnu"

// ===--- Vector parameters -----------------------------------------------===//

// Baseline VLEN in bits (physical minimum guaranteed by the target SoC).
// VLA code still works at larger VLEN — this is the minimum for tile sizing.
#define KS_RVV_VLEN           256

// Selected Element Width for f32 workloads (bits).
#define KS_RVV_SEW_F32        32

// LMUL: use LMUL=4 for compute-bound matmul (maximises VLMAX).
// For memory-bound kernels (activations, norms) use LMUL=1.
#define KS_RVV_LMUL_COMPUTE   4
#define KS_RVV_LMUL_MEMORY    1

// VLMAX = VLEN * LMUL / SEW  (elements per vector register group).
#define KS_RVV_VLMAX_F32_LMUL4  ((KS_RVV_VLEN * KS_RVV_LMUL_COMPUTE) / KS_RVV_SEW_F32)  // 32
#define KS_RVV_VLMAX_F32_LMUL1  ((KS_RVV_VLEN * KS_RVV_LMUL_MEMORY)  / KS_RVV_SEW_F32)  // 8

// ===--- Register tile (MR × NR) -----------------------------------------===//
// Inner micro-kernel that fits entirely in vector registers.
// NR is a multiple of VLMAX_LMUL4 (32) for alignment.
#define KS_RVV_TILE_MR        16   // rows of C micro-tile
#define KS_RVV_TILE_NR        32   // cols of C micro-tile (= VLMAX_LMUL4)
#define KS_RVV_TILE_KR         1   // reduction unroll (handled by SIMD width)

// ===--- L2 cache tile (L2T_M × L2T_N × L2T_K) --------------------------===//
// Outer tile sized to fit A-panel + B-panel + C-tile in L2 (~512 KB typical).
// A-panel: L2T_M × L2T_K × 4 bytes = 128 × 256 × 4 = 128 KB
// B-panel: L2T_K × L2T_N × 4 bytes = 256 × 128 × 4 = 128 KB
// C-tile : L2T_M × L2T_N × 4 bytes = 128 × 128 × 4 =  64 KB
//          Total: 320 KB — fits in 512 KB L2.
#define KS_RVV_TILE_L2_M      128
#define KS_RVV_TILE_L2_N      128
#define KS_RVV_TILE_L2_K      256

// ===--- Memory / cache parameters ----------------------------------------===//
#define KS_L1D_SIZE_BYTES     (32 * 1024)    // 32 KB L1 data cache
#define KS_L2_SIZE_BYTES      (512 * 1024)   // 512 KB L2 unified cache
#define KS_CACHELINE_BYTES    64

// ===--- Pack (B operand layout) ------------------------------------------===//
// B is packed into column-major panels of width NR to enable stride-free
// vector loads in the inner loop. Panel shape: [K/KR, NR, KR] → [K, NR].
#define KS_RVV_PACK_FACTOR    KS_RVV_TILE_NR   // = NR = 32 elements
#define KS_RVV_PACK_ALIGN     64               // bytes (cacheline aligned)

// ===--- Workspace size query macro ---------------------------------------===//
// Returns the number of bytes needed to pack a K×N B-matrix.
// Callers should round K and N up to pack-factor multiples.
#define KS_RVV_PACK_WORKSPACE_BYTES(K, N) \
  (((size_t)(K)) * ((size_t)(N)) * 4u)

// ===--- Derived ks-opt flags ---------------------------------------------===//
// Convenience string macros for building ks-opt command lines in scripts.
#define KS_RVV_TILE_L2_OPTS \
  "--ks-tile=tile-size-m=" KS_STRINGIFY(KS_RVV_TILE_L2_M) \
  " tile-size-n=" KS_STRINGIFY(KS_RVV_TILE_L2_N)          \
  " tile-size-k=" KS_STRINGIFY(KS_RVV_TILE_L2_K)

#define KS_RVV_TILE_REG_OPTS \
  "--ks-tile=tile-size-m=" KS_STRINGIFY(KS_RVV_TILE_MR) \
  " tile-size-n=" KS_STRINGIFY(KS_RVV_TILE_NR)          \
  " tile-size-k=0"

#define KS_STRINGIFY(x) KS_STRINGIFY2(x)
#define KS_STRINGIFY2(x) #x

#endif // KS_TARGET_RISCV_RVV_256_H
