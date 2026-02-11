/*
 * KernelSmith Matrix Multiplication — Reference Implementation
 *
 * Tiled triple-loop (ikj order) with stride support.
 * Tile sizes come from the compiled target profile.
 * No packing, no SIMD — pure scalar C99 reference.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_matmul.h"

/* Target profile is force-included via -include in CMake. */

#include <stddef.h>

/* Use L1 tile sizes from the target profile. */
#define TILE_M KS_MATMUL_TILE_M_L1
#define TILE_N KS_MATMUL_TILE_N_L1
#define TILE_K KS_MATMUL_TILE_K_L1

static size_t ks_min(size_t a, size_t b) { return a < b ? a : b; }

int ks_matmul_f32(const float *A, size_t lda,
                  const float *B, size_t ldb,
                  float *C, size_t ldc,
                  size_t M, size_t N, size_t K,
                  void *workspace, size_t ws_size) {
  /* Validate inputs. */
  if (!A || !B || !C)
    return KS_ERR_INVALID_ARG;
  if (M == 0 || N == 0)
    return KS_OK; /* Nothing to compute. */
  if (lda < K || ldb < N || ldc < N)
    return KS_ERR_INVALID_ARG;

  /* Check workspace. */
  size_t ws_needed = ks_matmul_f32_workspace(M, N, K);
  if (ws_needed > 0 && (ws_size < ws_needed || !workspace))
    return KS_ERR_WORKSPACE;

  (void)workspace; /* Unused for generic target. */

  /* Zero the output matrix. */
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      C[i * ldc + j] = 0.0f;
    }
  }

  /*
   * Tiled matmul with ikj loop order.
   * ikj is cache-friendly: broadcasts A[i,k], streams through B[k,:] and C[i,:].
   */
  for (size_t mc = 0; mc < M; mc += TILE_M) {
    size_t m_end = ks_min(mc + TILE_M, M);
    for (size_t nc = 0; nc < N; nc += TILE_N) {
      size_t n_end = ks_min(nc + TILE_N, N);
      for (size_t kc = 0; kc < K; kc += TILE_K) {
        size_t k_end = ks_min(kc + TILE_K, K);

        /* Inner tile: ikj order. */
        for (size_t i = mc; i < m_end; i++) {
          for (size_t k = kc; k < k_end; k++) {
            float a_val = A[i * lda + k];
            for (size_t j = nc; j < n_end; j++) {
              C[i * ldc + j] += a_val * B[k * ldb + j];
            }
          }
        }
      }
    }
  }

  return KS_OK;
}

size_t ks_matmul_f32_workspace(size_t M, size_t N, size_t K) {
  (void)M;
  (void)N;
  (void)K;
#if KS_MATMUL_PACK_B
  size_t pack_b = K * KS_MATMUL_NR * sizeof(float);
  pack_b = (pack_b + KS_PREFERRED_ALIGN - 1) & ~((size_t)KS_PREFERRED_ALIGN - 1);
#else
  size_t pack_b = 0;
#endif
#if KS_MATMUL_PACK_A
  size_t pack_a = KS_MATMUL_MR * K * sizeof(float);
  pack_a = (pack_a + KS_PREFERRED_ALIGN - 1) & ~((size_t)KS_PREFERRED_ALIGN - 1);
#else
  size_t pack_a = 0;
#endif
  return pack_b + pack_a;
}

size_t ks_matmul_alignment(void) { return KS_PREFERRED_ALIGN; }
