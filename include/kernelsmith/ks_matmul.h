/*
 * KernelSmith Matrix Multiplication
 *
 * General matrix multiplication: C = A * B
 * All matrices are row-major. Pure C99 API.
 *
 * Thread safety: All functions are reentrant. No global state.
 * Multiple threads may call concurrently with independent buffers.
 *
 * Determinism: Results are bitwise identical across calls with the
 * same inputs, dimensions, and workspace. Reduction order is fixed.
 */

#ifndef KERNELSMITH_MATMUL_H
#define KERNELSMITH_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_matmul_f32 - Single-precision matrix multiplication: C = A * B
 *
 * Computes C[M,N] = A[M,K] * B[K,N]. All matrices are row-major.
 * C must not alias A or B.
 *
 * Parameters:
 *   A           - Input matrix, M rows x K cols, row-major
 *   lda         - Leading dimension of A (stride between rows, >= K)
 *   B           - Input matrix, K rows x N cols, row-major
 *   ldb         - Leading dimension of B (stride between rows, >= N)
 *   C           - Output matrix, M rows x N cols, row-major (preallocated)
 *   ldc         - Leading dimension of C (stride between rows, >= N)
 *   M           - Number of rows in A and C
 *   N           - Number of columns in B and C
 *   K           - Shared dimension (cols of A, rows of B)
 *   workspace   - Scratch buffer (may be NULL if ws_size == 0)
 *   ws_size     - Size of workspace in bytes (>= ks_matmul_f32_workspace())
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer or invalid dimensions
 *   KS_ERR_WORKSPACE   - Workspace too small
 */
int ks_matmul_f32(const float *A, size_t lda,
                  const float *B, size_t ldb,
                  float *C, size_t ldc,
                  size_t M, size_t N, size_t K,
                  void *workspace, size_t ws_size);

/*
 * ks_matmul_f32_workspace - Workspace query for f32 matmul.
 *
 * Returns the minimum workspace size in bytes needed for the given
 * dimensions. Returns 0 if the kernel needs no workspace (e.g. generic
 * target with no packing).
 *
 * This is a pure function of (M, N, K) and the compiled target profile.
 */
size_t ks_matmul_f32_workspace(size_t M, size_t N, size_t K);

/*
 * ks_matmul_alignment - Preferred memory alignment in bytes.
 *
 * For best performance, input and output pointers should be aligned to
 * this value. Unaligned pointers are supported but may be slower.
 */
size_t ks_matmul_alignment(void);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_MATMUL_H */
