/*
 * Smoke tests for ks_matmul_f32.
 *
 * Validates: square, rectangular, non-aligned, strided, error cases.
 * Returns 0 on success, 1 on any failure.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_matmul.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_REL_ERR 1e-5f

static int tests_run = 0;
static int tests_passed = 0;

static void fill_sequential(float *buf, size_t n) {
  for (size_t i = 0; i < n; i++)
    buf[i] = (float)(i % 7) * 0.1f + 0.1f;
}

/* Naive matmul reference for checking. */
static void ref_matmul(const float *A, size_t lda, const float *B, size_t ldb,
                       float *C, size_t ldc, size_t M, size_t N, size_t K) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float sum = 0.0f;
      for (size_t k = 0; k < K; k++)
        sum += A[i * lda + k] * B[k * ldb + j];
      C[i * ldc + j] = sum;
    }
  }
}

static int check_close(const float *got, const float *ref, size_t ldc,
                        size_t M, size_t N, const char *label) {
  for (size_t i = 0; i < M; i++) {
    for (size_t j = 0; j < N; j++) {
      float g = got[i * ldc + j];
      float r = ref[i * N + j];
      float diff = fabsf(g - r);
      float denom = fabsf(r) > 1e-8f ? fabsf(r) : 1.0f;
      if (diff / denom > MAX_REL_ERR) {
        fprintf(stderr, "  FAIL %s: C[%zu,%zu] = %f, expected %f (relerr=%e)\n",
                label, i, j, g, r, (double)(diff / denom));
        return 0;
      }
    }
  }
  return 1;
}

static void run_matmul_test(size_t M, size_t N, size_t K, const char *label) {
  tests_run++;
  float *A = (float *)malloc(M * K * sizeof(float));
  float *B = (float *)malloc(K * N * sizeof(float));
  float *C = (float *)calloc(M * N, sizeof(float));
  float *C_ref = (float *)calloc(M * N, sizeof(float));

  fill_sequential(A, M * K);
  fill_sequential(B, K * N);

  int rc = ks_matmul_f32(A, K, B, N, C, N, M, N, K, NULL, 0);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL %s: ks_matmul_f32 returned %d\n", label, rc);
    goto cleanup;
  }

  ref_matmul(A, K, B, N, C_ref, N, M, N, K);

  if (check_close(C, C_ref, N, M, N, label)) {
    printf("  PASS %s\n", label);
    tests_passed++;
  }

cleanup:
  free(A);
  free(B);
  free(C);
  free(C_ref);
}

static void test_square(void) {
  run_matmul_test(64, 64, 64, "square 64x64x64");
}

static void test_rectangular(void) {
  run_matmul_test(32, 128, 64, "rect 32x128x64");
}

static void test_non_aligned(void) {
  run_matmul_test(17, 23, 31, "non-aligned 17x23x31");
}

static void test_small(void) {
  run_matmul_test(1, 1, 1, "small 1x1x1");
  run_matmul_test(4, 4, 4, "small 4x4x4");
}

static void test_stride(void) {
  tests_run++;

  /* Embed a 4x3 submatrix in a larger 8x8 buffer (lda=8). */
  size_t M = 4, N = 5, K = 3;
  size_t lda = 8, ldb = 8, ldc = 8;

  float *A = (float *)calloc(M * lda, sizeof(float));
  float *B = (float *)calloc(K * ldb, sizeof(float));
  float *C = (float *)calloc(M * ldc, sizeof(float));
  float *C_ref = (float *)calloc(M * N, sizeof(float));

  /* Fill only the used submatrix. */
  for (size_t i = 0; i < M; i++)
    for (size_t k = 0; k < K; k++)
      A[i * lda + k] = (float)((i * K + k) % 5) * 0.2f + 0.1f;
  for (size_t k = 0; k < K; k++)
    for (size_t j = 0; j < N; j++)
      B[k * ldb + j] = (float)((k * N + j) % 7) * 0.15f + 0.05f;

  int rc = ks_matmul_f32(A, lda, B, ldb, C, ldc, M, N, K, NULL, 0);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL stride: ks_matmul_f32 returned %d\n", rc);
    goto cleanup;
  }

  ref_matmul(A, lda, B, ldb, C_ref, N, M, N, K);

  if (check_close(C, C_ref, ldc, M, N, "stride 4x5x3 (ld=8)")) {
    printf("  PASS stride 4x5x3 (ld=8)\n");
    tests_passed++;
  }

cleanup:
  free(A);
  free(B);
  free(C);
  free(C_ref);
}

static void test_workspace_query(void) {
  tests_run++;
  size_t ws = ks_matmul_f32_workspace(64, 64, 64);
  /* Generic target: workspace should be 0. */
  if (ws == 0) {
    printf("  PASS workspace query returns 0 (generic)\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL workspace query: expected 0, got %zu\n", ws);
  }
}

static void test_alignment_query(void) {
  tests_run++;
  size_t align = ks_matmul_alignment();
  if (align > 0) {
    printf("  PASS alignment query returns %zu\n", align);
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL alignment query: returned 0\n");
  }
}

static void test_error_null(void) {
  tests_run++;
  float dummy = 0.0f;
  int rc = ks_matmul_f32(NULL, 1, &dummy, 1, &dummy, 1, 1, 1, 1, NULL, 0);
  if (rc == KS_ERR_INVALID_ARG) {
    printf("  PASS error on NULL A\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL error on NULL A: returned %d\n", rc);
  }
}

static void test_error_stride(void) {
  tests_run++;
  float A[4], B[4], C[4];
  /* lda < K should be rejected. */
  int rc = ks_matmul_f32(A, 1, B, 2, C, 2, 2, 2, 2, NULL, 0);
  if (rc == KS_ERR_INVALID_ARG) {
    printf("  PASS error on lda < K\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL error on lda < K: returned %d\n", rc);
  }
}

static void test_zero_dims(void) {
  tests_run++;
  float A[1], B[1], C[1];
  int rc = ks_matmul_f32(A, 1, B, 1, C, 1, 0, 1, 1, NULL, 0);
  if (rc == KS_OK) {
    printf("  PASS zero-dim M=0 returns OK\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL zero-dim M=0: returned %d\n", rc);
  }
}

int main(void) {
  printf("=== ks_matmul_f32 smoke tests ===\n");
  printf("Target: %s (SIMD width f32 = %zu)\n\n",
         ks_target_name(), ks_simd_width_f32());

  test_square();
  test_rectangular();
  test_non_aligned();
  test_small();
  test_stride();
  test_workspace_query();
  test_alignment_query();
  test_error_null();
  test_error_stride();
  test_zero_dims();

  printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
