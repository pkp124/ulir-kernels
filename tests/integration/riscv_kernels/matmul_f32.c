/*===----------------------------------------------------------------------===*
 * KernelSmith - RISC-V MatMul Reference Kernel (f32)
 *
 * Implements scalar and RVV-vectorized f32 matrix multiplication.
 * Used for integration testing: compile with riscv64-linux-gnu-gcc,
 * run on QEMU, validate against numpy reference.
 *
 * Usage: matmul_f32 <M> <N> <K> <input_A.bin> <input_B.bin> <output_C.bin>
 *   or:  matmul_f32 --self-test
 *===----------------------------------------------------------------------===*/

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*===----------------------------------------------------------------------===*
 * Scalar reference matmul: C[M,N] = A[M,K] * B[K,N]
 *===----------------------------------------------------------------------===*/
void matmul_scalar_f32(const float *A, const float *B, float *C,
                       int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k++) {
                acc += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = acc;
        }
    }
}

/*===----------------------------------------------------------------------===*
 * RVV-vectorized matmul (auto-vectorization friendly)
 *
 * This version is structured for the compiler's auto-vectorizer to exploit
 * RVV. The inner loop over N is the vectorizable dimension.
 *===----------------------------------------------------------------------===*/
void matmul_rvv_f32(const float *A, const float *B, float *C,
                    int M, int N, int K) {
    /* Zero-initialize output */
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * K + k];
            /* Inner loop over N: compiler will auto-vectorize with RVV */
            for (int j = 0; j < N; j++) {
                C[i * N + j] += a_ik * B[k * N + j];
            }
        }
    }
}

/*===----------------------------------------------------------------------===*
 * Tiled matmul for cache efficiency
 * Tile sizes chosen per RVV spec: M_TILE=16, N_TILE=16, K_TILE=16
 * (smaller tiles for testing; production would use 64x64x32)
 *===----------------------------------------------------------------------===*/
#define M_TILE 16
#define N_TILE 16
#define K_TILE 16

static inline int min_int(int a, int b) { return a < b ? a : b; }

void matmul_tiled_f32(const float *A, const float *B, float *C,
                      int M, int N, int K) {
    memset(C, 0, (size_t)M * N * sizeof(float));

    for (int i0 = 0; i0 < M; i0 += M_TILE) {
        int i_end = min_int(i0 + M_TILE, M);
        for (int j0 = 0; j0 < N; j0 += N_TILE) {
            int j_end = min_int(j0 + N_TILE, N);
            for (int k0 = 0; k0 < K; k0 += K_TILE) {
                int k_end = min_int(k0 + K_TILE, K);
                /* Micro-kernel: tile of C += tile of A * tile of B */
                for (int i = i0; i < i_end; i++) {
                    for (int k = k0; k < k_end; k++) {
                        float a_ik = A[i * K + k];
                        for (int j = j0; j < j_end; j++) {
                            C[i * N + j] += a_ik * B[k * N + j];
                        }
                    }
                }
            }
        }
    }
}

/*===----------------------------------------------------------------------===*
 * Comparison utility
 *===----------------------------------------------------------------------===*/
int compare_matrices(const float *computed, const float *reference,
                     int M, int N, float atol, float rtol) {
    int errors = 0;
    float max_abs_err = 0.0f;
    float max_rel_err = 0.0f;

    for (int i = 0; i < M * N; i++) {
        float abs_err = fabsf(computed[i] - reference[i]);
        float rel_err = abs_err / (fabsf(reference[i]) + 1e-10f);

        if (abs_err > max_abs_err) max_abs_err = abs_err;
        if (rel_err > max_rel_err) max_rel_err = rel_err;

        if (abs_err > atol && rel_err > rtol) {
            if (errors < 5) {
                printf("  MISMATCH [%d,%d]: computed=%e reference=%e "
                       "abs_err=%e rel_err=%e\n",
                       i / N, i % N, computed[i], reference[i],
                       abs_err, rel_err);
            }
            errors++;
        }
    }

    printf("  max_abs_error=%e  max_rel_error=%e  mismatches=%d/%d\n",
           max_abs_err, max_rel_err, errors, M * N);
    return errors;
}

/*===----------------------------------------------------------------------===*
 * Self-test: built-in validation without external files
 *===----------------------------------------------------------------------===*/
int run_self_test(void) {
    printf("=== MatMul f32 Self-Test ===\n\n");
    int total_failures = 0;

    /* Test case 1: 4x4 identity */
    {
        printf("Test 1: 4x4 * identity\n");
        float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float I[16] = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        float C_scalar[16], C_rvv[16], C_tiled[16];

        matmul_scalar_f32(A, I, C_scalar, 4, 4, 4);
        matmul_rvv_f32(A, I, C_rvv, 4, 4, 4);
        matmul_tiled_f32(A, I, C_tiled, 4, 4, 4);

        int err_s = compare_matrices(C_scalar, A, 4, 4, 1e-6f, 1e-5f);
        int err_v = compare_matrices(C_rvv, A, 4, 4, 1e-6f, 1e-5f);
        int err_t = compare_matrices(C_tiled, A, 4, 4, 1e-6f, 1e-5f);

        printf("  scalar: %s  rvv: %s  tiled: %s\n\n",
               err_s == 0 ? "PASS" : "FAIL",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_s + err_v + err_t);
    }

    /* Test case 2: 4x4 known values */
    {
        printf("Test 2: 4x4 known values\n");
        float A[16] = {1,2,3,4, 5,6,7,8, 9,10,11,12, 13,14,15,16};
        float B[16] = {16,15,14,13, 12,11,10,9, 8,7,6,5, 4,3,2,1};
        /* Expected: row 0 = 1*16+2*12+3*8+4*4 = 16+24+24+16 = 80, ... */
        float expected[16] = {
            80,  70,  60,  50,
            240, 214, 188, 162,
            400, 358, 316, 274,
            560, 502, 444, 386
        };

        float C_scalar[16], C_rvv[16], C_tiled[16];
        matmul_scalar_f32(A, B, C_scalar, 4, 4, 4);
        matmul_rvv_f32(A, B, C_rvv, 4, 4, 4);
        matmul_tiled_f32(A, B, C_tiled, 4, 4, 4);

        int err_s = compare_matrices(C_scalar, expected, 4, 4, 1e-5f, 1e-5f);
        int err_v = compare_matrices(C_rvv, expected, 4, 4, 1e-5f, 1e-5f);
        int err_t = compare_matrices(C_tiled, expected, 4, 4, 1e-5f, 1e-5f);

        printf("  scalar: %s  rvv: %s  tiled: %s\n\n",
               err_s == 0 ? "PASS" : "FAIL",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_s + err_v + err_t);
    }

    /* Test case 3: rectangular 3x5 * 5x2 */
    {
        printf("Test 3: 3x5 * 5x2 rectangular\n");
        float A[15] = {1,2,3,4,5, 6,7,8,9,10, 11,12,13,14,15};
        float B[10] = {1,2, 3,4, 5,6, 7,8, 9,10};
        /* Expected:
         * row0 = 1*1+2*3+3*5+4*7+5*9 = 1+6+15+28+45 = 95
         * row0 = 1*2+2*4+3*6+4*8+5*10 = 2+8+18+32+50 = 110
         */
        float expected[6] = {95, 110, 220, 260, 345, 410};

        float C_scalar[6], C_rvv[6], C_tiled[6];
        matmul_scalar_f32(A, B, C_scalar, 3, 2, 5);
        matmul_rvv_f32(A, B, C_rvv, 3, 2, 5);
        matmul_tiled_f32(A, B, C_tiled, 3, 2, 5);

        int err_s = compare_matrices(C_scalar, expected, 3, 2, 1e-5f, 1e-5f);
        int err_v = compare_matrices(C_rvv, expected, 3, 2, 1e-5f, 1e-5f);
        int err_t = compare_matrices(C_tiled, expected, 3, 2, 1e-5f, 1e-5f);

        printf("  scalar: %s  rvv: %s  tiled: %s\n\n",
               err_s == 0 ? "PASS" : "FAIL",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_s + err_v + err_t);
    }

    /* Test case 4: 1x1 degenerate */
    {
        printf("Test 4: 1x1 degenerate\n");
        float A[1] = {3.0f};
        float B[1] = {7.0f};
        float expected[1] = {21.0f};
        float C_scalar[1], C_rvv[1], C_tiled[1];

        matmul_scalar_f32(A, B, C_scalar, 1, 1, 1);
        matmul_rvv_f32(A, B, C_rvv, 1, 1, 1);
        matmul_tiled_f32(A, B, C_tiled, 1, 1, 1);

        int err_s = compare_matrices(C_scalar, expected, 1, 1, 1e-6f, 1e-5f);
        int err_v = compare_matrices(C_rvv, expected, 1, 1, 1e-6f, 1e-5f);
        int err_t = compare_matrices(C_tiled, expected, 1, 1, 1e-6f, 1e-5f);

        printf("  scalar: %s  rvv: %s  tiled: %s\n\n",
               err_s == 0 ? "PASS" : "FAIL",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_s + err_v + err_t);
    }

    /* Test case 5: non-power-of-2 (7x13 * 13x11) with random-ish data */
    {
        printf("Test 5: 7x13 * 13x11 non-power-of-2\n");
        int M = 7, K = 13, N = 11;
        float A[7 * 13], B[13 * 11];
        float C_scalar[7 * 11], C_rvv[7 * 11], C_tiled[7 * 11];

        /* Deterministic pseudo-random fill */
        for (int i = 0; i < M * K; i++)
            A[i] = (float)((i * 17 + 3) % 100) / 50.0f - 1.0f;
        for (int i = 0; i < K * N; i++)
            B[i] = (float)((i * 31 + 7) % 100) / 50.0f - 1.0f;

        matmul_scalar_f32(A, B, C_scalar, M, N, K);
        matmul_rvv_f32(A, B, C_rvv, M, N, K);
        matmul_tiled_f32(A, B, C_tiled, M, N, K);

        /* Compare RVV and tiled against scalar reference */
        int err_v = compare_matrices(C_rvv, C_scalar, M, N, 1e-5f, 1e-4f);
        int err_t = compare_matrices(C_tiled, C_scalar, M, N, 1e-5f, 1e-4f);

        printf("  rvv_vs_scalar: %s  tiled_vs_scalar: %s\n\n",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_v + err_t);
    }

    /* Test case 6: larger 32x32 stress test */
    {
        printf("Test 6: 32x32 stress test\n");
        int M = 32, K = 32, N = 32;
        float *A = (float *)malloc((size_t)M * K * sizeof(float));
        float *B = (float *)malloc((size_t)K * N * sizeof(float));
        float *C_scalar = (float *)malloc((size_t)M * N * sizeof(float));
        float *C_rvv = (float *)malloc((size_t)M * N * sizeof(float));
        float *C_tiled = (float *)malloc((size_t)M * N * sizeof(float));

        for (int i = 0; i < M * K; i++)
            A[i] = (float)((i * 23 + 5) % 200) / 100.0f - 1.0f;
        for (int i = 0; i < K * N; i++)
            B[i] = (float)((i * 37 + 11) % 200) / 100.0f - 1.0f;

        matmul_scalar_f32(A, B, C_scalar, M, N, K);
        matmul_rvv_f32(A, B, C_rvv, M, N, K);
        matmul_tiled_f32(A, B, C_tiled, M, N, K);

        int err_v = compare_matrices(C_rvv, C_scalar, M, N, 1e-4f, 1e-3f);
        int err_t = compare_matrices(C_tiled, C_scalar, M, N, 1e-4f, 1e-3f);

        printf("  rvv_vs_scalar: %s  tiled_vs_scalar: %s\n\n",
               err_v == 0 ? "PASS" : "FAIL",
               err_t == 0 ? "PASS" : "FAIL");
        total_failures += (err_v + err_t);

        free(A); free(B); free(C_scalar); free(C_rvv); free(C_tiled);
    }

    printf("=== Summary: %s (%d mismatches total) ===\n",
           total_failures == 0 ? "ALL PASSED" : "FAILURES DETECTED",
           total_failures);
    return total_failures == 0 ? 0 : 1;
}

/*===----------------------------------------------------------------------===*
 * File I/O mode: read binary inputs, compute matmul, write output
 *===----------------------------------------------------------------------===*/
float *read_matrix(const char *path, int rows, int cols) {
    FILE *f = fopen(path, "rb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s\n", path);
        return NULL;
    }
    float *data = (float *)malloc((size_t)rows * cols * sizeof(float));
    size_t read = fread(data, sizeof(float), (size_t)rows * cols, f);
    fclose(f);
    if ((int)read != rows * cols) {
        fprintf(stderr, "ERROR: Expected %d floats, read %zu from %s\n",
                rows * cols, read, path);
        free(data);
        return NULL;
    }
    return data;
}

int write_matrix(const char *path, const float *data, int rows, int cols) {
    FILE *f = fopen(path, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: Cannot open %s for writing\n", path);
        return -1;
    }
    fwrite(data, sizeof(float), (size_t)rows * cols, f);
    fclose(f);
    return 0;
}

/*===----------------------------------------------------------------------===*
 * Main entry point
 *===----------------------------------------------------------------------===*/
int main(int argc, char **argv) {
    if (argc == 2 && strcmp(argv[1], "--self-test") == 0) {
        return run_self_test();
    }

    if (argc == 7) {
        int M = atoi(argv[1]);
        int N = atoi(argv[2]);
        int K = atoi(argv[3]);

        if (M <= 0 || N <= 0 || K <= 0) {
            fprintf(stderr, "ERROR: Invalid dimensions M=%d N=%d K=%d\n",
                    M, N, K);
            return 1;
        }

        float *A = read_matrix(argv[4], M, K);
        float *B = read_matrix(argv[5], K, N);
        if (!A || !B) return 1;

        float *C = (float *)malloc((size_t)M * N * sizeof(float));

        /* Use tiled version (auto-vectorized by RVV compiler) */
        matmul_tiled_f32(A, B, C, M, N, K);

        int rc = write_matrix(argv[6], C, M, N);

        printf("MATMUL_DONE M=%d N=%d K=%d\n", M, N, K);

        free(A); free(B); free(C);
        return rc;
    }

    fprintf(stderr,
            "Usage: %s --self-test\n"
            "   or: %s <M> <N> <K> <A.bin> <B.bin> <C_out.bin>\n",
            argv[0], argv[0]);
    return 1;
}
