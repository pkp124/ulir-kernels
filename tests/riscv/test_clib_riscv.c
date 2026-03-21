/*
 * KernelSmith C library correctness test — RISC-V cross-compilation target.
 *
 * Tests ks_relu_f32 and ks_matmul_f32 from the reference C library
 * (lib/kernelsmith/) against expected outputs.
 *
 * Compile for riscv64:
 *   riscv64-linux-gnu-gcc -static \
 *       -I include/kernelsmith \
 *       tests/riscv/test_clib_riscv.c \
 *       -Lbuild-riscv/lib -lkernelsmith \
 *       -o test_clib_riscv -lm
 *
 * Run on QEMU:
 *   qemu-riscv64 -cpu rv64,v=true,vlen=256 ./test_clib_riscv
 *
 * Exit code 0 = all tests passed, non-zero = test count of failures.
 */

#include "ks_activations.h"
#include "ks_matmul.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

/* -------------------------------------------------------------------------
 * Helpers
 * ---------------------------------------------------------------------- */

#define N_RELU 16
#define M 4
#define K 4
#define N 4

static int failures = 0;

static void check_f32(const char *name, float got, float expected, float eps) {
    if (fabsf(got - expected) > eps) {
        printf("FAIL %s: got %.6f expected %.6f (delta %.2e)\n",
               name, (double)got, (double)expected, (double)fabsf(got - expected));
        failures++;
    }
}

/* -------------------------------------------------------------------------
 * ReLU test
 * ---------------------------------------------------------------------- */

static void test_relu(void) {
    float in[N_RELU], out[N_RELU];
    float expected[N_RELU];

    for (int i = 0; i < N_RELU; i++) {
        in[i] = (float)(i - 8);           /* -8, -7, ..., 7 */
        expected[i] = in[i] > 0.0f ? in[i] : 0.0f;
    }

    ks_status_t st = ks_relu_f32(in, out, N_RELU);
    if (st != KS_SUCCESS) {
        printf("FAIL ks_relu_f32 returned error %d\n", (int)st);
        failures++;
        return;
    }

    for (int i = 0; i < N_RELU; i++) {
        char name[32];
        snprintf(name, sizeof(name), "relu[%d]", i);
        check_f32(name, out[i], expected[i], 1e-6f);
    }
    printf("relu: %s\n", failures == 0 ? "PASS" : "FAIL");
}

/* -------------------------------------------------------------------------
 * Matmul test: 4x4 * 4x4 = 4x4
 * ---------------------------------------------------------------------- */

static void test_matmul(void) {
    /* A = identity, B = known matrix, expect C = B */
    float A[M * K], B[K * N], C[M * N], expected[M * N];

    /* Identity matrix */
    memset(A, 0, sizeof(A));
    for (int i = 0; i < M; i++) A[i * K + i] = 1.0f;

    /* B: B[i][j] = i * N + j + 1 */
    for (int i = 0; i < K; i++)
        for (int j = 0; j < N; j++)
            B[i * N + j] = (float)(i * N + j + 1);

    memcpy(expected, B, sizeof(B));  /* I * B = B */

    ks_status_t st = ks_matmul_f32(A, B, C, M, N, K,
                                    K, N, N,  /* lda, ldb, ldc */
                                    1.0f, 0.0f);
    if (st != KS_SUCCESS) {
        printf("FAIL ks_matmul_f32 returned error %d\n", (int)st);
        failures++;
        return;
    }

    int before = failures;
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            char name[32];
            snprintf(name, sizeof(name), "matmul[%d][%d]", i, j);
            check_f32(name, C[i * N + j], expected[i * N + j], 1e-4f);
        }
    }
    printf("matmul 4x4: %s\n", (failures == before) ? "PASS" : "FAIL");
}

/* -------------------------------------------------------------------------
 * Main
 * ---------------------------------------------------------------------- */

int main(void) {
    printf("KernelSmith C library test (RISC-V)\n");
    printf("=====================================\n");

    test_relu();
    test_matmul();

    printf("=====================================\n");
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TEST(S) FAILED\n", failures);
    }
    return failures;
}
