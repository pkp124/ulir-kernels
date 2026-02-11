/*
 * Smoke tests for activation functions: relu, gelu, silu.
 *
 * Returns 0 on success, 1 on any failure.
 */

#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_common.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define ABS_TOL 1e-5f

static int tests_run = 0;
static int tests_passed = 0;

/* Reference implementations for checking. */
static float ref_relu(float x) { return x > 0.0f ? x : 0.0f; }

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440
#endif

static float ref_gelu(float x) {
  return 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2));
}

static float ref_silu(float x) { return x / (1.0f + expf(-x)); }

typedef float (*ref_fn)(float);

static void test_activation(const char *name,
                            int (*ks_fn)(const float *, float *, size_t),
                            ref_fn ref) {
  tests_run++;

  float input[] = {-3.0f, -1.5f, -0.5f, 0.0f, 0.5f, 1.5f, 3.0f};
  size_t n = sizeof(input) / sizeof(input[0]);
  float output[7];
  float expected[7];

  for (size_t i = 0; i < n; i++)
    expected[i] = ref(input[i]);

  int rc = ks_fn(input, output, n);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL %s: returned %d\n", name, rc);
    return;
  }

  for (size_t i = 0; i < n; i++) {
    if (fabsf(output[i] - expected[i]) > ABS_TOL) {
      fprintf(stderr, "  FAIL %s: output[%zu] = %f, expected %f (input=%f)\n",
              name, i, output[i], expected[i], input[i]);
      return;
    }
  }

  printf("  PASS %s (%zu values)\n", name, n);
  tests_passed++;
}

static void test_inplace(void) {
  tests_run++;

  float buf[] = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
  size_t n = 5;
  float expected[5];
  for (size_t i = 0; i < n; i++)
    expected[i] = ref_relu(buf[i]);

  /* In-place: output == input. */
  int rc = ks_relu_f32(buf, buf, n);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL relu in-place: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < n; i++) {
    if (fabsf(buf[i] - expected[i]) > ABS_TOL) {
      fprintf(stderr, "  FAIL relu in-place: buf[%zu] = %f, expected %f\n",
              i, buf[i], expected[i]);
      return;
    }
  }

  printf("  PASS relu in-place\n");
  tests_passed++;
}

static void test_error_null(void) {
  tests_run++;
  float dummy;
  int rc = ks_relu_f32(NULL, &dummy, 1);
  if (rc == KS_ERR_INVALID_ARG) {
    printf("  PASS error on NULL input\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL error on NULL input: returned %d\n", rc);
  }
}

static void test_empty(void) {
  tests_run++;
  float dummy;
  int rc = ks_relu_f32(&dummy, &dummy, 0);
  if (rc == KS_OK) {
    printf("  PASS empty (n=0) returns OK\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL empty (n=0): returned %d\n", rc);
  }
}

int main(void) {
  printf("=== activation smoke tests ===\n");
  printf("Target: %s\n\n", ks_target_name());

  test_activation("relu", ks_relu_f32, ref_relu);
  test_activation("gelu", ks_gelu_f32, ref_gelu);
  test_activation("silu", ks_silu_f32, ref_silu);
  test_inplace();
  test_error_null();
  test_empty();

  printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
