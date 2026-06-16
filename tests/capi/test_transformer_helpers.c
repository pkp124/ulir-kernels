/*
 * Smoke tests for transformer helper kernels: add, mul, RMSNorm, softmax.
 *
 * Returns 0 on success, 1 on any failure.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_elementwise.h"
#include "kernelsmith/ks_normalization.h"

#include <math.h>
#include <stdio.h>

#define ABS_TOL 1e-5f

static int tests_run = 0;
static int tests_passed = 0;

static int nearly_equal(float actual, float expected, float tol) {
  return fabsf(actual - expected) <= tol;
}

static void test_add(void) {
  tests_run++;

  const float lhs[] = {1.0f, -2.0f, 3.5f, 0.0f, -4.0f};
  const float rhs[] = {2.0f, 5.0f, -1.5f, 7.0f, -0.5f};
  const float expected[] = {3.0f, 3.0f, 2.0f, 7.0f, -4.5f};
  float output[5];

  int rc = ks_add_f32(lhs, rhs, output, 5);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL add: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < 5; i++) {
    if (!nearly_equal(output[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL add: output[%zu] = %f, expected %f\n", i,
              output[i], expected[i]);
      return;
    }
  }

  printf("  PASS add\n");
  tests_passed++;
}

static void test_mul(void) {
  tests_run++;

  const float lhs[] = {1.0f, -2.0f, 3.5f, 0.0f, -4.0f};
  const float rhs[] = {2.0f, 5.0f, -1.5f, 7.0f, -0.5f};
  const float expected[] = {2.0f, -10.0f, -5.25f, 0.0f, 2.0f};
  float output[5];

  int rc = ks_mul_f32(lhs, rhs, output, 5);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL mul: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < 5; i++) {
    if (!nearly_equal(output[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL mul: output[%zu] = %f, expected %f\n", i,
              output[i], expected[i]);
      return;
    }
  }

  printf("  PASS mul\n");
  tests_passed++;
}

static void test_elementwise_inplace(void) {
  tests_run++;

  float lhs[] = {1.0f, 2.0f, 3.0f};
  const float rhs[] = {4.0f, 5.0f, 6.0f};
  const float expected[] = {4.0f, 10.0f, 18.0f};

  int rc = ks_mul_f32(lhs, rhs, lhs, 3);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL elementwise in-place: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < 3; i++) {
    if (!nearly_equal(lhs[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL elementwise in-place: lhs[%zu] = %f, expected %f\n",
              i, lhs[i], expected[i]);
      return;
    }
  }

  printf("  PASS elementwise in-place\n");
  tests_passed++;
}

static void test_rms_norm(void) {
  tests_run++;

  const size_t outer = 2;
  const size_t inner = 4;
  const float input[] = {1.0f, 2.0f, 3.0f, 4.0f,
                         -1.0f, 0.5f, -0.5f, 2.0f};
  const float weight[] = {1.0f, 0.5f, 2.0f, -1.0f};
  float output[8];
  float expected[8];
  const float eps = 1e-5f;

  for (size_t row = 0; row < outer; row++) {
    const size_t base = row * inner;
    float sum_squares = 0.0f;
    for (size_t col = 0; col < inner; col++) {
      const float value = input[base + col];
      sum_squares += value * value;
    }
    const float scale = 1.0f / sqrtf(sum_squares / (float)inner + eps);
    for (size_t col = 0; col < inner; col++)
      expected[base + col] = input[base + col] * scale * weight[col];
  }

  int rc = ks_rms_norm_f32(input, weight, output, outer, inner, eps);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL rms_norm: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < outer * inner; i++) {
    if (!nearly_equal(output[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL rms_norm: output[%zu] = %f, expected %f\n", i,
              output[i], expected[i]);
      return;
    }
  }

  printf("  PASS rms_norm\n");
  tests_passed++;
}

static void test_rms_norm_inplace(void) {
  tests_run++;

  float input[] = {1.0f, -2.0f, 3.0f};
  const float weight[] = {1.0f, 1.0f, 1.0f};
  float expected[3];
  float sum_squares = 0.0f;

  for (size_t i = 0; i < 3; i++)
    sum_squares += input[i] * input[i];

  const float scale = 1.0f / sqrtf(sum_squares / 3.0f + 1e-5f);
  for (size_t i = 0; i < 3; i++)
    expected[i] = input[i] * scale;

  int rc = ks_rms_norm_f32(input, weight, input, 1, 3, 1e-5f);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL rms_norm in-place: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < 3; i++) {
    if (!nearly_equal(input[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL rms_norm in-place: input[%zu] = %f, expected %f\n",
              i, input[i], expected[i]);
      return;
    }
  }

  printf("  PASS rms_norm in-place\n");
  tests_passed++;
}

static void test_softmax(void) {
  tests_run++;

  const size_t outer = 2;
  const size_t inner = 3;
  const float input[] = {1.0f, 2.0f, 3.0f, 1000.0f, 1001.0f, 999.0f};
  float output[6];
  float expected[6];

  for (size_t row = 0; row < outer; row++) {
    const size_t base = row * inner;
    float max_value = input[base];
    for (size_t col = 1; col < inner; col++) {
      if (input[base + col] > max_value)
        max_value = input[base + col];
    }

    float sum = 0.0f;
    for (size_t col = 0; col < inner; col++) {
      expected[base + col] = expf(input[base + col] - max_value);
      sum += expected[base + col];
    }

    for (size_t col = 0; col < inner; col++)
      expected[base + col] /= sum;
  }

  int rc = ks_softmax_f32(input, output, outer, inner);
  if (rc != KS_OK) {
    fprintf(stderr, "  FAIL softmax: returned %d\n", rc);
    return;
  }

  for (size_t i = 0; i < outer * inner; i++) {
    if (!nearly_equal(output[i], expected[i], ABS_TOL)) {
      fprintf(stderr, "  FAIL softmax: output[%zu] = %f, expected %f\n", i,
              output[i], expected[i]);
      return;
    }
  }

  printf("  PASS softmax\n");
  tests_passed++;
}

static void test_errors(void) {
  tests_run++;

  float value = 1.0f;
  float output = 0.0f;
  const int ok = ks_add_f32(NULL, &value, &output, 1) == KS_ERR_INVALID_ARG &&
                 ks_mul_f32(&value, NULL, &output, 1) == KS_ERR_INVALID_ARG &&
                 ks_rms_norm_f32(&value, &value, &output, 1, 0, 1e-5f) ==
                     KS_ERR_INVALID_ARG &&
                 ks_rms_norm_f32(&value, &value, &output, 1, 1, 0.0f) ==
                     KS_ERR_INVALID_ARG &&
                 ks_softmax_f32(&value, &value, 1, 1) == KS_ERR_INVALID_ARG &&
                 ks_softmax_f32(&value, &output, 1, 0) == KS_ERR_INVALID_ARG;

  if (ok) {
    printf("  PASS invalid argument handling\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL invalid argument handling\n");
  }
}

static void test_empty_outer(void) {
  tests_run++;

  float value = 1.0f;
  float output = 0.0f;
  const int ok = ks_add_f32(&value, &value, &output, 0) == KS_OK &&
                 ks_mul_f32(&value, &value, &output, 0) == KS_OK &&
                 ks_rms_norm_f32(&value, &value, &output, 0, 1, 1e-5f) ==
                     KS_OK &&
                 ks_softmax_f32(&value, &output, 0, 1) == KS_OK;

  if (ok) {
    printf("  PASS empty sizes\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL empty sizes\n");
  }
}

int main(void) {
  printf("=== transformer helper smoke tests ===\n");
  printf("Target: %s\n\n", ks_target_name());

  test_add();
  test_mul();
  test_elementwise_inplace();
  test_rms_norm();
  test_rms_norm_inplace();
  test_softmax();
  test_errors();
  test_empty_outer();

  printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
