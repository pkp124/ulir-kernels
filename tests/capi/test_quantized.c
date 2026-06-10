/*
 * Smoke tests for quantized dot/GEMV C APIs.
 *
 * Validates: i8 dot/GEMV, W4A8 dot/GEMV, layout helpers, and error cases.
 * Returns 0 on success, 1 on any failure.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_quantized.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>

#define ABS_TOL 1e-5f

static int tests_run = 0;
static int tests_passed = 0;

static uint8_t pack_i4_pair(int8_t low, int8_t high) {
  return (uint8_t)(((uint8_t)low & 0x0f) | (((uint8_t)high & 0x0f) << 4));
}

static int check_float(float got, float expected, const char *label) {
  float diff = fabsf(got - expected);
  if (diff > ABS_TOL) {
    fprintf(stderr, "  FAIL %s: got %f, expected %f\n", label, got, expected);
    return 0;
  }
  return 1;
}

static void test_dot_i8(void) {
  tests_run++;

  const int8_t x[] = {3, -2, 7, 1};
  const int8_t w[] = {2, -5, 4, -3};
  int32_t out = 0;

  int rc = ks_dot_i8(x, w, 4, 1, -1, &out, NULL, 0);
  if (rc == KS_OK && out == 48) {
    printf("  PASS ks_dot_i8 asymmetric zero points\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL ks_dot_i8: rc=%d out=%d\n", rc, (int)out);
  }
}

static void test_matvec_i8(void) {
  tests_run++;

  const int8_t x[] = {1, 2, 3};
  const int8_t weights[] = {
      1, 0, -1, 0,
      -2, 2, 3, 9,
  };
  int32_t y[] = {-1, -1};

  int rc = ks_matvec_i8(x, weights, 4, y, 2, 3, 0, 0, NULL, 0);
  if (rc == KS_OK && y[0] == -2 && y[1] == 11) {
    printf("  PASS ks_matvec_i8 row-major weights\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL ks_matvec_i8: rc=%d y=[%d,%d]\n",
            rc, (int)y[0], (int)y[1]);
  }
}

static void test_w4a8_helpers(void) {
  tests_run++;

  size_t row_bytes = ks_w4a8_packed_row_bytes(5);
  size_t groups = ks_w4a8_scale_groups(17, 8);
  size_t bad_groups = ks_w4a8_scale_groups(17, 0);
  if (row_bytes == 3 && groups == 3 && bad_groups == 0) {
    printf("  PASS W4A8 layout helpers\n");
    tests_passed++;
  } else {
    fprintf(stderr,
            "  FAIL W4A8 helpers: row_bytes=%zu groups=%zu bad_groups=%zu\n",
            row_bytes, groups, bad_groups);
  }
}

static void test_dot_w4a8(void) {
  tests_run++;

  const int8_t x[] = {4, -2, 3, 1};
  const uint8_t packed_w[] = {
      pack_i4_pair(1, -2),
      pack_i4_pair(3, -4),
  };
  const float scales[] = {0.5f, 0.25f};
  float out = 0.0f;

  int rc = ks_dot_w4a8(x, packed_w, scales, 4, 2, 0.25f, 0, 0, &out, NULL, 0);
  if (rc == KS_OK && check_float(out, 1.3125f, "ks_dot_w4a8")) {
    printf("  PASS ks_dot_w4a8 grouped scales\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL ks_dot_w4a8: rc=%d out=%f\n", rc, out);
  }
}

static void test_matvec_w4a8(void) {
  tests_run++;

  const int8_t x[] = {2, -1, 3};
  const uint8_t packed_weights[] = {
      pack_i4_pair(1, 2), pack_i4_pair(-3, 0),
      pack_i4_pair(-1, 0), pack_i4_pair(4, 0),
  };
  const float scales[] = {0.5f, 0.25f};
  float y[] = {0.0f, 0.0f};

  int rc = ks_matvec_w4a8(x, packed_weights, 2, scales, 1, y, 2, 3, 3,
                          0.5f, 0, 0, NULL, 0);
  if (rc == KS_OK &&
      check_float(y[0], -2.25f, "ks_matvec_w4a8 row 0") &&
      check_float(y[1], 1.25f, "ks_matvec_w4a8 row 1")) {
    printf("  PASS ks_matvec_w4a8 packed rows\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL ks_matvec_w4a8: rc=%d y=[%f,%f]\n", rc, y[0], y[1]);
  }
}

static void test_workspace_and_alignment(void) {
  tests_run++;

  if (ks_dot_i8_workspace(16) == 0 &&
      ks_matvec_i8_workspace(4, 16) == 0 &&
      ks_dot_w4a8_workspace(16, 8) == 0 &&
      ks_matvec_w4a8_workspace(4, 16, 8) == 0 &&
      ks_quantized_alignment() > 0) {
    printf("  PASS quantized workspace/alignment queries\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL quantized workspace/alignment queries\n");
  }
}

static void test_error_cases(void) {
  tests_run++;

  int32_t i32_out = 0;
  float f32_out = 0.0f;
  int8_t x = 1;
  int8_t w = 1;
  uint8_t packed = 0;
  float scale = 1.0f;

  int ok = 1;
  ok &= ks_dot_i8(NULL, &w, 1, 0, 0, &i32_out, NULL, 0) == KS_ERR_INVALID_ARG;
  ok &= ks_dot_i8(&x, &w, 1, 129, 0, &i32_out, NULL, 0) == KS_ERR_INVALID_ARG;
  ok &= ks_matvec_i8(&x, &w, 0, &i32_out, 1, 1, 0, 0, NULL, 0) ==
        KS_ERR_INVALID_ARG;
  ok &= ks_dot_w4a8(&x, &packed, &scale, 1, 0, 1.0f, 0, 0, &f32_out,
                    NULL, 0) == KS_ERR_INVALID_ARG;
  ok &= ks_dot_w4a8(&x, &packed, &scale, 1, 1, -1.0f, 0, 0, &f32_out,
                    NULL, 0) == KS_ERR_INVALID_ARG;
  ok &= ks_dot_w4a8(&x, &packed, &scale, 1, 1, 1.0f, 0, 8, &f32_out,
                    NULL, 0) == KS_ERR_INVALID_ARG;

  if (ok) {
    printf("  PASS quantized error cases\n");
    tests_passed++;
  } else {
    fprintf(stderr, "  FAIL quantized error cases\n");
  }
}

int main(void) {
  printf("=== quantized C API smoke tests ===\n");
  printf("Target: %s\n\n", ks_target_name());

  test_dot_i8();
  test_matvec_i8();
  test_w4a8_helpers();
  test_dot_w4a8();
  test_matvec_w4a8();
  test_workspace_and_alignment();
  test_error_cases();

  printf("\n%d/%d tests passed.\n", tests_passed, tests_run);
  return tests_passed == tests_run ? 0 : 1;
}
