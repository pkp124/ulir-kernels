/*
 * Verifies that a profile build can replace public INT8 C API symbols with
 * generated target objects while preserving the rest of the quantized API.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_quantized.h"

#include <stdint.h>
#include <stdio.h>
#include <string.h>

static int check_generated_dot(void) {
  const int8_t input[] = {1, 2, 3};
  const int8_t weight[] = {4, 5, 6};
  int32_t output = 0;

  int rc = ks_dot_i8(input, weight, 3, 0, 0, &output, NULL, 0);
  if (rc != KS_OK || output != 0x12345678) {
    fprintf(stderr, "ks_dot_i8 did not use generated object: rc=%d out=%d\n",
            rc, (int)output);
    return 1;
  }

  return 0;
}

static int check_generated_matvec(void) {
  const int8_t input[] = {1, 2};
  const int8_t weights[] = {3, 4, 5, 6};
  int32_t output[] = {0, 0};

  int rc = ks_matvec_i8(input, weights, 2, output, 2, 2, 0, 0, NULL, 0);
  if (rc != KS_OK || output[0] != 0x23450000 || output[1] != 0x23450001) {
    fprintf(stderr,
            "ks_matvec_i8 did not use generated object: rc=%d out=[%d,%d]\n",
            rc, (int)output[0], (int)output[1]);
    return 1;
  }

  return 0;
}

static int check_reference_w4a8_still_linked(void) {
  const int8_t input[] = {2, -1};
  const uint8_t packed_weight[] = {0x11};
  const float scales[] = {0.5f};
  float output = 0.0f;

  int rc = ks_dot_w4a8(input, packed_weight, scales, 2, 2, 1.0f, 0, 0,
                       &output, NULL, 0);
  if (rc != KS_OK || output != 0.5f) {
    fprintf(stderr, "ks_dot_w4a8 reference path changed: rc=%d out=%f\n", rc,
            output);
    return 1;
  }

  return 0;
}

int main(void) {
  if (strcmp(ks_target_name(), "riscv-rvv-256") != 0) {
    fprintf(stderr, "unexpected target profile: %s\n", ks_target_name());
    return 1;
  }

  if (check_generated_dot() != 0)
    return 1;
  if (check_generated_matvec() != 0)
    return 1;
  if (check_reference_w4a8_still_linked() != 0)
    return 1;

  printf("PASS generated INT8 profile override\n");
  return 0;
}
