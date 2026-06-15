/*
 * Mock generated INT8 object used by the CMake profile-integration test.
 *
 * Real TASK-018 generated RVV objects export these same public symbols. This
 * object returns sentinel values so the test can prove the linker selected the
 * external implementation instead of the scalar reference implementation.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_quantized.h"

#include <stddef.h>
#include <stdint.h>

int ks_dot_i8(const int8_t *input, const int8_t *weight, size_t k,
              int32_t input_zero_point, int32_t weight_zero_point,
              int32_t *output, void *workspace, size_t ws_size) {
  (void)k;
  (void)input_zero_point;
  (void)weight_zero_point;
  (void)workspace;
  (void)ws_size;

  if (!input || !weight || !output)
    return KS_ERR_INVALID_ARG;

  *output = 0x12345678;
  return KS_OK;
}

int ks_matvec_i8(const int8_t *input, const int8_t *weights,
                 size_t weight_stride, int32_t *output, size_t rows,
                 size_t cols, int32_t input_zero_point,
                 int32_t weight_zero_point, void *workspace, size_t ws_size) {
  (void)weight_stride;
  (void)cols;
  (void)input_zero_point;
  (void)weight_zero_point;
  (void)workspace;
  (void)ws_size;

  if (!input || !weights || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < rows; ++row)
    output[row] = 0x23450000 + (int32_t)row;

  return KS_OK;
}
