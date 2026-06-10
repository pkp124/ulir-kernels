/*
 * KernelSmith Quantized Dot/GEMV — Reference Implementation
 *
 * Scalar C99 implementations for ABI validation and golden tests.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_quantized.h"

/* Target profile is force-included via -include in CMake. */

#include <math.h>
#include <stddef.h>
#include <stdint.h>

static int ks_i8_zero_point_valid(int32_t zero_point) {
  return zero_point >= -128 && zero_point <= 127;
}

static int ks_i4_zero_point_valid(int32_t zero_point) {
  return zero_point >= -8 && zero_point <= 7;
}

static int ks_positive_finite(float value) {
  return value > 0.0f && isfinite(value);
}

static int8_t ks_unpack_i4(uint8_t byte, size_t index) {
  uint8_t nibble = (index & 1u) ? (byte >> 4) : (byte & 0x0f);
  if (nibble & 0x08)
    nibble |= 0xf0;
  return (int8_t)nibble;
}

static int ks_validate_w4a8_scales(const float *scales, size_t groups) {
  for (size_t i = 0; i < groups; ++i) {
    if (!ks_positive_finite(scales[i]))
      return 0;
  }
  return 1;
}

int ks_dot_i8(const int8_t *input, const int8_t *weight, size_t k,
              int32_t input_zero_point, int32_t weight_zero_point,
              int32_t *output, void *workspace, size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !weight || !output)
    return KS_ERR_INVALID_ARG;
  if (!ks_i8_zero_point_valid(input_zero_point) ||
      !ks_i8_zero_point_valid(weight_zero_point))
    return KS_ERR_INVALID_ARG;

  int32_t accumulator = 0;
  for (size_t i = 0; i < k; ++i) {
    int32_t x = (int32_t)input[i] - input_zero_point;
    int32_t w = (int32_t)weight[i] - weight_zero_point;
    accumulator += x * w;
  }

  *output = accumulator;
  return KS_OK;
}

int ks_matvec_i8(const int8_t *input, const int8_t *weights,
                 size_t weight_stride, int32_t *output, size_t rows,
                 size_t cols, int32_t input_zero_point,
                 int32_t weight_zero_point, void *workspace, size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !weights || !output)
    return KS_ERR_INVALID_ARG;
  if (rows == 0 || cols == 0 || weight_stride < cols)
    return KS_ERR_INVALID_ARG;
  if (!ks_i8_zero_point_valid(input_zero_point) ||
      !ks_i8_zero_point_valid(weight_zero_point))
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < rows; ++row) {
    const int8_t *weight_row = weights + row * weight_stride;
    int rc = ks_dot_i8(input, weight_row, cols, input_zero_point,
                       weight_zero_point, &output[row], NULL, 0);
    if (rc != KS_OK)
      return rc;
  }

  return KS_OK;
}

int ks_dot_w4a8(const int8_t *input, const uint8_t *packed_weight,
                const float *weight_scales, size_t k, size_t group_size,
                float input_scale, int32_t input_zero_point,
                int32_t weight_zero_point, float *output, void *workspace,
                size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !packed_weight || !weight_scales || !output)
    return KS_ERR_INVALID_ARG;
  if (group_size == 0)
    return KS_ERR_INVALID_ARG;
  if (!ks_positive_finite(input_scale))
    return KS_ERR_INVALID_ARG;
  if (!ks_i8_zero_point_valid(input_zero_point) ||
      !ks_i4_zero_point_valid(weight_zero_point))
    return KS_ERR_INVALID_ARG;

  size_t groups = ks_w4a8_scale_groups(k, group_size);
  if (!ks_validate_w4a8_scales(weight_scales, groups))
    return KS_ERR_INVALID_ARG;

  float accumulator = 0.0f;
  for (size_t i = 0; i < k; ++i) {
    uint8_t packed = packed_weight[i / 2];
    int32_t x = (int32_t)input[i] - input_zero_point;
    int32_t w = (int32_t)ks_unpack_i4(packed, i) - weight_zero_point;
    float weight_scale = weight_scales[i / group_size];
    accumulator += ((float)x * input_scale) * ((float)w * weight_scale);
  }

  *output = accumulator;
  return KS_OK;
}

int ks_matvec_w4a8(const int8_t *input, const uint8_t *packed_weights,
                   size_t packed_stride_bytes, const float *weight_scales,
                   size_t scale_stride, float *output, size_t rows,
                   size_t cols, size_t group_size, float input_scale,
                   int32_t input_zero_point, int32_t weight_zero_point,
                   void *workspace, size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !packed_weights || !weight_scales || !output)
    return KS_ERR_INVALID_ARG;
  if (rows == 0 || cols == 0 || group_size == 0)
    return KS_ERR_INVALID_ARG;

  size_t packed_row_bytes = ks_w4a8_packed_row_bytes(cols);
  size_t scale_groups = ks_w4a8_scale_groups(cols, group_size);
  if (packed_stride_bytes < packed_row_bytes || scale_stride < scale_groups)
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < rows; ++row) {
    const uint8_t *weight_row = packed_weights + row * packed_stride_bytes;
    const float *scale_row = weight_scales + row * scale_stride;
    int rc = ks_dot_w4a8(input, weight_row, scale_row, cols, group_size,
                         input_scale, input_zero_point, weight_zero_point,
                         &output[row], NULL, 0);
    if (rc != KS_OK)
      return rc;
  }

  return KS_OK;
}

size_t ks_dot_i8_workspace(size_t k) {
  (void)k;
  return 0;
}

size_t ks_matvec_i8_workspace(size_t rows, size_t cols) {
  (void)rows;
  (void)cols;
  return 0;
}

size_t ks_dot_w4a8_workspace(size_t k, size_t group_size) {
  (void)k;
  (void)group_size;
  return 0;
}

size_t ks_matvec_w4a8_workspace(size_t rows, size_t cols, size_t group_size) {
  (void)rows;
  (void)cols;
  (void)group_size;
  return 0;
}

size_t ks_quantized_alignment(void) { return KS_PREFERRED_ALIGN; }

size_t ks_w4a8_packed_row_bytes(size_t cols) { return (cols + 1u) / 2u; }

size_t ks_w4a8_scale_groups(size_t cols, size_t group_size) {
  if (group_size == 0)
    return 0;
  return (cols + group_size - 1u) / group_size;
}
