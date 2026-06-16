/*
 * KernelSmith Normalization Functions - Reference Implementation
 *
 * Contiguous scalar C99. Reductions use f32 accumulation.
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_normalization.h"

#include <math.h>
#include <stddef.h>

int ks_rms_norm_f32(const float *input, const float *weight, float *output,
                    size_t outer, size_t inner, float eps) {
  if (!input || !weight || !output || inner == 0 || eps <= 0.0f ||
      !isfinite(eps))
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < outer; row++) {
    const size_t base = row * inner;
    float sum_squares = 0.0f;

    for (size_t col = 0; col < inner; col++) {
      const float value = input[base + col];
      sum_squares += value * value;
    }

    const float mean_squares = sum_squares / (float)inner;
    const float scale = 1.0f / sqrtf(mean_squares + eps);

    for (size_t col = 0; col < inner; col++)
      output[base + col] = input[base + col] * scale * weight[col];
  }

  return KS_OK;
}

int ks_softmax_f32(const float *input, float *output, size_t outer,
                   size_t inner) {
  if (!input || !output || input == output || inner == 0)
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < outer; row++) {
    const size_t base = row * inner;
    float max_value = input[base];

    for (size_t col = 1; col < inner; col++) {
      const float value = input[base + col];
      if (value > max_value)
        max_value = value;
    }

    float sum = 0.0f;
    for (size_t col = 0; col < inner; col++) {
      const float value = expf(input[base + col] - max_value);
      output[base + col] = value;
      sum += value;
    }

    const float inv_sum = 1.0f / sum;
    for (size_t col = 0; col < inner; col++)
      output[base + col] *= inv_sum;
  }

  return KS_OK;
}
