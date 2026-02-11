/*
 * KernelSmith Activation Functions — Reference Implementation
 *
 * Element-wise scalar C99. In-place safe (output may alias input).
 */

#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_common.h"

#include <math.h>
#include <stddef.h>

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.70710678118654752440 /* 1/sqrt(2) */
#endif

int ks_relu_f32(const float *input, float *output, size_t n) {
  if (!input || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t i = 0; i < n; i++) {
    output[i] = input[i] > 0.0f ? input[i] : 0.0f;
  }
  return KS_OK;
}

int ks_gelu_f32(const float *input, float *output, size_t n) {
  if (!input || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t i = 0; i < n; i++) {
    float x = input[i];
    /* Exact GELU: 0.5 * x * (1 + erf(x / sqrt(2))) */
    output[i] = 0.5f * x * (1.0f + erff(x * (float)M_SQRT1_2));
  }
  return KS_OK;
}

int ks_silu_f32(const float *input, float *output, size_t n) {
  if (!input || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t i = 0; i < n; i++) {
    float x = input[i];
    /* SiLU: x * sigmoid(x) = x / (1 + exp(-x)) */
    output[i] = x / (1.0f + expf(-x));
  }
  return KS_OK;
}
