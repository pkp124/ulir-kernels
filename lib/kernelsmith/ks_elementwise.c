/*
 * KernelSmith Elementwise Functions - Reference Implementation
 *
 * Contiguous scalar C99. In-place safe (output may alias either input).
 */

#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_elementwise.h"

#include <stddef.h>

int ks_add_f32(const float *lhs, const float *rhs, float *output, size_t n) {
  if (!lhs || !rhs || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t i = 0; i < n; i++)
    output[i] = lhs[i] + rhs[i];

  return KS_OK;
}

int ks_mul_f32(const float *lhs, const float *rhs, float *output, size_t n) {
  if (!lhs || !rhs || !output)
    return KS_ERR_INVALID_ARG;

  for (size_t i = 0; i < n; i++)
    output[i] = lhs[i] * rhs[i];

  return KS_OK;
}
