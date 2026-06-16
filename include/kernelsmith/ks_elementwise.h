/*
 * KernelSmith Elementwise Functions
 *
 * Contiguous f32 elementwise helpers for transformer residual paths.
 * In-place operation: output may alias either input.
 *
 * Thread safety: All functions are reentrant. No global state.
 */

#ifndef KERNELSMITH_ELEMENTWISE_H
#define KERNELSMITH_ELEMENTWISE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_add_f32 - Elementwise addition: output[i] = lhs[i] + rhs[i]
 *
 * Parameters:
 *   lhs     - Left input array of n elements
 *   rhs     - Right input array of n elements
 *   output  - Output array of n elements (may alias lhs or rhs)
 *   n       - Number of elements
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer
 */
int ks_add_f32(const float *lhs, const float *rhs, float *output, size_t n);

/*
 * ks_mul_f32 - Elementwise multiplication: output[i] = lhs[i] * rhs[i]
 *
 * Parameters:
 *   lhs     - Left input array of n elements
 *   rhs     - Right input array of n elements
 *   output  - Output array of n elements (may alias lhs or rhs)
 *   n       - Number of elements
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer
 */
int ks_mul_f32(const float *lhs, const float *rhs, float *output, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_ELEMENTWISE_H */
