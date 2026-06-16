/*
 * KernelSmith Normalization Functions
 *
 * Contiguous f32 normalization helpers for transformer blocks.
 *
 * Thread safety: All functions are reentrant. No global state.
 */

#ifndef KERNELSMITH_NORMALIZATION_H
#define KERNELSMITH_NORMALIZATION_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_rms_norm_f32 - RMSNorm over each row of an outer x inner tensor.
 *
 * output[row, col] = input[row, col] * weight[col] /
 *                    sqrt(mean(input[row, :]^2) + eps)
 *
 * Accumulation is f32 and reduction order is fixed from col 0 to inner - 1.
 * output may alias input, but must not alias weight.
 *
 * Parameters:
 *   input   - Input array with outer * inner elements, row-major
 *   weight  - Scale array with inner elements
 *   output  - Output array with outer * inner elements
 *   outer   - Number of rows
 *   inner   - Number of columns normalized per row; must be non-zero
 *   eps     - Positive finite epsilon added before rsqrt
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer, inner == 0, or invalid eps
 */
int ks_rms_norm_f32(const float *input, const float *weight, float *output,
                    size_t outer, size_t inner, float eps);

/*
 * ks_softmax_f32 - Numerically stable softmax over each row.
 *
 * Uses max-subtract-exp-sum-divide over the inner dimension.
 * output must not alias input.
 *
 * Parameters:
 *   input   - Input array with outer * inner elements, row-major
 *   output  - Output array with outer * inner elements
 *   outer   - Number of rows
 *   inner   - Number of columns normalized per row; must be non-zero
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer, output == input, or inner == 0
 */
int ks_softmax_f32(const float *input, float *output, size_t outer,
                   size_t inner);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_NORMALIZATION_H */
