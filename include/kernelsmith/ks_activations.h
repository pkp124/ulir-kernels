/*
 * KernelSmith Activation Functions
 *
 * Element-wise activation functions: relu, gelu, silu.
 * All operate on contiguous f32 arrays. Pure C99 API.
 *
 * In-place operation: output may alias input for all activation functions.
 *
 * Thread safety: All functions are reentrant. No global state.
 */

#ifndef KERNELSMITH_ACTIVATIONS_H
#define KERNELSMITH_ACTIVATIONS_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_relu_f32 - Rectified Linear Unit: output[i] = max(0, input[i])
 *
 * Parameters:
 *   input   - Input array of n elements
 *   output  - Output array of n elements (may alias input)
 *   n       - Number of elements
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer
 */
int ks_relu_f32(const float *input, float *output, size_t n);

/*
 * ks_gelu_f32 - Gaussian Error Linear Unit
 *
 * output[i] = 0.5 * x * (1 + erf(x / sqrt(2)))
 *
 * Uses the exact formulation (not the tanh approximation).
 *
 * Parameters:
 *   input   - Input array of n elements
 *   output  - Output array of n elements (may alias input)
 *   n       - Number of elements
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer
 */
int ks_gelu_f32(const float *input, float *output, size_t n);

/*
 * ks_silu_f32 - Sigmoid Linear Unit (SiLU / Swish)
 *
 * output[i] = x * sigmoid(x) = x / (1 + exp(-x))
 *
 * Parameters:
 *   input   - Input array of n elements
 *   output  - Output array of n elements (may alias input)
 *   n       - Number of elements
 *
 * Returns:
 *   KS_OK              - Success
 *   KS_ERR_INVALID_ARG - NULL pointer
 */
int ks_silu_f32(const float *input, float *output, size_t n);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_ACTIVATIONS_H */
