/*
 * KernelSmith Quantized Dot/GEMV Kernels
 *
 * Public C99 API for INT8 and W4A8 dot products and matrix-vector products.
 * These functions are the stable ABI that generated RVV kernels will replace.
 *
 * Layout summary:
 *   i8 weights: row-major int8_t matrices, shape rows x cols.
 *   W4A8 weights: signed int4 values packed two per byte, low nibble first.
 *   W4A8 scales: row-major scale groups, shape rows x ceil(cols / group_size).
 *
 * Thread safety: All functions are reentrant. No global state.
 */

#ifndef KERNELSMITH_QUANTIZED_H
#define KERNELSMITH_QUANTIZED_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_dot_i8 - INT8 dot product with i32 accumulation.
 *
 * Computes:
 *   output = sum_k ((input[k] - input_zero_point) *
 *                   (weight[k] - weight_zero_point))
 *
 * Zero points must fit in int8_t. The output pointer receives the i32
 * accumulator directly; requantization is handled by later APIs/lowering.
 */
int ks_dot_i8(const int8_t *input, const int8_t *weight, size_t k,
              int32_t input_zero_point, int32_t weight_zero_point,
              int32_t *output, void *workspace, size_t ws_size);

/*
 * ks_matvec_i8 - INT8 matrix-vector product with i32 outputs.
 *
 * Weights are row-major with shape rows x cols and row stride
 * weight_stride elements. Computes one i32 accumulator per row.
 */
int ks_matvec_i8(const int8_t *input, const int8_t *weights,
                 size_t weight_stride, int32_t *output, size_t rows,
                 size_t cols, int32_t input_zero_point,
                 int32_t weight_zero_point, void *workspace, size_t ws_size);

/*
 * ks_dot_w4a8 - W4A8 dot product with fused dequantization.
 *
 * Packed weights store signed int4 values in two's-complement form:
 *   byte bit layout: [high k+1 nibble][low k nibble]
 *   low nibble holds even k, high nibble holds odd k.
 *
 * Each weight group uses one f32 scale:
 *   group index = k / group_size
 *
 * Computes:
 *   output = sum_k ((input[k] - input_zero_point) * input_scale) *
 *                  ((unpack_i4(weight[k]) - weight_zero_point) *
 *                   weight_scales[k / group_size])
 *
 * The first W4A8 ABI uses signed int4 weights and therefore typically uses
 * weight_zero_point = 0.
 */
int ks_dot_w4a8(const int8_t *input, const uint8_t *packed_weight,
                const float *weight_scales, size_t k, size_t group_size,
                float input_scale, int32_t input_zero_point,
                int32_t weight_zero_point, float *output, void *workspace,
                size_t ws_size);

/*
 * ks_matvec_w4a8 - W4A8 matrix-vector product with f32 outputs.
 *
 * packed_weights is row-major over packed rows. packed_stride_bytes is the
 * byte distance between rows and must be at least ceil(cols / 2).
 *
 * weight_scales is row-major over scale groups. scale_stride is the number of
 * scale values between rows and must be at least ceil(cols / group_size).
 */
int ks_matvec_w4a8(const int8_t *input, const uint8_t *packed_weights,
                   size_t packed_stride_bytes, const float *weight_scales,
                   size_t scale_stride, float *output, size_t rows,
                   size_t cols, size_t group_size, float input_scale,
                   int32_t input_zero_point, int32_t weight_zero_point,
                   void *workspace, size_t ws_size);

size_t ks_dot_i8_workspace(size_t k);
size_t ks_matvec_i8_workspace(size_t rows, size_t cols);
size_t ks_dot_w4a8_workspace(size_t k, size_t group_size);
size_t ks_matvec_w4a8_workspace(size_t rows, size_t cols, size_t group_size);
size_t ks_quantized_alignment(void);

/* Layout helpers for callers and packers. */
size_t ks_w4a8_packed_row_bytes(size_t cols);
size_t ks_w4a8_scale_groups(size_t cols, size_t group_size);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_QUANTIZED_H */
