# ks.quantize / ks.dequantize - Quantization Specification

## Overview

Quantization converts floating-point tensors to integer tensors using affine
scale and zero-point metadata. Dequantization reconstructs floating-point values
from integer tensors with the same metadata.

These operations make quantization explicit in the KernelSmith IR before later
lowering to target-specific integer kernels.

## Operation Definitions

```mlir
%q = ks.quantize %x {scale = 0.03125 : f64, zero_point = 0 : i64}
     : tensor<...xf32> -> tensor<...xi8>

%x = ks.dequantize %q {scale = 0.03125 : f64, zero_point = 0 : i64}
     : tensor<...xi8> -> tensor<...xf32>

%acc = ks.dot_i8 %input, %weight
       {input_zero_point = 0 : i64, weight_zero_point = 0 : i64}
       : tensor<...xi8>, tensor<...xi8> -> tensor<i32>

%out = ks.matvec_i8 %input, %weights
       {input_zero_point = 0 : i64, weight_zero_point = 0 : i64}
       : tensor<...xi8>, tensor<...x...xi8> -> tensor<...xi32>

%acc = ks.dot_w4a8 %input, %packed_weight, %weight_scales
       {group_size = 64 : i64, input_scale = 0.03125 : f64,
        input_zero_point = 0 : i64, weight_zero_point = 0 : i64}
       : tensor<...xi8>, tensor<...xi8>, tensor<...xf32>
         -> tensor<f32>

%out = ks.matvec_w4a8 %input, %packed_weights, %weight_scales
       {group_size = 64 : i64, input_scale = 0.03125 : f64,
        input_zero_point = 0 : i64, weight_zero_point = 0 : i64}
       : tensor<...xi8>, tensor<...x...xi8>, tensor<...x...xf32>
         -> tensor<...xf32>
```

## Mathematical Definition

For quantization:

```
q[i] = clamp(round(x[i] / scale) + zero_point)
```

For dequantization:

```
x[i] = (q[i] - zero_point) * scale
```

For INT8 dot:

```
acc = sum_k ((input[k] - input_zero_point) *
             (weight[k] - weight_zero_point))
```

For INT8 GEMV:

```
output[row] = sum_col ((input[col] - input_zero_point) *
                       (weights[row, col] - weight_zero_point))
```

For W4A8 dot:

```
acc = sum_k (((input[k] - input_zero_point) * input_scale) *
             ((unpack_i4(packed_weight[k]) - weight_zero_point) *
              weight_scales[k / group_size]))
```

For W4A8 GEMV:

```
output[row] = sum_col (((input[col] - input_zero_point) * input_scale) *
                       ((unpack_i4(weights[row, col]) -
                         weight_zero_point) *
                        weight_scales[row, col / group_size]))
```

Rounding mode and saturation details are defined by future lowering passes. At
the dialect level, the operations carry semantic metadata and verify type and
shape compatibility.

## Input/Output Specification

### `ks.quantize`

| Name | Type | Description |
|------|------|-------------|
| `input` | ranked floating-point tensor | Source floating-point values |
| `output` | ranked integer tensor | Quantized values |

### `ks.dequantize`

| Name | Type | Description |
|------|------|-------------|
| `input` | ranked integer tensor | Quantized values |
| `output` | ranked floating-point tensor | Reconstructed values |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scale` | `f64` | required | Positive finite quantization scale |
| `zero_point` | `i64` | `0` | Integer offset in the quantized domain |

### `ks.dot_i8`

| Name | Type | Description |
|------|------|-------------|
| `input` | rank-1 `i8` tensor | Quantized activation vector |
| `weight` | rank-1 `i8` tensor | Quantized weight vector |
| `result` | rank-0 `i32` tensor | Exact signed accumulator |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_zero_point` | `i64` | `0` | Activation zero-point, must fit signed i8 |
| `weight_zero_point` | `i64` | `0` | Weight zero-point, must fit signed i8 |

### `ks.matvec_i8`

| Name | Type | Description |
|------|------|-------------|
| `input` | rank-1 `i8` tensor | Quantized activation vector of length `cols` |
| `weights` | rank-2 `i8` tensor | Row-major quantized weights `[rows, cols]` |
| `result` | rank-1 `i32` tensor | Exact signed accumulators, one per row |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `input_zero_point` | `i64` | `0` | Activation zero-point, must fit signed i8 |
| `weight_zero_point` | `i64` | `0` | Weight zero-point, must fit signed i8 |

### `ks.dot_w4a8`

| Name | Type | Description |
|------|------|-------------|
| `input` | rank-1 `i8` tensor | Quantized activation vector of length `K` |
| `packed_weight` | rank-1 signless `i8` tensor | Signed int4 weights packed two per byte, length `ceil(K / 2)` |
| `weight_scales` | rank-1 `f32` tensor | Per-group weight scales, length `ceil(K / group_size)` |
| `result` | rank-0 `f32` tensor | Fused dequantized accumulator |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `group_size` | `i64` | required | Number of input elements sharing one weight scale, must be positive |
| `input_scale` | `f64` | required | Positive finite activation scale |
| `input_zero_point` | `i64` | `0` | Activation zero-point, must fit signed i8 |
| `weight_zero_point` | `i64` | `0` | Packed weight zero-point, must fit signed i4 |

### `ks.matvec_w4a8`

| Name | Type | Description |
|------|------|-------------|
| `input` | rank-1 `i8` tensor | Quantized activation vector of length `cols` |
| `packed_weights` | rank-2 signless `i8` tensor | Row-major signed int4 weights packed two per byte, shape `[rows, ceil(cols / 2)]` |
| `weight_scales` | rank-2 `f32` tensor | Row-major per-group weight scales, shape `[rows, ceil(cols / group_size)]` |
| `result` | rank-1 `f32` tensor | Fused dequantized accumulators, one per row |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `group_size` | `i64` | required | Number of input elements sharing one weight scale, must be positive |
| `input_scale` | `f64` | required | Positive finite activation scale |
| `input_zero_point` | `i64` | `0` | Activation zero-point, must fit signed i8 |
| `weight_zero_point` | `i64` | `0` | Packed weight zero-point, must fit signed i4 |

## Verification Rules

1. Input and result must be ranked tensors.
2. Input and result shapes must match, allowing dynamic dimensions.
3. `ks.quantize` input element type must be floating-point.
4. `ks.quantize` result element type must be integer.
5. `ks.dequantize` input element type must be integer.
6. `ks.dequantize` result element type must be floating-point.
7. `scale` must be positive and finite.
8. `zero_point` must fit in the integer storage element type.
9. `ks.dot_i8` operands must be ranked 1D signless `i8` tensors with signed
   semantics.
10. `ks.dot_i8` operand shapes must match, allowing dynamic dimensions.
11. `ks.dot_i8` result must be a rank-0 tensor with i32 element type.
12. `ks.dot_i8` zero-point attributes must fit in signed i8.
13. `ks.matvec_i8` input must be a ranked 1D signless `i8` tensor with signed
    semantics.
14. `ks.matvec_i8` weights must be a ranked 2D signless `i8` tensor with
    signed semantics.
15. `ks.matvec_i8` input length must match the weights column dimension,
    allowing dynamic dimensions.
16. `ks.matvec_i8` result must be a rank-1 tensor with i32 element type and a
    length matching the weights row dimension, allowing dynamic dimensions.
17. `ks.matvec_i8` zero-point attributes must fit in signed i8.
18. `ks.dot_w4a8` input, packed weight, weight scales, and result must be ranked
    tensors.
19. `ks.dot_w4a8` input, packed weight, and weight scale operands must be
    rank-1 tensors.
20. `ks.dot_w4a8` input and packed weights must have signless i8 element type,
    and weight scales must have f32 element type.
21. `ks.dot_w4a8` result must be a rank-0 tensor with f32 element type.
22. `ks.dot_w4a8` packed weight length must equal `ceil(K / 2)` for static `K`.
23. `ks.dot_w4a8` weight scale length must equal `ceil(K / group_size)` for
    static `K`.
24. `ks.dot_w4a8` `group_size` must be positive and `input_scale` must be
    positive and finite.
25. `ks.dot_w4a8` input zero-point must fit signed i8, and weight zero-point
    must fit signed i4.
26. `ks.matvec_w4a8` input, packed weights, weight scales, and result must be
    ranked tensors.
27. `ks.matvec_w4a8` input must be rank-1, packed weights and weight scales
    rank-2, and result rank-1.
28. `ks.matvec_w4a8` input and packed weights must have signless i8 element
    type, and weight scales and result must have f32 element type.
29. `ks.matvec_w4a8` packed weight column dimension must equal
    `ceil(cols / 2)` for static `cols`.
30. `ks.matvec_w4a8` weight scale column dimension must equal
    `ceil(cols / group_size)` for static `cols`.
31. `ks.matvec_w4a8` weight scale and result row dimensions must match packed
    weight rows, allowing dynamic dimensions.
32. `ks.matvec_w4a8` `group_size` must be positive and `input_scale` must be
    positive and finite.
33. `ks.matvec_w4a8` input zero-point must fit signed i8, and weight zero-point
    must fit signed i4.

## Lowering Strategy

Initial lowering will target linalg or vector forms:

```mlir
%scaled = arith.divf %input, %scale
%rounded = math.roundeven %scaled
%shifted = arith.addi %rounded, %zero_point
%clamped = arith.minsi/arithi.maxsi %shifted, storage_bounds
```

RVV lowering can later fuse quantize/dequantize with dot, GEMV, and matmul
patterns to avoid materializing intermediate tensors.

`ks.dot_i8` lowers to a `linalg.generic` reduction with two rank-1 inputs and a
rank-0 i32 output. The body sign-extends each i8 operand to i32, subtracts the
zero-points, multiplies, and adds into the accumulator.

`ks.matvec_i8` lowers to a `linalg.generic` with one parallel row iterator and
one reduction column iterator. The input vector is indexed by column, the weight
matrix by `[row, column]`, and the output by row.

`ks.dot_w4a8` and `ks.matvec_w4a8` lower to fused `linalg.generic` reductions.
The linalg body extracts packed bytes, selects and sign-extends signed int4
nibbles, applies activation and weight scales, and accumulates in f32 without
materializing dequantized weights.

## Public C API Foundation

The first quantized C APIs prioritize batch-1 decode dot/GEMV kernels:

| Function | Inputs | Output | Accumulator | Notes |
|---|---|---|---|---|
| `ks_dot_i8` | int8 activations, int8 weights | `int32_t` scalar | i32 | Applies input and weight zero-points before accumulation. |
| `ks_matvec_i8` | int8 activation vector, row-major int8 weights | `int32_t[rows]` | i32 | Weight matrix layout is `[rows, cols]` with an element stride between rows. |
| `ks_dot_w4a8` | int8 activations, packed signed int4 weights | `float` scalar | f32 | Fuses activation scale, per-group weight scale, int4 unpack, and dot. |
| `ks_matvec_w4a8` | int8 activation vector, row-major packed int4 weights | `float[rows]` | f32 | Primary batch-1 transformer decode ABI. |

### INT8 Arithmetic

INT8 dot/GEMV accumulates exactly into signed 32-bit outputs:

```
acc += (input[k] - input_zero_point) * (weight[k] - weight_zero_point)
```

Both zero-points must fit in `int8_t`. Requantization is intentionally outside
the first ABI so lowering and validation can prove the accumulator contract
before adding output quantization policy.

### W4A8 Packed Weight Layout

> This layout is ratified as the versioned native quantized contract
> (`KS_QUANT_LAYOUT_VERSION = 1`) in
> [DES-015](../../docs/design/DES-015-quantized-layout-abi-and-integration-conversion-paths.md),
> which also specifies the offline conversion paths from GGML `Q4_0`/`Q8_0` and
> to IRON/MLIR-AIE tile panels.

W4A8 weights are signed int4 values stored in two's-complement form. In MLIR IR,
packed byte tensors use signless `i8` storage so arithmetic lowering can use
standard `arith` integer ops. The public C ABI stores the same bytes as
`uint8_t`. Two weights are packed per byte:

```
byte[k / 2] bits 0..3 = weight[k]     for even k
byte[k / 2] bits 4..7 = weight[k + 1] for odd k
```

Rows are stored independently. A packed row contains `ceil(cols / 2)` bytes.
For GEMV, `packed_stride_bytes` is the byte distance between packed rows.

W4A8 scales are row-major by scale group. A row contains
`ceil(cols / group_size)` f32 scales. For element `k`, the scale index is
`k / group_size`. The first ABI uses signed int4 weights; `weight_zero_point`
is supported for metadata completeness but is expected to be `0` for symmetric
W4A8 models.

The W4A8 mathematical contract is:

```
acc += ((input[k] - input_zero_point) * input_scale) *
       ((unpack_i4(weight[k]) - weight_zero_point) *
        weight_scales[k / group_size])
```

### Workspace and Alignment

The generic reference implementation returns zero workspace for all quantized
dot/GEMV workspace queries. Optimized target implementations may require
workspace later, but must expose that through the same query functions before
using caller-provided scratch memory.

## Test Cases

1. Parse and print `ks.quantize` for f32 to i8 tensors.
2. Parse and print `ks.dequantize` for i8 to f32 tensors.
3. Reject unranked tensors.
4. Reject non-floating quantize inputs.
5. Reject non-integer quantize results.
6. Reject non-integer dequantize inputs.
7. Reject non-floating dequantize results.
8. Reject shape mismatches.
9. Reject non-positive scale values.
10. Reject zero-points outside the storage range.
11. Parse and print `ks.dot_i8` for i8 vectors to an i32 scalar tensor.
12. Reject invalid `ks.dot_i8` ranks, element types, shapes, result types, and
    zero-points.
13. Parse and print `ks.matvec_i8` for an i8 vector and row-major i8 matrix to
    an i32 output vector.
14. Reject invalid `ks.matvec_i8` ranks, element types, shapes, result types,
    and zero-points.
15. Parse and print `ks.dot_w4a8` for an i8 activation vector, packed i4
    weights, f32 scales, and an f32 scalar tensor.
16. Reject invalid `ks.dot_w4a8` ranks, element types, packed-weight length,
    scale-group length, scale metadata, result type, and zero-points.
17. Parse and print `ks.matvec_w4a8` for an i8 activation vector, row-major
    packed i4 weights, f32 scales, and an f32 output vector.
18. Reject invalid `ks.matvec_w4a8` ranks, element types, packed-weight column
    length, scale-group shape, result shape, scale metadata, and zero-points.
