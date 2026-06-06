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

## Verification Rules

1. Input and result must be ranked tensors.
2. Input and result shapes must match, allowing dynamic dimensions.
3. `ks.quantize` input element type must be floating-point.
4. `ks.quantize` result element type must be integer.
5. `ks.dequantize` input element type must be integer.
6. `ks.dequantize` result element type must be floating-point.
7. `scale` must be positive and finite.
8. `zero_point` must fit in the integer storage element type.

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
