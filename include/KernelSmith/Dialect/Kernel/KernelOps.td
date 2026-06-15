//===----------------------------------------------------------------------===//
// KernelSmith Kernel Operations
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_DIALECT_KERNEL_KERNELOPS_TD
#define KERNELSMITH_DIALECT_KERNEL_KERNELOPS_TD

include "KernelSmith/Dialect/Kernel/KernelDialect.td"
include "KernelSmith/Dialect/Kernel/KernelTypes.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

//===----------------------------------------------------------------------===//
// Type Constraints
//===----------------------------------------------------------------------===//

def KS_FloatTensor : TensorOf<[F16, BF16, F32, F64]>;

//===----------------------------------------------------------------------===//
// Matrix Operations
//===----------------------------------------------------------------------===//

def KS_MatmulOp : KS_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication: C = A × B";
  let description = [{
    Performs matrix multiplication of two 2D tensors.

    Example:
    ```mlir
    %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                            -> tensor<64x256xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs,
    OptionalAttr<I64ArrayAttr>:$tile_sizes
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;
}

def KS_BatchMatmulOp : KS_Op<"batch_matmul", [Pure]> {
  let summary = "Batched matrix multiplication";

  let arguments = (ins
    AnyTensor:$lhs,
    AnyTensor:$rhs
  );

  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Convolution Operations
//===----------------------------------------------------------------------===//

def KS_Conv2DOp : KS_Op<"conv2d", [Pure]> {
  let summary = "2D convolution (NHWC format)";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$filter,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{0, 0, 0, 0}">:$padding,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$dilation
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $filter attr-dict `:` type($input) `,` type($filter) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Attention Operations
//===----------------------------------------------------------------------===//

def KS_ScaledDotProductAttentionOp : KS_Op<"attention", [Pure]> {
  let summary = "Scaled dot-product attention";
  let description = [{
    Computes: softmax(Q × K^T / sqrt(d_k)) × V
  }];

  let arguments = (ins
    AnyTensor:$query,
    AnyTensor:$key,
    AnyTensor:$value,
    Optional<AnyTensor>:$mask,
    DefaultValuedAttr<F32Attr, "1.0">:$scale
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $query `,` $key `,` $value (`,` `mask` `=` $mask^ `:` type($mask))? attr-dict
    `:` type($query) `,` type($key) `,` type($value) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Elementwise Operations
//===----------------------------------------------------------------------===//

def KS_AddOp : KS_Op<"add", [Pure]> {
  let summary = "Elementwise floating-point addition with broadcasting";
  let description = [{
    Computes elementwise floating-point addition with NumPy-style broadcasting.

    Example:
    ```mlir
    %z = ks.add %x, %y : tensor<4x8xf32>, tensor<8xf32>
                         -> tensor<4x8xf32>
    ```
  }];

  let arguments = (ins
    KS_FloatTensor:$lhs,
    KS_FloatTensor:$rhs
  );
  let results = (outs KS_FloatTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

def KS_MulOp : KS_Op<"mul", [Pure]> {
  let summary = "Elementwise floating-point multiplication with broadcasting";
  let description = [{
    Computes elementwise floating-point multiplication with NumPy-style
    broadcasting.

    Example:
    ```mlir
    %z = ks.mul %x, %y : tensor<4x8xf32>, tensor<8xf32>
                         -> tensor<4x8xf32>
    ```
  }];

  let arguments = (ins
    KS_FloatTensor:$lhs,
    KS_FloatTensor:$rhs
  );
  let results = (outs KS_FloatTensor:$result);

  let assemblyFormat = [{
    $lhs `,` $rhs attr-dict `:` type($lhs) `,` type($rhs) `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Activation Functions
//===----------------------------------------------------------------------===//

def KS_ReLUOp : KS_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation: max(0, x)";

  let arguments = (ins KS_FloatTensor:$input);
  let results = (outs KS_FloatTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_GELUOp : KS_Op<"gelu", [Pure, SameOperandsAndResultType]> {
  let summary = "GELU activation";

  let arguments = (ins
    KS_FloatTensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$approximate
  );
  let results = (outs KS_FloatTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_SiLUOp : KS_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "SiLU/Swish activation: x × sigmoid(x)";

  let arguments = (ins KS_FloatTensor:$input);
  let results = (outs KS_FloatTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_SoftmaxOp : KS_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax activation";

  let arguments = (ins
    KS_FloatTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  let results = (outs KS_FloatTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Quantization Operations
//===----------------------------------------------------------------------===//

def KS_QuantizeOp : KS_Op<"quantize", [Pure]> {
  let summary = "Affine quantization from floating-point to integer tensor";
  let description = [{
    Converts a floating-point tensor to an integer tensor using scale and
    zero-point metadata.
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$scale,
    DefaultValuedAttr<I64Attr, "0">:$zero_point
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
}

def KS_DequantizeOp : KS_Op<"dequantize", [Pure]> {
  let summary = "Affine dequantization from integer to floating-point tensor";
  let description = [{
    Converts an integer tensor to a floating-point tensor using scale and
    zero-point metadata.
  }];

  let arguments = (ins
    AnyTensor:$input,
    F64Attr:$scale,
    DefaultValuedAttr<I64Attr, "0">:$zero_point
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
}

def KS_DotI8Op : KS_Op<"dot_i8", [Pure]> {
  let summary = "INT8 dot product with i32 accumulation";
  let description = [{
    Computes a quantized INT8 dot product with signed i32 accumulation:

      acc += (input[k] - input_zero_point) *
             (weight[k] - weight_zero_point)

    The result is a rank-0 tensor carrying the exact accumulator value.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    DefaultValuedAttr<I64Attr, "0">:$input_zero_point,
    DefaultValuedAttr<I64Attr, "0">:$weight_zero_point
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` type($input) `,` type($weight)
    `->` type($result)
  }];

  let hasVerifier = 1;
}

def KS_DotW4A8Op : KS_Op<"dot_w4a8", [Pure]> {
  let summary = "W4A8 dot product with fused dequantization";
  let description = [{
    Computes a mixed-precision dot product over signed INT4 packed weights and
    signed INT8 activations with f32 accumulation:

      acc += ((input[k] - input_zero_point) * input_scale) *
             ((unpack_i4(packed_weight[k]) - weight_zero_point) *
              weight_scales[k / group_size])

    Packed weights store two signed INT4 values per byte, with even k in the
    low nibble and odd k in the high nibble. The result is a rank-0 f32 tensor.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$packed_weight,
    AnyTensor:$weight_scales,
    I64Attr:$group_size,
    F64Attr:$input_scale,
    DefaultValuedAttr<I64Attr, "0">:$input_zero_point,
    DefaultValuedAttr<I64Attr, "0">:$weight_zero_point
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $packed_weight `,` $weight_scales attr-dict `:`
    type($input) `,` type($packed_weight) `,` type($weight_scales)
    `->` type($result)
  }];

  let hasVerifier = 1;
}

def KS_MatvecW4A8Op : KS_Op<"matvec_w4a8", [Pure]> {
  let summary = "W4A8 matrix-vector product with fused dequantization";
  let description = [{
    Computes a mixed-precision matrix-vector product over row-major signed INT4
    packed weights and signed INT8 activations with f32 accumulation:

      output[row] += ((input[col] - input_zero_point) * input_scale) *
                     ((unpack_i4(packed_weights[row, col]) -
                       weight_zero_point) *
                      weight_scales[row, col / group_size])

    Packed weight rows store two signed INT4 values per byte, with even columns
    in the low nibble and odd columns in the high nibble. The result is a rank-1
    f32 tensor carrying one fused accumulator per row.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$packed_weights,
    AnyTensor:$weight_scales,
    I64Attr:$group_size,
    F64Attr:$input_scale,
    DefaultValuedAttr<I64Attr, "0">:$input_zero_point,
    DefaultValuedAttr<I64Attr, "0">:$weight_zero_point
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $packed_weights `,` $weight_scales attr-dict `:`
    type($input) `,` type($packed_weights) `,` type($weight_scales)
    `->` type($result)
  }];

  let hasVerifier = 1;
}

def KS_MatvecI8Op : KS_Op<"matvec_i8", [Pure]> {
  let summary = "INT8 matrix-vector product with i32 accumulation";
  let description = [{
    Computes a quantized INT8 matrix-vector product with signed i32
    accumulation:

      output[row] += (input[col] - input_zero_point) *
                     (weights[row, col] - weight_zero_point)

    Weights are row-major with shape rows x cols. The result is a rank-1 i32
    tensor carrying one exact accumulator per row.
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weights,
    DefaultValuedAttr<I64Attr, "0">:$input_zero_point,
    DefaultValuedAttr<I64Attr, "0">:$weight_zero_point
  );
  let results = (outs AnyTensor:$result);

  let assemblyFormat = [{
    $input `,` $weights attr-dict `:` type($input) `,` type($weights)
    `->` type($result)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Normalization Operations
//===----------------------------------------------------------------------===//

def KS_LayerNormOp : KS_Op<"layer_norm", [Pure]> {
  let summary = "Layer normalization";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    AnyTensor:$bias,
    DefaultValuedAttr<F32Attr, "1e-5">:$eps,
    I64ArrayAttr:$normalized_shape
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight `,` $bias attr-dict 
    `:` type($input) `,` type($weight) `,` type($bias) `->` type($output)
  }];

  let hasVerifier = 1;
}

def KS_RMSNormOp : KS_Op<"rms_norm", [Pure]> {
  let summary = "RMS normalization";

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    DefaultValuedAttr<F32Attr, "1e-5">:$eps
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight attr-dict `:` type($input) `,` type($weight) `->` type($output)
  }];

  let hasVerifier = 1;
}

//===----------------------------------------------------------------------===//
// Reduction Operations
//===----------------------------------------------------------------------===//

def KS_ReduceSumOp : KS_Op<"reduce_sum", [Pure]> {
  let summary = "Sum reduction";

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
}

def KS_ReduceMaxOp : KS_Op<"reduce_max", [Pure]> {
  let summary = "Max reduction";

  let arguments = (ins
    AnyTensor:$input,
    I64ArrayAttr:$axes,
    DefaultValuedAttr<BoolAttr, "false">:$keepdims
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input attr-dict `:` type($input) `->` type($output)
  }];

  let hasVerifier = 1;
}

#endif // KERNELSMITH_DIALECT_KERNEL_KERNELOPS_TD
