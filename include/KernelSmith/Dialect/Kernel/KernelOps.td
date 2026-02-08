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
// Activation Functions
//===----------------------------------------------------------------------===//

def KS_ReLUOp : KS_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation: max(0, x)";

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_GELUOp : KS_Op<"gelu", [Pure, SameOperandsAndResultType]> {
  let summary = "GELU activation";

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<BoolAttr, "true">:$approximate
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_SiLUOp : KS_Op<"silu", [Pure, SameOperandsAndResultType]> {
  let summary = "SiLU/Swish activation: x × sigmoid(x)";

  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
}

def KS_SoftmaxOp : KS_Op<"softmax", [Pure, SameOperandsAndResultType]> {
  let summary = "Softmax activation";

  let arguments = (ins
    AnyTensor:$input,
    DefaultValuedAttr<I64Attr, "-1">:$axis
  );
  let results = (outs AnyTensor:$output);

  let assemblyFormat = "$input attr-dict `:` type($input)";
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
}

#endif // KERNELSMITH_DIALECT_KERNEL_KERNELOPS_TD
