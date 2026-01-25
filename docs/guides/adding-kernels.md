# Adding New Kernels

This guide explains how to add new kernel operations to the project.

## Overview

Adding a new kernel involves:

1. **Specification**: Define the kernel's semantics
2. **Operation Definition**: Add TableGen definition
3. **Verifier**: Implement verification logic
4. **Lowering**: Implement lowering passes
5. **Testing**: Add comprehensive tests
6. **Documentation**: Update docs

## Step 1: Create Specification

First, create a specification document:

```bash
make new-kernel NAME=my_kernel
```

Edit `specs/kernels/my_kernel.md`:

```markdown
# MyKernel Specification

## Overview
Brief description of what this kernel does.

## Mathematical Definition
```
output[i, j] = f(input[i, j], ...)
```

## Input/Output Specification

### Inputs
- `input`: Shape [N, C, H, W], element type f32/f16
- `weight`: Shape [OC, C, KH, KW], element type f32/f16

### Outputs
- `output`: Shape [N, OC, OH, OW], element type same as input

### Shape Relationships
- OH = (H + 2*pad_h - KH) / stride_h + 1
- OW = (W + 2*pad_w - KW) / stride_w + 1

### Attributes
- `strides`: [stride_h, stride_w], default [1, 1]
- `padding`: [top, bottom, left, right], default [0, 0, 0, 0]

## Lowering Strategy

1. Tile for L1 cache
2. Vectorize inner loops
3. Use FMA instructions

## Test Cases

1. Basic case: 1x1 kernel, no padding, stride 1
2. Larger kernel with padding
3. Non-square inputs
4. Edge cases: single element, empty batch
```

## Step 2: Add Operation Definition

Add to `src/dialects/kernel/KernelOps.td`:

```tablegen
def Kernel_MyKernelOp : Kernel_Op<"my_kernel", [Pure]> {
  let summary = "My kernel operation";
  let description = [{
    Detailed description of the operation.
    
    Example:
    ```mlir
    %out = kernel.my_kernel %input, %weight {strides = [1, 1]}
           : tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32> 
           -> tensor<1x64x26x26xf32>
    ```
  }];

  let arguments = (ins
    AnyTensor:$input,
    AnyTensor:$weight,
    DefaultValuedAttr<I64ArrayAttr, "{1, 1}">:$strides,
    DefaultValuedAttr<I64ArrayAttr, "{0, 0, 0, 0}">:$padding
  );

  let results = (outs AnyTensor:$output);

  let assemblyFormat = [{
    $input `,` $weight attr-dict 
    `:` type($input) `,` type($weight) `->` type($output)
  }];

  let hasVerifier = 1;
  let hasCanonicalizer = 1;  // Optional: if you need canonicalization
}
```

## Step 3: Implement Verifier

In `src/dialects/kernel/KernelOps.cpp`:

```cpp
LogicalResult MyKernelOp::verify() {
  auto inputType = getInput().getType().cast<RankedTensorType>();
  auto weightType = getWeight().getType().cast<RankedTensorType>();
  auto outputType = getOutput().getType().cast<RankedTensorType>();

  // Check ranks
  if (inputType.getRank() != 4)
    return emitOpError("input must be 4D tensor");
  
  if (weightType.getRank() != 4)
    return emitOpError("weight must be 4D tensor");

  // Check element types match
  if (inputType.getElementType() != weightType.getElementType())
    return emitOpError("input and weight must have same element type");

  // Check channel dimensions
  int64_t inputChannels = inputType.getDimSize(1);
  int64_t weightInputChannels = weightType.getDimSize(1);
  
  if (inputChannels != ShapedType::kDynamic &&
      weightInputChannels != ShapedType::kDynamic &&
      inputChannels != weightInputChannels) {
    return emitOpError("input channels (")
           << inputChannels << ") must match weight input channels ("
           << weightInputChannels << ")";
  }

  // Verify output shape
  // ... shape calculation and verification

  return success();
}
```

## Step 4: Implement Lowering

Create `src/passes/LowerMyKernel.cpp`:

```cpp
namespace {

struct LowerMyKernelPattern : public OpRewritePattern<kernel::MyKernelOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(kernel::MyKernelOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    
    // Get operands
    Value input = op.getInput();
    Value weight = op.getWeight();
    
    // Create output tensor
    auto outputType = op.getOutput().getType().cast<RankedTensorType>();
    Value output = rewriter.create<tensor::EmptyOp>(
        loc, outputType.getShape(), outputType.getElementType());
    
    // Create linalg operation
    // For convolution-like operations:
    SmallVector<AffineMap> indexingMaps = {
      // ... define affine maps
    };
    
    SmallVector<utils::IteratorType> iteratorTypes = {
      // ... define iterator types
    };
    
    auto linalgOp = rewriter.create<linalg::GenericOp>(
        loc, outputType, ValueRange{input, weight}, ValueRange{output},
        indexingMaps, iteratorTypes,
        [&](OpBuilder &b, Location loc, ValueRange args) {
          // ... body
        });
    
    rewriter.replaceOp(op, linalgOp.getResults());
    return success();
  }
};

struct LowerMyKernelPass 
    : public impl::LowerMyKernelBase<LowerMyKernelPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerMyKernelPattern>(&getContext());
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
```

## Step 5: Add Tests

### Lit Test for Parsing/Printing

`tests/lit/Dialect/Kernel/my_kernel.mlir`:

```mlir
// RUN: aikernel-opt %s | aikernel-opt | FileCheck %s

// CHECK-LABEL: func @test_my_kernel
func.func @test_my_kernel(%input: tensor<1x3x28x28xf32>, 
                          %weight: tensor<64x3x3x3xf32>) 
    -> tensor<1x64x26x26xf32> {
  // CHECK: kernel.my_kernel
  // CHECK-SAME: strides = [1, 1]
  %out = kernel.my_kernel %input, %weight {strides = [1, 1]}
         : tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32> 
         -> tensor<1x64x26x26xf32>
  return %out : tensor<1x64x26x26xf32>
}
```

### Lit Test for Verifier

`tests/lit/Dialect/Kernel/my_kernel_invalid.mlir`:

```mlir
// RUN: aikernel-opt %s -split-input-file -verify-diagnostics

func.func @test_wrong_rank(%input: tensor<28x28xf32>, 
                           %weight: tensor<64x3x3x3xf32>) {
  // expected-error @+1 {{input must be 4D tensor}}
  %out = kernel.my_kernel %input, %weight
         : tensor<28x28xf32>, tensor<64x3x3x3xf32> 
         -> tensor<1x64x26x26xf32>
  return
}

// -----

func.func @test_channel_mismatch(%input: tensor<1x3x28x28xf32>, 
                                 %weight: tensor<64x5x3x3xf32>) {
  // expected-error @+1 {{input channels (3) must match weight input channels (5)}}
  %out = kernel.my_kernel %input, %weight
         : tensor<1x3x28x28xf32>, tensor<64x5x3x3xf32> 
         -> tensor<1x64x26x26xf32>
  return
}
```

### Lit Test for Lowering

`tests/lit/Transforms/lower-my-kernel.mlir`:

```mlir
// RUN: aikernel-opt %s --lower-my-kernel | FileCheck %s

// CHECK-LABEL: func @test_lower_my_kernel
func.func @test_lower_my_kernel(%input: tensor<1x3x28x28xf32>, 
                                %weight: tensor<64x3x3x3xf32>) 
    -> tensor<1x64x26x26xf32> {
  // CHECK-NOT: kernel.my_kernel
  // CHECK: linalg.generic
  %out = kernel.my_kernel %input, %weight {strides = [1, 1]}
         : tensor<1x3x28x28xf32>, tensor<64x3x3x3xf32> 
         -> tensor<1x64x26x26xf32>
  return %out : tensor<1x64x26x26xf32>
}
```

## Step 6: Update Documentation

1. Add to kernel table in `README.md`
2. Add API documentation in `docs/api/`
3. Create example in `examples/`

## Checklist

- [ ] Specification complete and reviewed
- [ ] Operation defined in TableGen
- [ ] Verifier implemented and tested
- [ ] Parsing/printing round-trips correctly
- [ ] Lowering pass implemented
- [ ] All lit tests passing
- [ ] Documentation updated
- [ ] Example added
