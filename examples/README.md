# KernelSmith Examples

This directory contains example kernels demonstrating KernelSmith usage.

## Examples

### Basic Operations

```mlir
// examples/basic.mlir
func.func @relu_example(%input: tensor<1024xf32>) -> tensor<1024xf32> {
  %output = ks.relu %input : tensor<1024xf32>
  return %output : tensor<1024xf32>
}
```

### Matrix Multiplication

```mlir
// examples/matmul.mlir
func.func @gemm(%A: tensor<64x128xf32>, 
                %B: tensor<128x256xf32>) -> tensor<64x256xf32> {
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
```

### Attention

```mlir
// examples/attention.mlir
func.func @attention(%Q: tensor<8x128x64xf32>,
                     %K: tensor<8x128x64xf32>,
                     %V: tensor<8x128x64xf32>) -> tensor<8x128x64xf32> {
  %out = ks.attention %Q, %K, %V 
         : tensor<8x128x64xf32>, tensor<8x128x64xf32>, tensor<8x128x64xf32>
         -> tensor<8x128x64xf32>
  return %out : tensor<8x128x64xf32>
}
```

## Running Examples

```bash
# Parse and print
ks-opt examples/matmul.mlir

# Lower to linalg (when implemented)
ks-opt examples/matmul.mlir --ks-lower-to-linalg

# Full pipeline to RVV (when implemented)
ks-opt examples/matmul.mlir \
  --ks-lower-to-linalg \
  --ks-tile \
  --ks-vectorize \
  --ks-lower-to-rvv
```
