# 2D Convolution (conv2d) Specification

## Overview

2D convolution operation commonly used in CNNs for image processing.

## Operation Definition

```mlir
%output = kernel.conv2d %input, %filter 
    {strides = [sh, sw], padding = [pt, pb, pl, pr], dilation = [dh, dw]}
    : tensor<N×H×W×C_in×dtype>, tensor<KH×KW×C_in×C_out×dtype> 
    -> tensor<N×OH×OW×C_out×dtype>
```

## Data Formats

This specification uses **NHWC** for input and **HWIO** for filters:

- Input: `[batch, height, width, channels]`
- Filter: `[kernel_height, kernel_width, in_channels, out_channels]`
- Output: `[batch, out_height, out_width, out_channels]`

## Mathematical Definition

```
output[n, oh, ow, oc] = Σ(kh, kw, ic) 
    input[n, oh*sh + kh*dh - pt, ow*sw + kw*dw - pl, ic] × filter[kh, kw, ic, oc]
```

Where:
- `sh, sw`: Strides (height, width)
- `dh, dw`: Dilation (height, width)
- `pt, pb, pl, pr`: Padding (top, bottom, left, right)

## Output Shape Calculation

```
OH = floor((H + pt + pb - dh*(KH-1) - 1) / sh + 1)
OW = floor((W + pl + pr - dw*(KW-1) - 1) / sw + 1)
```

## Input/Output Specification

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `input` | `tensor<N×H×W×C×dtype>` | Input feature map (NHWC) |
| `filter` | `tensor<KH×KW×C×OC×dtype>` | Convolution filter (HWIO) |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | `tensor<N×OH×OW×OC×dtype>` | Output feature map |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `strides` | `array<i64, 2>` | `[1, 1]` | Stride in H and W |
| `padding` | `array<i64, 4>` | `[0,0,0,0]` | Padding: top, bottom, left, right |
| `dilation` | `array<i64, 2>` | `[1, 1]` | Dilation in H and W |

## Verification Rules

1. `input.rank == 4` (NHWC format)
2. `filter.rank == 4` (HWIO format)
3. `input.shape[3] == filter.shape[2]` (input channels match)
4. `strides[i] >= 1` for all i
5. `dilation[i] >= 1` for all i
6. `padding[i] >= 0` for all i
7. Output shape is valid (OH > 0, OW > 0)

## Lowering Strategy

### 1. Kernel → Linalg

```mlir
%output = linalg.conv_2d_nhwc_hwcf
    {strides = [...], dilations = [...]}
    ins(%padded_input, %filter)
    outs(%output_init)
```

### 2. Im2Col Transformation (Alternative)

For some targets, converting to matmul is more efficient:

```
1. Extract patches from input → [N*OH*OW, KH*KW*C_in]
2. Reshape filter → [KH*KW*C_in, C_out]
3. Matmul → [N*OH*OW, C_out]
4. Reshape → [N, OH, OW, C_out]
```

### 3. Direct Convolution with Tiling

```mlir
scf.for %n = 0 to N {
  scf.for %oh = 0 to OH step tile_oh {
    scf.for %ow = 0 to OW step tile_ow {
      scf.for %oc = 0 to OC step tile_oc {
        // Accumulate over filter
        scf.for %kh = 0 to KH {
          scf.for %kw = 0 to KW {
            scf.for %ic = 0 to C step tile_ic {
              // Vectorized inner computation
            }
          }
        }
      }
    }
  }
}
```

### 4. Vectorization for RVV

Vectorize over:
- Output channels (OC) - most common
- Input channels (IC) - with reduction
- Output width (OW) - spatial vectorization

## Special Cases

### Depthwise Convolution

When `C_out == C_in` and each filter operates on one channel:
- Use `kernel.depthwise_conv2d` instead
- No cross-channel computation

### 1×1 Convolution

When `KH == KW == 1` and `stride == 1`:
- Equivalent to pointwise operation
- Can be optimized as batched matmul

### Grouped Convolution

Not currently supported in base `conv2d`.
Future: Add `groups` attribute.

## Performance Model

### Arithmetic Intensity

- Operations: `2 × N × OH × OW × OC × KH × KW × C_in`
- Memory: Input + Filter + Output sizes
- Typically memory-bound for small kernels, compute-bound for large

### Tiling Strategy

1. **Tile output spatial (OH, OW)**: For cache locality
2. **Tile output channels (OC)**: For vectorization
3. **Tile input channels (IC)**: For register reuse

## Test Cases

### Basic Cases

1. **Simple**: 1×8×8×3, 3×3×3×16, stride=1, no padding
2. **With padding**: Same as above with padding=[1,1,1,1]
3. **Strided**: stride=[2,2]
4. **Dilated**: dilation=[2,2]

### Edge Cases

1. **1×1 kernel**: filter shape [1,1,C,OC]
2. **Large kernel**: filter shape [7,7,C,OC]
3. **Single channel**: C_in=1
4. **Non-square**: H≠W, KH≠KW

### Shape Cases

1. **Square input**: 32×32
2. **Non-square**: 28×14
3. **Large batch**: N=32
4. **Many channels**: C=512, OC=512

## References

- PyTorch conv2d documentation
- cuDNN convolution algorithms
- [Optimal Convolution Algorithms](https://arxiv.org/abs/1509.09308)
