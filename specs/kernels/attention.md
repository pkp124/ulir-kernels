# ks.attention - Scaled Dot-Product Attention Specification

## Overview

Scaled dot-product attention mechanism used in transformers.

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

## Operation Definition

```mlir
%output = ks.attention %Q, %K, %V
    : tensor<B×L×D×dtype>, tensor<B×L×D×dtype>, tensor<B×L×D×dtype>
    -> tensor<B×L×D×dtype>

// With optional mask
%output = ks.attention %Q, %K, %V, mask = %mask
    : ... -> tensor<B×L×D×dtype>
```

## Mathematical Definition

```
scores[b, i, j] = Σ(d) Q[b, i, d] × K[b, j, d]
scaled_scores = scores / sqrt(d_k)
weights = softmax(scaled_scores, axis=-1)
output[b, i, d] = Σ(j) weights[b, i, j] × V[b, j, d]
```

## Input/Output Specification

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `query` | `tensor<B×L_q×D×dtype>` | Query tensor |
| `key` | `tensor<B×L_kv×D×dtype>` | Key tensor |
| `value` | `tensor<B×L_kv×D_v×dtype>` | Value tensor |
| `mask` | `tensor<B×L_q×L_kv×bool>` | Optional attention mask |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | `tensor<B×L_q×D_v×dtype>` | Attention output |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scale` | `f32` | `1/sqrt(D)` | Scaling factor |

## Verification Rules

1. Query, key, value must be 3D tensors
2. Batch dimensions must match
3. Query and key feature dimensions must match
4. Key and value sequence lengths must match

## Lowering Strategy

### Fused Implementation (FlashAttention-style)

For memory efficiency, compute in tiles with online softmax.

### Unfused Implementation

```mlir
%scores = ks.batch_matmul %Q, transpose(%K)
%scaled = arith.mulf %scores, %scale
%weights = ks.softmax %scaled {axis = -1}
%output = ks.batch_matmul %weights, %V
```

## Test Cases

1. Small: B=1, L=16, D=32
2. Medium: B=4, L=128, D=64
3. Large: B=8, L=512, D=128
4. With causal mask
5. Cross-attention (different Q/KV lengths)
