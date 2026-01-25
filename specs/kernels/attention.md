# Scaled Dot-Product Attention Specification

## Overview

Scaled dot-product attention is the core mechanism in transformer models.

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

## Operation Definition

```mlir
%output = kernel.scaled_dot_product_attention %Q, %K, %V
    {scale = 0.125}  // 1/sqrt(d_k) for d_k=64
    : tensor<B×L×D×dtype>, tensor<B×L×D×dtype>, tensor<B×L×D×dtype>
    -> tensor<B×L×D×dtype>

// With optional mask
%output = kernel.scaled_dot_product_attention %Q, %K, %V, mask = %mask
    : ... -> tensor<B×L×D×dtype>
```

## Mathematical Definition

```
# Step 1: Compute attention scores
scores[b, i, j] = Σ(d) Q[b, i, d] × K[b, j, d]

# Step 2: Scale
scaled_scores = scores / sqrt(d_k)

# Step 3: Apply mask (optional)
if mask:
    scaled_scores = where(mask, scaled_scores, -inf)

# Step 4: Softmax over key dimension
weights[b, i, j] = softmax(scaled_scores, axis=-1)

# Step 5: Weighted sum of values
output[b, i, d] = Σ(j) weights[b, i, j] × V[b, j, d]
```

## Input/Output Specification

### Inputs

| Name | Type | Description |
|------|------|-------------|
| `Q` | `tensor<B×L_q×D×dtype>` | Query tensor |
| `K` | `tensor<B×L_kv×D×dtype>` | Key tensor |
| `V` | `tensor<B×L_kv×D_v×dtype>` | Value tensor |
| `mask` | `tensor<B×L_q×L_kv×bool>` | Optional attention mask |

### Outputs

| Name | Type | Description |
|------|------|-------------|
| `output` | `tensor<B×L_q×D_v×dtype>` | Attention output |

### Attributes

| Name | Type | Default | Description |
|------|------|---------|-------------|
| `scale` | `f32` | `1/sqrt(D)` | Scaling factor for scores |

### Shape Relationships

- Query length `L_q` can differ from key/value length `L_kv`
- Key and Value must have same sequence length `L_kv`
- Output sequence length matches query length `L_q`
- Output feature dimension matches value dimension `D_v`

## Verification Rules

1. `Q.rank == 3` (batch, seq, dim)
2. `K.rank == 3`
3. `V.rank == 3`
4. `Q.shape[0] == K.shape[0] == V.shape[0]` (batch size)
5. `Q.shape[2] == K.shape[2]` (query/key dimension)
6. `K.shape[1] == V.shape[1]` (key/value sequence length)
7. If mask: `mask.shape == [B, L_q, L_kv]`

## Lowering Strategy

### Fused Implementation (Preferred)

For memory efficiency, compute attention in tiles:

```mlir
scf.for %b = 0 to B {
  scf.for %q_tile = 0 to L_q step TILE_Q {
    // Load Q tile
    %Q_tile = load Q[b, q_tile:q_tile+TILE_Q, :]
    
    // Initialize output accumulator
    %O_acc = zero : tensor<TILE_Q×D_v>
    %l_acc = zero : tensor<TILE_Q>  // normalizer
    %m_acc = -inf : tensor<TILE_Q>  // max for stable softmax
    
    scf.for %kv_tile = 0 to L_kv step TILE_KV {
      // Load K, V tiles
      %K_tile = load K[b, kv_tile:kv_tile+TILE_KV, :]
      %V_tile = load V[b, kv_tile:kv_tile+TILE_KV, :]
      
      // Compute scores: Q_tile @ K_tile^T
      %S = matmul(%Q_tile, transpose(%K_tile)) * scale
      
      // Online softmax update
      %m_new = max(%m_acc, row_max(%S))
      %P = exp(%S - %m_new)
      %l_new = exp(%m_acc - %m_new) * %l_acc + row_sum(%P)
      
      // Update output
      %O_acc = exp(%m_acc - %m_new) * %O_acc + %P @ %V_tile
      %m_acc = %m_new
      %l_acc = %l_new
    }
    
    // Normalize and store
    %O = %O_acc / %l_acc
    store %O to output[b, q_tile:q_tile+TILE_Q, :]
  }
}
```

This is the **FlashAttention** algorithm for O(N) memory usage.

### Unfused Implementation (Simpler)

For smaller sequences or debugging:

```mlir
// Step 1: QK^T
%scores = kernel.batch_matmul %Q, transpose(%K) : -> tensor<B×L×L>

// Step 2: Scale
%scaled = arith.mulf %scores, %scale

// Step 3: Mask (optional)
%masked = select(%mask, %scaled, -inf)

// Step 4: Softmax
%weights = kernel.softmax %masked {axis = -1}

// Step 5: Weighted sum
%output = kernel.batch_matmul %weights, %V
```

Memory usage: O(N²) for attention matrix.

## Performance Considerations

### Memory Bandwidth

- Unfused: `O(B × L² × D)` memory traffic
- Fused (FlashAttention): `O(B × L × D)` memory traffic

### Arithmetic Intensity

- QK^T: `2 × B × L × L × D` ops
- Softmax: `O(B × L × L)` ops
- AV: `2 × B × L × L × D` ops
- Total: `O(B × L² × D)` ops

### Tiling for RVV

- Tile Q along sequence dimension
- Vectorize over D (feature dimension)
- Use vector reductions for softmax

## Causal Masking

For autoregressive models, apply causal mask:

```
mask[i, j] = (j <= i)  // Can only attend to past
```

Optimization: Skip computation for masked positions.

## Multi-Head Attention

`kernel.multi_head_attention` combines:
1. Linear projections (Q, K, V from input)
2. Split into heads
3. Parallel attention per head
4. Concatenate and project output

```mlir
%output = kernel.multi_head_attention %input, %wq, %wk, %wv, %wo
    {num_heads = 8}
    : tensor<B×L×D>, ... -> tensor<B×L×D>
```

## Test Cases

### Basic Cases

1. **Small**: B=1, L=16, D=32
2. **Medium**: B=4, L=128, D=64
3. **Large**: B=8, L=512, D=128

### Mask Cases

1. **No mask**: Full attention
2. **Causal mask**: Lower triangular
3. **Padding mask**: Variable length sequences

### Numerical Cases

1. **Softmax stability**: Large score values
2. **Precision**: Long sequences

### Edge Cases

1. **Single token**: L=1
2. **Long sequence**: L=4096+
3. **Different Q/KV lengths**: Cross-attention

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [FlashAttention](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2](https://arxiv.org/abs/2307.08691)
