# DES-004: Attention Kernel Implementation

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-06 |
| **Phase** | Phase 5 - Advanced Kernels |
| **Priority** | High |

---

## Context

### Problem Statement

Scaled Dot-Product Attention (SDPA) is critical for transformer models:

```
SDPA(Q, K, V) = softmax(Q·K^T / √d_k) · V
```

We need to:
1. Define `ks.attention` operation
2. Decompose into matmul + softmax + reduction operations
3. Optimize for sequence length and embedding dimension
4. Handle numerical stability (log-softmax, online normalization)

### Key Challenges

- **Multi-stage computation**: 3 separate matrix multiplications + reductions
- **Numerical stability**: Softmax requires careful handling
- **Memory bandwidth**: Bottleneck for large sequence lengths
- **Masking support**: Causal masking, padding masks

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Parse `ks.attention` operation | Must Have |
| REQ-2 | Verify Q, K, V shapes match | Must Have |
| REQ-3 | Decompose to matmul + softmax + matmul | Must Have |
| REQ-4 | Support optional causal masking | Should Have |
| REQ-5 | Numerically stable softmax | Must Have |
| REQ-6 | Performance target: 70%+ of reference | Should Have |
| REQ-7 | Multi-VLEN testing | Must Have |

---

## Design Overview

### Mathematical Decomposition

```
Input: Q (B, N_q, d), K (B, N_k, d), V (B, N_k, d_v)

Step 1: Compute attention scores
  S = Q · K^T                    // (B, N_q, N_k)
  S = S / √d

Step 2: Apply masking (optional)
  if causal_mask:
    S[i, j] = -inf for j > i

Step 3: Softmax normalization
  A = softmax(S, dim=-1)         // (B, N_q, N_k)

Step 4: Apply to values
  O = A · V                      // (B, N_q, d_v)

Output: O (B, N_q, d_v)
```

### Implementation Strategy

```
ks.attention %Q, %K, %V
  {mask_type = "causal"}
    ↓
Decompose to sub-operations:
  1. %S = matmul(Q, transpose(K))
  2. Scale: %S = S / sqrt(d)
  3. Mask application (if needed)
  4. %A = softmax(%S, dim=-1)
  5. %O = matmul(A, V)
    ↓
Optimize individual stages:
  - Stage 1 & 5: Use vectorized matmul (DES-002)
  - Stage 4: Vectorized softmax with numerical stability
    ↓
RVV Lowering (DES-001)
    ↓
RISC-V Assembly
```

### MLIR Operation Definition

```mlir
%output = ks.attention %Q, %K, %V
  {mask_type = "none"}
  : tensor<1x64x128xf32>,      // Q: (batch, seq_len, d_model)
    tensor<1x64x128xf32>,      // K: (batch, seq_len, d_model)
    tensor<1x64x128xf32>       // V: (batch, seq_len, d_model)
  -> tensor<1x64x128xf32>      // Output: (batch, seq_len, d_model)
```

### Components

#### 1. Attention Operation Definition

```cpp
class AttentionOp : public Op<AttentionOp> {
  // %output = ks.attention %Q, %K, %V
  //   {mask_type = "causal", scale = 1.0 / sqrt(d)}
  //   : tensor<...>, tensor<...>, tensor<...> -> tensor<...>

  StringAttr maskType;     // "none", "causal", "custom"
  FloatAttr scale;         // 1.0 / sqrt(embed_dim)
};
```

#### 2. Softmax Kernel

Numerically stable softmax:
```
softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x - max(x)))
```

This requires:
- Row-wise max reduction
- Broadcast subtraction
- Exponential
- Sum reduction
- Division

```mlir
// Vectorized softmax for attention scores
func.func @softmax_stable(%input: tensor<64x128xf32>) -> tensor<64x128xf32> {
  // Row-wise max
  %max = ks.reduce_max %input {axis = 1} : tensor<64xf32>

  // Subtract max (broadcasting)
  %centered = arith.subf %input, %max : tensor<64x128xf32>

  // Exponentiate
  %exp = math.exp %centered : tensor<64x128xf32>

  // Row-wise sum
  %sum = ks.reduce_sum %exp {axis = 1} : tensor<64xf32>

  // Divide (broadcasting)
  %result = arith.divf %exp, %sum : tensor<64x128xf32>

  return %result : tensor<64x128xf32>
}
```

#### 3. Masking Support

**Causal Mask** (for autoregressive models):
```mlir
// Apply -inf to upper triangle
%mask = arith.constant dense<...> : tensor<64x64xi1>  // Lower triangle = true
%masked_scores = scf.if %mask_type_is_causal {
  %inf = arith.constant 0x7f800000 : f32  // +inf in f32
  %result = scf.for %i = 0 to 64 iter_args %s = %scores {
    %updated = scf.for %j = 0 to 64 iter_args %s_row = %s {
      %should_mask = arith.cmpi gt, %j, %i
      %value = scf.if %should_mask { scf.yield %inf } else { scf.yield %s[i, j] }
      scf.yield %value
    }
    scf.yield %updated
  }
  scf.yield %result
}
```

---

## Test Strategy

### Tests Structure

```
tests/lit/Dialect/Kernel/attention_verifier.mlir
  ├─ Valid operations (various shapes)
  ├─ Invalid dimension mismatches
  └─ Masking options

tests/lit/Transforms/lower_attention_*.mlir
  ├─ Decompose to sub-operations
  ├─ Softmax lowering
  ├─ Mask application
  └─ RVV intrinsics

tests/unit/AttentionTest.cpp
  ├─ Operation parsing
  ├─ Shape verification
  └─ Mask type validation

tests/functional_validator.py
  ├─ Correctness vs reference (numpy/torch)
  ├─ Numerical stability (no NaN/inf leakage)
  ├─ Causal mask behavior
  └─ Different sequence lengths
```

### Test Cases

- [ ] Small attention (seq_len=8, d_model=64)
- [ ] Medium attention (seq_len=64, d_model=256)
- [ ] Large attention (seq_len=512, d_model=1024)
- [ ] Batch processing (batch=4)
- [ ] Causal masking enabled
- [ ] Causal masking disabled
- [ ] Mixed data types (f16, f32)

### Numerical Validation

Critical tests:
- [ ] Softmax outputs sum to 1.0 (within tolerance)
- [ ] No NaN or infinity in output
- [ ] Gradients (if needed) are stable
- [ ] Results match torch.nn.functional.scaled_dot_product_attention

---

## Implementation Plan

### Phase 1: Specification (Week 1)
- [ ] Define ks.attention operation with verifier
- [ ] Write all tests in RED phase
- [ ] Create functional test framework

### Phase 2: Decomposition (Week 1-2)
- [ ] Implement attention → matmul decomposition
- [ ] Implement softmax operation
- [ ] Handle masking

### Phase 3: Integration (Week 2)
- [ ] Integrate with MatMul kernel (DES-002)
- [ ] Add softmax-specific optimizations
- [ ] Handle reductions properly

### Phase 4: Validation (Week 3)
- [ ] Functional correctness testing
- [ ] Numerical stability verification
- [ ] Multi-VLEN testing
- [ ] Performance profiling

---

## Performance Considerations

### Bottlenecks

1. **Memory bandwidth**: Attention is memory-bound
   - Q·K^T creates large intermediate tensor
   - Softmax requires sequential row processing
   - A·V is memory-bound on output

2. **Register pressure**: Multiple tensor accesses
   - Keep Q, K, V in registers where possible
   - Use tiling to fit intermediate results

### Optimization Strategy

```
Cache blocking for attention:
- Tile over query dimension (outer loop)
- Tile over key dimension (reduce inner loop)
- Full reduction of value dimension
- Reuse computed softmax in memory-efficient way
```

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Numerical instability (softmax) | High | Medium | Careful implementation, extensive tests |
| Memory overflow (large seq_len) | High | Medium | Streaming softmax or block-wise softmax |
| Performance suboptimal | Medium | High | Profile and optimize tiling/blocking |
| Mask handling complex | Medium | Medium | Clear specification with examples |

---

## Success Criteria

- ✅ All lit tests passing
- ✅ All unit tests passing
- ✅ Functional validation passing (matches reference implementation)
- ✅ Numerical stability verified (no NaN/inf)
- ✅ Multi-VLEN testing (128, 256, 512)
- ✅ Performance acceptable (70%+ of reference)
- ✅ Handles edge cases (various dimensions, masking)

---

## Dependencies

- DES-001: Vector Operations Lowering
- DES-002: MatMul Kernel (for SDPA matmuls)
- Attention specification: `specs/kernels/attention.md`
- Reference implementation (numpy/torch for validation)

---

## Related Documents

- [Attention Specification](../../specs/kernels/attention.md)
- [MatMul Kernel Design](DES-002-matmul-kernel.md)
- [Vector Operations Lowering](DES-001-vector-operations-lowering.md)
- [Testing Guide](../guides/testing-guide.md)

---

## References

- Vaswani et al., "Attention is All You Need" (2017)
- Pytorch's scaled_dot_product_attention documentation
- Numerical stability in softmax: https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
