# TASK-003: Implement Matrix Multiplication (matmul) Operation

## Status
[x] Complete

## Priority
P1 (High)

## Description

Implement the `ks.matmul` operation according to the specification in `specs/kernels/matmul.md`.

This is a foundational operation used in:
- Fully connected layers
- Attention mechanisms
- Many other neural network components

## Acceptance Criteria

- [ ] Operation defined in KernelOps.td
- [ ] Verifier validates shapes and types
- [ ] Parsing/printing works correctly
- [ ] Round-trip test passes
- [ ] Invalid input tests pass
- [ ] Batch matmul variant defined
- [ ] Documentation updated

## Implementation Notes

### Operation Definition

```tablegen
def KS_MatmulOp : KS_Op<"matmul", [Pure]> {
  let summary = "Matrix multiplication";
  let description = [{
    C = A × B where A is M×K and B is K×N, producing M×N output.
  }];
  
  let arguments = (ins
    AnyTensor:$lhs,    // M × K
    AnyTensor:$rhs,    // K × N
    OptionalAttr<I64ArrayAttr>:$tile_sizes
  );
  
  let results = (outs AnyTensor:$result);  // M × N
  
  let hasVerifier = 1;
}
```

### Verifier Logic

```cpp
LogicalResult MatmulOp::verify() {
  auto lhsType = getLhs().getType().cast<RankedTensorType>();
  auto rhsType = getRhs().getType().cast<RankedTensorType>();
  
  // Check ranks
  if (lhsType.getRank() != 2 || rhsType.getRank() != 2)
    return emitOpError("operands must be 2D tensors");
  
  // Check inner dimensions
  if (lhsType.getDimSize(1) != rhsType.getDimSize(0))
    return emitOpError("inner dimensions must match");
  
  // Verify output shape
  ...
}
```

## Test Cases

See `specs/kernels/matmul.md` for full test case list.

Key tests:
1. Square matrices: 64×64 × 64×64
2. Rectangular: 32×128 × 128×64
3. Small: 4×4 × 4×4
4. Non-power-of-2: 17×23 × 23×31

## Dependencies

- TASK-002: Kernel dialect core

## Specification

See: `specs/kernels/matmul.md`

## Verification

```bash
make test-lit TESTS=tests/lit/Dialect/Kernel/matmul.mlir
```

## Log

### [Date TBD]
- Task created
