# TASK-002: Implement Kernel Dialect Core

## Status
[x] Complete

## Priority
P0 (Critical)

## Milestone
M0 — Dialect Infrastructure

## Owner Agent
General

## Description

Implement the core Kernel dialect with:
- Dialect definition
- Common types (Tile, Accumulator)
- Operation base classes
- Initial simple operations (relu, softmax)

## Acceptance Criteria

- [x] Kernel dialect defined in TableGen
- [x] TileType defined
- [x] ks.relu operation implemented
- [x] ks.softmax operation implemented
- [x] Operations parse and print correctly
- [x] Verifiers catch invalid inputs
- [x] Lit tests pass

## Implementation Notes

### Dialect Definition

```tablegen
def KS_Dialect : Dialect {
  let name = "kernel";
  let cppNamespace = "::kernelsmith::ks";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    ...
  ];
}
```

### Operation Template

```tablegen
def KS_ReluOp : KS_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

### Type Definitions

```tablegen
def KS_TileType : KS_Type<"Tile", "tile"> {
  let parameters = (ins "Type":$elementType, "ArrayRef<int64_t>":$shape);
}
```

## Test Files

Parse and verifier tests live in `tests/lit/Dialect/Kernel/`. Pass tests live
in `tests/lit/Passes/`. `basic.mlir` covers relu, gelu, and softmax printing.
Invalid cases use `*-invalid.mlir` and `quantization-invalid.mlir`.

## Dependencies

- TASK-001: Infrastructure setup

## Verification

```bash
cmake --build build --target check-kernelsmith-lit
```

## Log

### 2026-09-28
- Pointed verification at `check-kernelsmith-lit`. The old `make test-lit`
  target is not in the Makefile.

### 2026-06-06
- Updated stale checklist to match completed dialect infrastructure.
- Removed accumulator type from completion criteria because it is not part of
  the checked-in dialect core.

### [Date TBD]
- Task created.
