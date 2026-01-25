# TASK-002: Implement Kernel Dialect Core

## Status
[ ] Not Started

## Priority
P0 (Critical)

## Description

Implement the core Kernel dialect with:
- Dialect definition
- Common types (Tile, Accumulator)
- Operation base classes
- Initial simple operations (relu, softmax)

## Acceptance Criteria

- [ ] Kernel dialect defined in TableGen
- [ ] TileType and AccumulatorType defined
- [ ] kernel.relu operation implemented
- [ ] kernel.softmax operation implemented
- [ ] Operations parse and print correctly
- [ ] Verifiers catch invalid inputs
- [ ] Lit tests pass

## Implementation Notes

### Dialect Definition

```tablegen
def Kernel_Dialect : Dialect {
  let name = "kernel";
  let cppNamespace = "::aikernel::kernel";
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    ...
  ];
}
```

### Operation Template

```tablegen
def Kernel_ReLUOp : Kernel_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

### Type Definitions

```tablegen
def Kernel_TileType : Kernel_Type<"Tile", "tile"> {
  let parameters = (ins "Type":$elementType, "ArrayRef<int64_t>":$shape);
}
```

## Test Files

```
tests/lit/Dialect/Kernel/
├── basic.mlir          # Basic parsing/printing
├── relu.mlir           # ReLU operation tests
├── softmax.mlir        # Softmax operation tests
└── invalid.mlir        # Verifier error tests
```

## Dependencies

- TASK-001: Infrastructure setup

## Verification

```bash
make test-lit TESTS=tests/lit/Dialect/Kernel/
```

## Log

### [Date TBD]
- Task created
