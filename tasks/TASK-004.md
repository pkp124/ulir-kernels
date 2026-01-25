# TASK-004: Implement Kernel Tiling Pass

## Status
[ ] Not Started

## Priority
P1 (High)

## Description

Implement transformation passes that tile kernel operations for efficient execution:

1. **TileKernelsPass**: Convert high-level ops to tiled loops
2. **VectorizeKernelsPass**: Convert tiled ops to vector operations

## Acceptance Criteria

- [ ] TileKernelsPass defined in Passes.td
- [ ] Pass tiles matmul operation
- [ ] Configurable tile sizes (via pass options or attributes)
- [ ] Tiled output is functionally equivalent
- [ ] VectorizeKernelsPass implemented
- [ ] Integration test with matmul passes
- [ ] Performance test shows expected tiling

## Implementation Notes

### Pass Declaration

```tablegen
def TileKernels : Pass<"tile-kernels", "func::FuncOp"> {
  let summary = "Tile kernel operations for efficient execution";
  let options = [
    ListOption<"tileSizes", "tile-sizes", "int64_t",
               "Tile sizes for each dimension">
  ];
}
```

### Tiling Strategy

For matmul M×K × K×N:

```mlir
// Before
%C = kernel.matmul %A, %B : tensor<M×K>, tensor<K×N> -> tensor<M×N>

// After (tiled)
scf.for %i = 0 to M step tile_M {
  scf.for %j = 0 to N step tile_N {
    %c_tile = tensor.extract_slice %C[%i, %j][tile_M, tile_N]
    %c_init = linalg.fill zeros into %c_tile
    
    scf.for %k = 0 to K step tile_K {
      %a_tile = tensor.extract_slice %A[%i, %k][tile_M, tile_K]
      %b_tile = tensor.extract_slice %B[%k, %j][tile_K, tile_N]
      %c_tile = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
    }
    
    tensor.insert_slice %c_tile into %C[%i, %j]
  }
}
```

### Vectorization

After tiling, vectorize the innermost computation:

```mlir
// After vectorization
%a_vec = vector.transfer_read %A[%i, %k]
%b_vec = vector.broadcast %B[%k, %j]
%c_vec = vector.fma %a_vec, %b_vec, %c_acc
```

## Test Cases

1. **Basic tiling**: Matmul with 64×64×64 tiles
2. **Non-divisible**: 100×100 with 64×64 tiles
3. **Different tile sizes**: Various configurations
4. **Correctness**: Compare tiled vs untiled output

## Dependencies

- TASK-003: Matmul operation

## Verification

```bash
# Transformation test
make test-lit TESTS=tests/lit/Transforms/tile-kernels.mlir

# Integration test
python tests/integration/test_tiled_matmul.py
```

## Log

### [Date TBD]
- Task created
