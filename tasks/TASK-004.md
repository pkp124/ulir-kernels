# TASK-004: Implement Kernel Tiling and Vectorization Passes

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M3/M4 — Generic MatMul Lowering and RISC-V RVV Target

## Owner Agent
`.cursor/agents/mlir-pass-agent.md`

## Description

Implement transformation passes that tile and vectorize kernel operations for
efficient execution:

1. **TileKernelsPass**: Convert high-level ops to tiled loops
2. **VectorizeKernelsPass**: Convert tiled ops to vector operations

## Acceptance Criteria

- [x] TileKernelsPass defined in Passes.td as `--ks-tile`
- [x] Pass tiles linalg.matmul operations after `--ks-lower-to-linalg`
- [x] Configurable tile sizes via pass options
- [x] Tiled output has lit coverage for expected IR structure and tail handling
- [x] VectorizeKernelsPass implemented as `--ks-vectorize`
- [x] Integration test with matmul lowering and tiling passes
- [x] Performance benchmark follow-up moved to `TASK-006`

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
%C = ks.matmul %A, %B : tensor<M×K>, tensor<K×N> -> tensor<M×N>

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
ctest --test-dir build -R kernelsmith-lit --output-on-failure
ctest --test-dir build --output-on-failure
```

## Notes

- `--ks-tile` and `--ks-vectorize` are implemented and covered by lit tests in
  `tests/lit/Passes/tile.mlir` and `tests/lit/Passes/vectorize.mlir`.
- Benchmarking and hardware-oriented RVV validation are tracked separately in
  `TASK-006`.

## Log

### 2026-06-06
- Marked task complete to match the implemented `--ks-tile` and
  `--ks-vectorize` passes.
- Moved performance validation into `TASK-006`.

### 2026-05-31
- Updated status to reflect the implemented `--ks-tile` pass and remaining
  vectorization/performance work.
