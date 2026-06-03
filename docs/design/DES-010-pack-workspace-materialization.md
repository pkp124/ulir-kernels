# DES-010: Pack Workspace Materialization

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Author** | KernelSmith team |
| **Created** | 2026-06-03 |
| **Related** | DES-006, DES-008, DES-009 |

## Context

The M4 RVV path introduces `linalg.pack` for matmul B-panel packing so the
micro-kernel can use unit-stride vector loads. The initial tensor-level pack
representation was useful for expressing the layout, but it left ownership of
the packed storage to one-shot bufferization.

KernelSmith's runtime design requires the opposite: kernels must not allocate
internally. Temporary packed panels must live in caller-provided workspace.

## Requirements

From `specs/kernels/matmul.md`, `specs/targets/riscv-rvv.md`, and DES-006:

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Preserve B layout `[K, N] -> [N/NR, K, NR]` for RVV unit-stride loads. | Must Have |
| REQ-2 | Use explicit `memref<?xi8>` workspace for packed buffers. | Must Have |
| REQ-3 | Avoid hidden `memref.alloc` for packed panels. | Must Have |
| REQ-4 | Keep C in logical row-major layout unless a future ABI requires packed C. | Should Have |
| REQ-5 | Reject unsupported dynamic/layout cases with diagnostics. | Must Have |

## Design

`--ks-pack` now packs only the B operand and rewrites matmul to a
`linalg.generic` that indexes C directly:

```mlir
%Bp = linalg.pack %B inner_dims_pos = [1] inner_tiles = [NR]
    outer_dims_perm = [1, 0]
    : tensor<KxNxf32> -> tensor<N/NRxKxNRxf32>

%D = linalg.generic
  indexing_maps = [
    (m, np, k, nr) -> (m, k),
    (m, np, k, nr) -> (np, k, nr),
    (m, np, k, nr) -> (m, np * NR + nr)
  ]
  ins(%A, %Bp) outs(%C)
```

`--ks-materialize-pack-workspace` then rewrites supported tensor packs to use
explicit workspace:

```mlir
%offset = arith.constant 0 : index
%packed = memref.view %workspace[%offset][]
    : memref<?xi8> to memref<N/NRxKxNRxf32>
%src = bufferization.to_buffer %B
linalg.pack %src ... into %packed
%packed_tensor = bufferization.to_tensor %packed restrict
```

The bridge back to tensor IR lets the existing tensor pipeline continue until
the final one-shot bufferization boundary.

## Supported Scope

The first implementation intentionally supports only the current KernelSmith
matmul B-panel layout:

- static source/result shapes
- source rank 2, packed rank 3
- `inner_dims_pos = [1]`
- `outer_dims_perm = [1, 0]`
- one positive static tile
- explicit `memref<?xi8>` workspace function argument

Dynamic shapes, padding, multiple packed operands, and workspace size runtime
checks are follow-up work.

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Let one-shot bufferization allocate packed tensors | Simple | Violates no-internal-allocation runtime design | Rejected |
| Lower `linalg.pack` directly to loops before workspace selection | Removes pack earlier | Loses layout intent and still needs storage ownership | Rejected |
| Pack and unpack C | Symmetric packed IR | Extra copies/workspace; C row tiles are already contiguous across NR | Rejected |
| Materialize B pack into explicit workspace | Aligns with BLAS practice and KernelSmith runtime model | Requires conservative layout checks | Selected |

## Test Strategy

- `tests/lit/Passes/pack.mlir`
  - verifies B pack and packed `linalg.generic`
  - verifies C unpack is not introduced
- `tests/lit/Passes/materialize-pack-workspace.mlir`
  - verifies `memref.view` from `memref<?xi8>` workspace
  - verifies memref `linalg.pack` into workspace
  - verifies no `memref.alloc`
- `tests/lit/Passes/materialize-pack-workspace-invalid.mlir`
  - missing workspace diagnostic
  - unsupported layout diagnostic
  - dynamic shape diagnostic

## Future Work

- Workspace ABI integration with generated C kernels.
- Runtime workspace size checks when `memref<?xi8>` extent is statically or
  dynamically available.
- Panel-by-panel workspace reuse after L2 tiling.
- Dynamic tail handling via padding or masking.
