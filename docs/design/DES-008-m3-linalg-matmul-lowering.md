# DES-008: M3 — MatMul Lowering to Linalg (Generic Target)

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Approved |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-17 |
| **Milestone** | M3 — MLIR Lowering: MatMul (Generic Target) |
| **Implements** | `--ks-lower-to-linalg`, `--ks-tile` |
| **Depends On** | DES-006 (library architecture), M2 (pass infrastructure) |

---

## Context

### Problem Statement

`ks.matmul` has no lowering path. The pass infrastructure exists (`--ks-lower-activations`
is implemented and working), but matrix multiplication remains at the KS dialect level
with no route to executable code. This is the single highest-priority gap for M3.

This design covers two passes that together convert `ks.matmul` into cache-efficient
tiled `linalg.matmul` loops on the generic target (no SIMD, no packing):

```
ks.matmul
  |  --ks-lower-to-linalg
linalg.matmul (zero-initialized accumulator)
  |  --ks-tile
scf.for (M) / scf.for (N) / scf.for (K)
  linalg.matmul [tile_m × tile_n × tile_k]
  |  one-shot-bufferize + --convert-linalg-to-loops + --convert-to-llvm
LLVM IR → libkernelsmith.a
```

### Background

- M2 is complete: `KSLowerActivationsPass` proves the pass infrastructure end-to-end.
- DES-006 establishes the library-first approach: the MLIR pipeline replaces the
  handwritten `ks_matmul_f32` reference implementation behind the stable C API.
- The generic target profile (`target/generic.h`) specifies conservative tile sizes
  (single-level, no packing, portable C).
- Previous design DES-002 sketched the full RVV pipeline; this doc focuses exclusively
  on M3 (generic target). Packing, vectorization, and RVV are M4 concerns.

### Scope

**In scope**:
- `KSLowerToLinalgPass` (`--ks-lower-to-linalg`): `ks.matmul` → `linalg.matmul`
- `KSTilePass` (`--ks-tile`): tile `linalg.matmul` with profile-driven sizes via `scf.for`
- Static and dynamic shape support for both passes
- Lit tests for each pass in isolation and chained

**Out of scope**:
- B-matrix packing (`--ks-pack`) — M4
- Vectorization (`--ks-vectorize`) — M4
- RVV lowering (`--ks-lower-to-rvv`) — M4
- Bufferization pipeline — separate integration concern
- `ks.batch_matmul`, `ks.conv2d` lowering — M6

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | `ks.matmul` lowers to `linalg.matmul` with zero-initialized accumulator | Must Have |
| REQ-2 | Dynamic shapes (one or both dims `?`) are handled correctly | Must Have |
| REQ-3 | Tile sizes are configurable via pass CLI options, not hard-coded | Must Have |
| REQ-4 | Three nested `scf.for` loops (M, N, K) with correct slice/insert | Must Have |
| REQ-5 | Tile size `0` on any dimension means "no tiling" for that dimension | Must Have |
| REQ-6 | Non-divisible dimensions produce correct tail-handling (via `affine.min`) | Must Have |
| REQ-7 | `--ks-lower-to-linalg` and `--ks-tile` are independently composable passes | Must Have |
| REQ-8 | Lit tests: parse/print, transformation (before→after), and invalid-input | Must Have |
| REQ-9 | Default tile sizes are compatible with the generic target profile values | Should Have |
| REQ-10 | Pass does not insert any `memref.alloc` (all tensors via SSA iter_args) | Must Have |

---

## Design

### Architecture Overview

```
ks.matmul %A, %B : tensor<MxKxf32>, tensor<KxNxf32> -> tensor<MxNxf32>
           │
           │  KSLowerToLinalgPass
           ▼
%init = tensor.empty() : tensor<MxNxf32>
%zero = arith.constant 0.0 : f32
%acc  = linalg.fill ins(%zero) outs(%init) : f32, tensor<MxNxf32> -> tensor<MxNxf32>
%C    = linalg.matmul ins(%A, %B) outs(%acc)
           │
           │  KSTilePass (tile_m=64, tile_n=64, tile_k=256)
           ▼
%C = scf.for %m = 0 to M step 64 iter_args(%c0 = %acc) -> tensor<MxNxf32> {
  %C1 = scf.for %n = 0 to N step 64 iter_args(%c1 = %c0) -> tensor<MxNxf32> {
    %C2 = scf.for %k = 0 to K step 256 iter_args(%c2 = %c1) -> tensor<MxNxf32> {
      %a_tile = tensor.extract_slice %A[%m, %k][tile_m, tile_k][1,1]
      %b_tile = tensor.extract_slice %B[%k, %n][tile_k, tile_n][1,1]
      %c_tile = tensor.extract_slice %c2[%m, %n][tile_m, tile_n][1,1]
      %c_tile_new = linalg.matmul ins(%a_tile, %b_tile) outs(%c_tile)
      %c2_new = tensor.insert_slice %c_tile_new into %c2[%m, %n][tile_m, tile_n][1,1]
      scf.yield %c2_new
    }
    scf.yield %C2
  }
  scf.yield %C1
}
```

Tail handling (non-divisible dimensions) is automatic via `scf::tileUsingSCF`, which
inserts `affine.min` for the last tile boundary.

### Component Design

#### Pass 1: `KSLowerToLinalgPass` (`--ks-lower-to-linalg`)

**Approach**: `OpRewritePattern<ks::MatmulOp>` — consistent with the existing
`LowerActivationsPass.cpp` infrastructure.

**Why named `linalg.matmul` (not `linalg.generic`)**:

`linalg.matmul` is a first-class named structured op. The TilingInterface, vectorization
transform, and packing transforms (`linalg::packMatmulGreedily`) all operate on named ops
by convention. Using `linalg.generic` with an equivalent affine map works but loses
structural information that downstream transforms rely on. Activations correctly use
`linalg.generic` (element-wise — no named op exists); matmul should use `linalg.matmul`.

**Zero-initialized accumulator**:

`linalg.matmul` semantics: `C += A × B`. The outs operand is the initial value of C.
For `ks.matmul %A, %B → C` semantics (`C = A × B`, not `C += A × B`), we must
initialize outs to zero. Pattern:

```
%empty  = tensor.empty() : tensor<MxNxf32>
%zero   = arith.constant 0.0 : f32
%filled = linalg.fill ins(%zero : f32) outs(%empty : tensor<MxNxf32>) -> tensor<MxNxf32>
%C      = linalg.matmul ins(%A, %B) outs(%filled)
```

**Dynamic shape handling**:

The result type of `ks.matmul %A[M,K], %B[K,N]` is `tensor<MxNxf32>`. If M or N is `?`,
we extract the runtime size:
- M = `tensor.dim %A, 0` (when result dim 0 is dynamic)
- N = `tensor.dim %B, 1` (when result dim 1 is dynamic)

These are passed as dynamic sizes to `tensor.empty`.

#### Pass 2: `KSTilePass` (`--ks-tile`)

**Approach**: Walk `linalg::MatmulOp`, apply `scf::tileUsingSCF` (MLIR 20 TilingInterface API).

**Why `scf::tileUsingSCF` over `linalg::tileLinalgOp`**:

| | `linalg::tileLinalgOp` | `scf::tileUsingSCF` |
|---|---|---|
| API status | Deprecated in MLIR 18+ | Recommended, stable in MLIR 20 |
| Works with | `linalg::LinalgOp` subclasses | Any `TilingInterface` op |
| Tail handling | Manual | Automatic (`affine.min`) |
| Future M4 compatibility | Must migrate later | Already the right API for RVV |

`linalg::MatmulOp` implements `TilingInterface`, so `scf::tileUsingSCF` works directly.

**Pass options** (CLI-configurable, read by build script from target profile):

```
--ks-tile="tile-size-m=64 tile-size-n=64 tile-size-k=256"
```

Default values match `target/generic.h` (conservative, no SIMD). M4 will read
`target/riscv_rvv_256.h` values instead.

**Zero tile size semantics**: A tile size of 0 on a dimension means "do not tile that
dimension" — the loop is not emitted, and the full dimension is processed in one shot.
This matches MLIR's standard convention for partial tiling.

**Op collection before transformation**: Walk collects `linalg::MatmulOp` into a
`SmallVector` before iterating, to avoid IR invalidation during in-place rewriting.

### Pass CLI Interface

```bash
# Lower ks.matmul → linalg.matmul (zero-init accumulator)
ks-opt input.mlir --ks-lower-to-linalg

# Lower + tile (with profile-driven sizes)
ks-opt input.mlir \
  --ks-lower-to-linalg \
  "--ks-tile=tile-size-m=64 tile-size-n=64 tile-size-k=256"

# Default tile sizes (generic profile: 32×32×64)
ks-opt input.mlir --ks-lower-to-linalg --ks-tile
```

### Data Flow

```
ks.matmul %A, %B → KSLowerToLinalgPass → linalg.matmul (zero-init) →
    KSTilePass → scf.for[m] / scf.for[n] / scf.for[k] { linalg.matmul tile } →
    (future M4) --ks-pack → tensor.pack B + linalg.mmt4d →
    (future M4) --ks-vectorize → vector.contract →
    (future M4) --ks-lower-to-rvv → RVV intrinsics →
    --convert-linalg-to-loops --convert-to-llvm → LLVM IR
```

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| `linalg.generic` for matmul | Full flexibility | Loses named-op tiling support; M4 packing requires named op | Rejected |
| Transform dialect (`transform.structured.tile`) | User-visible scheduling, optimal for tuning | Adds scheduling IR complexity to M3; better for M4+ | Deferred to M4 |
| `linalg::tileLinalgOp` | Simpler API, already known | Deprecated in MLIR 18+; blocks M4 migration | Rejected |
| Hard-coded tile sizes | Simpler pass | Breaks profile-driven build; inflexible for M4 | Rejected |
| Accumulator type attribute on `ks.matmul` | Enables mixed-precision (f16→f32) | Out of M3 scope; tracked for M5 | Deferred to M5 |

### Rationale for Chosen Approach

Named `linalg.matmul` + `scf::tileUsingSCF` is the pattern used by IREE's CPU codegen
pipeline and matches the community direction in MLIR 20. It minimizes technical debt
between M3 and M4 (when packing and RVV vectorization are added). The
`OpRewritePattern`-based lowering mirrors the existing `LowerActivationsPass.cpp`,
keeping the pass infrastructure consistent and familiar.

---

## Test Strategy

### Lit Tests

#### `tests/lit/Passes/lower-to-linalg.mlir`
- `@test_static`: square matmul, static shapes → linalg.matmul + fill present
- `@test_rectangular`: non-square static → correct result type
- `@test_dynamic_m`: M is dynamic → tensor.dim + tensor.empty
- `@test_dynamic_both`: both M and N dynamic
- `@test_f16`: f16 element type → correct zero fill type
- `@test_no_ks_matmul`: CHECK-NOT ks.matmul (completely eliminated)

#### `tests/lit/Passes/tile.mlir`
- `@test_tile_static`: static shapes, tile_m=4, tile_n=4, tile_k=8 → 3× scf.for
- `@test_tile_partial_k`: tile_k=0 → only 2 scf.for (no K loop)
- `@test_tile_dynamic`: dynamic shapes → scf.for with dynamic bounds + affine.min
- `@test_pipeline`: `--ks-lower-to-linalg --ks-tile` chained: no ks.matmul, has scf.for

### Edge Cases
- [ ] Non-divisible dimensions (M=33, tile_m=8) — tail handled by affine.min
- [ ] Single-element matrix (1×1)
- [ ] All tile sizes = 0 (no tiling, single linalg.matmul)
- [ ] K-only tiling (tile_m=0, tile_n=0, tile_k=32)

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| `scf::tileUsingSCF` API differs from expected | High | Low | API verified against MLIR 20 headers |
| `scf.for` iter_arg semantics break accumulation | High | Low | Zero-fill init ensures correct identity for += |
| Tile size 0 not handled by scf::tileUsingSCF | Medium | Low | MLIR convention: 0 = no loop; verified in lit test |
| Dynamic shape `tensor.dim` placement | Medium | Low | Uses same pattern as LowerActivationsPass helper |
| Build: missing MLIRSCFTransforms link | Medium | Medium | Add to CMakeLists.txt alongside MLIRSCFDialect |

---

## Dependencies

- M2 (`KSLowerActivationsPass`): pass infrastructure proven end-to-end — complete.
- `mlir/Dialect/SCF/Transforms/TileUsingInterface.h`: MLIR 20 stable.
- `target/generic.h`: default tile sizes (KS_MATMUL_TILE_M/N/K defines).
- `linalg.matmul` named op: in MLIR upstream, stable since MLIR 14.

---

## Implementation Plan

### Phase 1: Pass infrastructure and lowering (this PR)
- [x] Design doc DES-008
- [ ] `Passes.td`: add `KSLowerToLinalgPass`, `KSTilePass` with options
- [ ] `LowerToLinalgPass.cpp`: `MatmulToLinalgPattern` with zero-fill + dynamic shape
- [ ] `TilePass.cpp`: walk + `scf::tileUsingSCF` with option-driven sizes
- [ ] `CMakeLists.txt`: add source files + `MLIRSCFDialect`, `MLIRSCFTransforms`

### Phase 2: Tests
- [ ] `tests/lit/Passes/lower-to-linalg.mlir`
- [ ] `tests/lit/Passes/tile.mlir`

### Phase 3: Integration (follow-up)
- [ ] Build script: read `target/generic.h` defines → `--ks-tile` options
- [ ] Replace handwritten `ks_matmul_f32` with MLIR-generated `.o`
- [ ] Verify C API tests still pass (same numerical results)

---

## Review History

(To be filled during review)
