# DES-009: Milestone 4 — RISC-V RVV Lowering Pipeline

| Field    | Value |
|----------|-------|
| ID       | DES-009 |
| Status   | Draft |
| Date     | 2026-03-16 |
| Authors  | KernelSmith team |
| Related  | DES-001, DES-008, ROADMAP.md §Milestone 4 |

## Overview

This document describes the lowering pipeline for Milestone 4: generating
optimized matrix multiplication code for RISC-V vector extension (RVV) targets
from KernelSmith's MLIR dialect representation.

The pipeline transforms `ks.matmul` all the way to LLVM IR that, when compiled
with `llc -march=riscv64 -mattr=+v`, emits RVV instructions
(`vle32.v`, `vfmul.vv`/reductions or `vfmacc.vv`, `vse32.v`, etc.).

---

## 1. Motivation

Milestone 3 (DES-008) established the generic target path:
```
ks.matmul → linalg.matmul → tiled scf.for → (handwritten C)
```

For RISC-V edge SoCs (T-Head C908/C910, SiFive X280, Kendryte K230), a
hand-written or LLVM-autovectorized scalar loop leaves significant performance
on the table. RVV's variable-length vector registers (VLEN=128–16384 bits)
require:

1. **Tiling** to match register tile to VLMAX (elements per vector group).
2. **B-matrix packing** to eliminate strided memory accesses in the inner loop.
3. **Explicit vectorization** to produce vector transfer/arithmetic/reduction
   ops that the RISC-V backend can lower to RVV instructions.
4. **Full lowering to LLVM dialect** so that `mlir-translate` + `llc` can
   target `riscv64` with V extension features.

---

## 2. Target Profile

`target/riscv_rvv_256.h` defines the tile constants for VLEN=256, f32, LMUL=4:

| Parameter      | Value | Formula |
|---------------|-------|---------|
| VLEN           | 256   | hardware baseline |
| SEW            | 32    | f32 element width |
| LMUL           | 4     | register grouping (compute-bound) |
| VLMAX          | 32    | `VLEN * LMUL / SEW = 256*4/32` |
| L2 tile M      | 128   | fits A-panel + C-tile in 512KB L2 |
| L2 tile N      | 128   | |
| L2 tile K      | 256   | |
| Register MR    | 16    | micro-kernel rows |
| Register NR    | 32    | micro-kernel cols = VLMAX |
| Pack factor    | 32    | = NR (column panel width) |

---

## 3. Pass Pipeline

### 3.1 Full Pipeline Invocation

```bash
ks-opt input.mlir \
  --ks-lower-to-linalg \
  "--ks-tile=tile-size-m=128 tile-size-n=128 tile-size-k=256" \
  "--ks-pack=pack-factor=32" \
  "--ks-tile=tile-size-m=16 tile-size-n=32 tile-size-k=0" \
  --ks-vectorize \
  --ks-lower-to-rvv \
  -o lowered.mlir
```

Or use the driver script:
```bash
scripts/compile-rvv.sh kernel.mlir kernel.o
```

### 3.2 Stage-by-Stage Transformations

#### Stage 1: `--ks-lower-to-linalg` (from M3)
```
ks.matmul %A[M,K], %B[K,N] → linalg.fill(0) + linalg.matmul
```

#### Stage 2: `--ks-tile` (L2 tile, from M3)
Tile `linalg.matmul` into L2-sized `scf.for` loops:
```
scf.for %m = 0 to M step 128 {
  scf.for %n = 0 to N step 128 {
    scf.for %k = 0 to K step 256 {
      linalg.matmul  ins(%A_tile[128,256], %B_tile[256,128]) outs(%C_tile[128,128])
    }
  }
}
```

#### Stage 3: `--ks-pack` (NEW in M4)
Pack the B operand into column-panel layout `[N/NR, K, NR]`:
```
%Bp = linalg.pack %B inner_dims_pos=[1] inner_tiles=[32]
        : tensor<256x128xf32> → tensor<4x256x32xf32>
linalg.generic {packed GEMM indexing}
  ins(%A, %Bp) outs(%Cp) { C[np,m,nr] += A[m,k] * B[np,k,nr] }
%C_out = linalg.unpack %Cp ...
```

**Why pack B?** In the unpacked layout, column access in the inner loop strides
by `N * sizeof(f32)` — cache-unfriendly for large N. Packing reorders B into
`NR`-wide panels so the inner loop's vector loads are unit-stride.

#### Stage 4: `--ks-tile` (register tile)
Tile the packed `linalg.generic` at micro-kernel size (MR×NR = 16×32):
```
scf.for %m = 0 to 128 step 16 {
  scf.for %np = 0 to 4 step 1 {   # NR panels (NR=32 fixed)
    linalg.generic[16, 256, 32]    # inner: statically-shaped micro-kernel
  }
}
```

#### Stage 5: `--ks-vectorize` (NEW in M4)
`linalg.vectorize` converts the inner statically-shaped `linalg.generic` /
`linalg.matmul` into vector dialect:
```
%a = vector.transfer_read %A[m, k]   : tensor<16x256xf32>, vector<16x256xf32>
%b = vector.transfer_read %Bp[np,k,0]: tensor<4x256x32xf32>, vector<256x32xf32>
%c = vector.transfer_read %Cp[np,m,0]: tensor<4x16x32xf32>, vector<16x32xf32>
%d = vector.contract {indexing_maps=...} %a, %b, %c : ... into vector<16x32xf32>
vector.transfer_write %d, %Cp[np,m,0]
```

MLIR 21 may represent matmul vectorization as multiply plus `vector.multi_reduction`; the LLVM RVV backend lowers the resulting vector operations to RVV instructions.

#### Current integration note
The checked-in lower-to-rvv smoke test currently validates `ks.matmul -> linalg -> tile -> vectorize -> LLVM`. `--ks-pack` has independent lit coverage; full `linalg.pack`/`linalg.unpack` bufferization into the RVV lowering pipeline remains follow-up work.

#### Stage 6: `--ks-lower-to-rvv` (NEW in M4)
Full lowering from vector/tensor/scf to LLVM dialect via a nested PassManager:

| Sub-stage | Conversion |
|-----------|-----------|
| one-shot-bufferize | tensor → memref (function-boundary buffers) |
| convert-linalg-to-loops | remaining `linalg.generic` → `scf.for` |
| lower-affine | `affine.apply` → `arith` |
| convert-scf-to-cf | `scf.for/if` → `cf.br` (flat CFG) |
| convert-vector-to-llvm | `vector.*` → `llvm.*` with SIMD semantics |
| finalize-memref-to-llvm | `memref.*` → `llvm.*` (GEPs) |
| convert-arith-to-llvm | `arith.*` → `llvm.*` |
| convert-func-to-llvm | `func.func` → `llvm.func` |
| reconcile-unrealized-casts | clean up cast chains |

After this stage, `mlir-translate --mlir-to-llvmir` produces LLVM IR.

#### Stage 7: `llc` (external)
```bash
llc -march=riscv64 -mattr=+v,+zve64d,+zvl256b \
    -float-abi=hard -filetype=obj module.ll -o module.o
```

The LLVM RISC-V backend emits:
- `vle32.v` for unit-stride vector loads (from `vector.transfer_read`)
- `vfmul.vv` plus vector reductions, or `vfmacc.vv` where LLVM combines the pattern
- `vse32.v` for vector stores (from `vector.transfer_write`)
- `vsetvli` is inserted by the backend, minimized across loop bodies

---

## 4. B-Matrix Packing Design

### 4.1 Layout Transformation

```
Before packing — B[K, N] (row-major):
  B[0][0]  B[0][1]  ... B[0][NR-1]  B[0][NR]  ...
  B[1][0]  B[1][1]  ... B[1][NR-1]  B[1][NR]  ...
  ...
  B[K-1][0] ...

After packing — Bp[N/NR, K, NR] (panel-major):
  Panel 0: B[0..K-1][0..NR-1]   (all K rows of first NR columns, contiguous)
  Panel 1: B[0..K-1][NR..2NR-1] (next NR columns)
  ...
```

### 4.2 Workspace Size

The packed B buffer requires `K * N * sizeof(f32)` bytes — same total as
unpackaged B, but laid out differently. The workspace query in the C API
(`ks_matmul_f32_workspace_size`) accounts for this.

### 4.3 Limitations (M4)
- Only static shapes are supported (dynamic shapes deferred to M5+).
- N must be divisible by `pack-factor`; padding to the next multiple is the
  caller's responsibility (or use `--ks-pad` when available).

---

## 5. Vectorization Design

### 5.1 Vector Size Selection

The inner tile N dimension (NR=32) equals VLMAX for VLEN=256/LMUL=4/f32.
This ensures:
- No tail-handling loops at the vector register level.
- `vector.transfer_read/write` of width 32 map to single `vle32.v`/`vse32.v`.
- `vector.contract` of shape `[16, 256, 32]` → multiple `vfmacc.vv` with
  the LLVM backend unrolling the M dimension (16 accumulators).

### 5.2 LMUL Strategy

Using LMUL=4 for compute-bound matmul:
- 4× the register width → fewer `vsetvl` transitions.
- The LLVM backend handles LMUL selection based on vector element count.
- For memory-bound ops (relu, softmax), LMUL=1 is preferred (configured via
  linalg vectorization on smaller tile sizes).

---

## 6. Testing Strategy

### 6.1 Lit tests (MLIR level)
- `tests/lit/Passes/pack.mlir` — verify `linalg.pack` shape and indexing maps.
- `tests/lit/Passes/vectorize.mlir` — verify vector operations are produced and original linalg ops are replaced.
- `tests/lit/Passes/lower-to-rvv.mlir` — verify LLVM dialect output for the lower/tile/vectorize path.

### 6.2 QEMU correctness tests
```bash
# Compile RVV matmul kernel
scripts/compile-rvv.sh tests/inputs/matmul_128x256x128.mlir \
    build-rvv/bin/test_matmul_rvv

# Run on multiple VLENs
python tests/qemu_runner.py \
    --binary build-rvv/bin/test_matmul_rvv \
    --vlens 256 512
```

Expected: supported baseline VLENs produce `PASS` with identical output (bit-for-bit). VLEN=128 requires a separate non-`zvl256b` profile/build.

### 6.3 Benchmark
```bash
python tests/qemu_runner.py \
    --binary build-rvv/bin/bench_matmul_rvv \
    --vlens 256 --benchmark
```

Expected speedup over generic (scalar) target: ≥4× for 128×256×128 matmul
on QEMU VLEN=256 (LMUL=4, 32 f32 elements/cycle).

---

## 7. Cross-Compilation Setup

### 7.1 Required tools
- `ks-opt` — built from this repo (`cmake --build build --parallel`)
- `mlir-translate` — from LLVM 21 install (`/usr/lib/llvm-21/bin/mlir-translate`)
- `llc` — from LLVM 21 install with RISC-V target enabled
- `qemu-riscv64` — user-mode QEMU (`apt install qemu-user`)
- `riscv64-unknown-linux-gnu-gcc` — for linking C test harness (optional)

### 7.2 Check LLVM has RISC-V target
```bash
llc --version | grep RISCV
```

### 7.3 Driver script
`scripts/compile-rvv.sh` wraps all three stages. Set `BUILD_DIR`, `MLIR_TRANSLATE`,
or `LLC` environment variables to override defaults.

---

## 8. Alternatives Considered

| Alternative | Why rejected |
|-------------|-------------|
| LLVM autovectorization only | VLA semantics require explicit vsetvl control; autovectorizer misses LMUL optimization for compute-bound matmul |
| Custom RVV intrinsic emission | Too brittle across LLVM versions; standard vector dialect lowering is portable and maintained by the LLVM community |
| im2col packing for conv | Out of scope for M4; deferred to M6 (Edge Op Coverage) |
| ARM NEON in M4 | Secondary target; LLVM autovectorizes NEON well enough — deferred to M8 |

---

## 9. Future Work

- **M5**: INT8 quantized path — `arith.extsi` widening + INT32 accumulator.
  The pack layout is reused; tile sizes change for i8 VLMAX.
- **M6**: `ks.depthwise_conv2d` — channel-parallel RVV vectorization.
- **M9**: Double buffering (`--ks-double-buffer`) to overlap compute and memory.
