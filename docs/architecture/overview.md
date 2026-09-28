# Architecture Overview

KernelSmith compiles high-level kernel operations into a static C library for
edge devices. RISC-V RVV is the first optimized target. Callers link
`libkernelsmith.a`, pass their own buffers, and do not need MLIR.

## Goals

- Edge inference first: small RISC-V SoCs, then ARM NEON
- Quantized dot and GEMV in the public ABI
- No library-internal malloc; workspace is caller-provided
- Each lowering stage has its own FileCheck tests

## Pipeline

```text
ks.matmul / ks.relu / ks.rms_norm / ks.dot_i8 / ...
        |
        |  --ks-lower-to-linalg    (structured, elementwise, norm, quantized)
        |  --ks-lower-activations  (relu, gelu, silu)
        v
linalg.matmul / linalg.generic
        |
        |  --ks-tile               (scf.for tiles; matmul)
        |  --ks-pack               (B panel layout [N/NR, K, NR])
        |  --ks-materialize-pack-workspace
        v
tiled and packed linalg
        |
        |  --ks-vectorize
        v
vector.transfer_read / vector.contract / vector.transfer_write
        |
        |  --ks-lower-to-rvv       (bufferize, loops, vector, LLVM dialect)
        v
LLVM IR  ->  llc -march=riscv64 -mattr=+v  ->  RVV object
```

Tile sizes and the pack factor come from the selected target profile in
`target/`. The RVV profile is `target/riscv_rvv_256.h` (VLEN baseline 256,
LMUL 4, f32). Code stays vector-length agnostic: VLEN is a runtime value, not
a hardcoded register width.

`--ks-lower-to-rvv` is the RVV path. ARM NEON is planned as LLVM
autovectorization driven by a future target profile. There is no NEON profile
in `target/` yet. `target/generic.h` is portable scalar C.
`target/x86_avx2.h` exists as a profile header.

## Dialect

The dialect prefix is `ks`. Definitions are TableGen in
`include/KernelSmith/Dialect/Kernel/KernelOps.td`. Implementations are
`lib/Dialect/Kernel/KernelOps.cpp`.

```mlir
%C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32>
                        -> tensor<64x256xf32>

%Y = ks.relu %X : tensor<1024xf32>

%A = ks.dot_i8 %X, %W {input_zero_point = 0 : i64, weight_zero_point = 0 : i64}
     : tensor<128xi8>, tensor<128xi8> -> tensor<i32>
```

Operations are pure. Verifiers check rank, element type, and shape
relationships. Three activations (`ks.relu`, `ks.gelu`, `ks.silu`) still rely
on type constraints only.

Which operations lower, and which have C entry points, is the table in the
[README](../../README.md#supported-kernels). Convolution, attention, batch
matmul, layer norm, quantize/dequantize, and the reductions currently stop at
parse and verify.

## C library

`lib/kernelsmith/` is C99. Headers are `include/kernelsmith/`. The library
returns `KS_OK` or a negative `KS_ERR_*` code and does not allocate. A target
profile is force-included at compile time, so the same sources build a generic
library or an RVV library.

Today the f32 matmul and activation symbols are handwritten. The compiler can
lower those operations, and the generated objects are not yet the ones inside
the default archive. INT8 dot and GEMV can link generated RVV objects through
`KS_INT8_RVV_OBJECTS`. W4A8 symbols are still the reference kernels.

The library design is [DES-006](../design/DES-006-kernel-library-architecture.md).
The RVV pipeline is [DES-009](../design/DES-009-m4-rvv-lowering.md).

## Passes

| Flag | Role |
|---|---|
| `--ks-lower-activations` | `ks.relu`, `ks.gelu`, `ks.silu` to `linalg.generic` |
| `--ks-lower-to-linalg` | matmul, add, mul, RMSNorm, softmax, INT8 and W4A8 dot/GEMV |
| `--ks-tile` | Tile `linalg.matmul` with profile tile sizes |
| `--ks-pack` | Pack matmul B into column panels |
| `--ks-materialize-pack-workspace` | Put the packed B buffer in caller workspace |
| `--ks-vectorize` | Linalg to vector contract and transfer ops |
| `--ks-lower-to-rvv` | Vector and buffer IR to the LLVM dialect for RVV |
| `--ks-alloc-check` | Error if a generated kernel still contains `memref.alloc` |

Pass declarations are `include/KernelSmith/Passes/Passes.td`. Implementations
are `lib/Passes/`.

## How to extend it

A new kernel starts as a `ks.` operation, lowers to linalg, and only then
participates in tiling and vectorization. A new fixed-width SIMD target adds
a profile header and uses LLVM autovectorization. A vector-length-agnostic
target follows the RVV pattern: an explicit lowering pass plus a profile.
Profile fields are specified in
[specs/targets/system-description.md](../../specs/targets/system-description.md).

The step-by-step contributor path is
[Adding a Kernel](../guides/adding-kernels.md).
