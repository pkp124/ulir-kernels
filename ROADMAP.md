# KernelSmith Roadmap

> **Vision**: Ship a C kernel library (`libkernelsmith.a` + headers) that provides
> optimized, widely-used ML operations behind a stable C99 API. No runtime
> dependencies. No MLIR knowledge required by users. Works with any toolchain,
> any RTOS, any task execution engine.
>
> **Architecture reference**: [DES-006](docs/design/DES-006-kernel-library-architecture.md)

---

## Milestone 0: Dialect Infrastructure (Complete)

**Status**: Done

- KS dialect with 13 operations defined in TableGen
- Verifiers for 5 operations (matmul, batch_matmul, conv2d, attention, layer_norm)
- `ks-opt` CLI tool (parse/print)
- Lit tests for parse round-trip and verifier errors
- Build system (CMake + LLVM/MLIR 18 integration)

---

## Milestone 1: C API + Reference Library (Complete)

**Status**: Done

- `include/kernelsmith/ks_matmul.h` — matmul C API (f32)
- `include/kernelsmith/ks_activations.h` — relu, gelu, silu C API
- `include/kernelsmith/ks_common.h` — error codes, version, target info
- `target/generic.h` — portable C target profile (no SIMD, no packing)
- Handwritten reference `ks_matmul_f32` (tiled triple-loop, stride support)
- Handwritten reference `ks_relu_f32`, `ks_gelu_f32`, `ks_silu_f32`
- `libkernelsmith.a` built with CMake
- C test programs validated against numpy
- Workspace query functions (return 0 for generic profile)

---

## Milestone 2: MLIR Lowering — Activations (Complete)

**Status**: Done

- `KSLowerActivationsPass` (`--ks-lower-activations`)
- Pass registration infrastructure (`Passes.h`, `Passes.td`)
- Activation type constraints tightened (`KS_FloatTensor`)
- Lit tests for each transformation (`tests/lit/Passes/lower-activations.mlir`)

---

## Milestone 3: MLIR Lowering — MatMul (Generic Target) (Current)

**Goal**: Replace handwritten matmul with MLIR-generated code. Single-level tiling,
no packing, generic target profile.

**Deliverables**:
- `KSLowerToLinalgPass` (`--ks-lower-to-linalg`)
- `KSTilePass` (`--ks-tile`) with profile-driven tile sizes
- `KSAllocCheckPass` (`--ks-alloc-check`) — fail if any `memref.alloc` survives
- Bufferization config: all buffers are function arguments, zero internal malloc
- Generated matmul for generic target
- C library tests still pass

**Tasks**:
1. Implement `--ks-lower-to-linalg` (ks.matmul -> linalg.matmul + tensor.empty)
2. Add accumulator type attribute to ks.matmul for mixed-precision (f16->f32, i8->i32)
3. Implement `--ks-tile` (single-level, reads tile sizes from pass options)
4. Configure one-shot-bufferize: function-argument buffers only
5. Implement `--ks-alloc-check` (reject stray memref.alloc as hard error)
6. Build script: read target/generic.h, translate to pass options
7. Pipeline: ks.matmul -> linalg -> tile -> bufferize -> LLVM IR -> .o
8. Replace handwritten matmul with generated version
9. Verify: same C tests pass
10. Lit tests for each pass in isolation

---

## Milestone 4: Packing + Multi-Level Tiling + Vectorization (x86 AVX2)

**Goal**: Performance-optimized matmul for x86 AVX2. Multi-level tiling, B packing,
SIMD vectorization. This is where the library starts outperforming naive C.

**Deliverables**:
- `target/x86_avx2.h` target profile
- `KSPackPass` (`--ks-pack`) — B operand packing into workspace
- Multi-level tiling (L2 + L1/register tile)
- `KSVectorizePass` (`--ks-vectorize`) — SIMD using profile vector width
- Workspace query returns correct size for pack buffers
- Benchmark: generic vs x86_avx2, demonstrating speedup

**Tasks**:
1. Implement `--ks-pack` (linalg.pack for B operand, workspace memref)
2. Extend `--ks-tile` to multi-level (L2 outer + MR/NR register tile)
3. Update workspace query to account for pack buffer size
4. Implement `--ks-vectorize` (inner loops -> vector.load/fma/store)
5. Stride tests: non-contiguous input matrices (lda != K)
6. Tail handling tests: non-tile-aligned dimensions
7. Benchmark against reference (numpy/OpenBLAS) on x86
8. Lit tests for packing and multi-level tiling

---

## Milestone 5: Type Variants

**Goal**: f16 and i8 matmul variants. Activation variants for all supported types.

**Deliverables**:
- `ks_matmul_f16`, `ks_matmul_i8` generated from same lowering passes
- Accumulator type promotion (f16 input -> f32 accumulator -> f16 output)
- Variant generation build script (one MLIR template, N type expansions)
- All type variants tested for numerical correctness

**Tasks**:
1. Update TableGen: replace `AnyTensor` with type-class constraints per op
2. Implement accumulator promotion in `--ks-lower-to-linalg`
3. Build script: stamp out f32, f16, i8 variants from MLIR templates
4. Test f16 vs f32 reference (within f16 tolerance)
5. Test i8 with i32 accumulator (exact for small values)

---

## Milestone 6: Additional Targets

**Goal**: Same C API, optimized for ARM and RISC-V.

**Tasks per target**:
1. Write target profile header (`target/<name>.h`)
2. Set LLVM target triple and features in build script
3. For RVV: implement `KSLowerToRVVPass` (custom RVV intrinsics, vsetvl management)
4. For ARM NEON/SVE: LLVM autovectorization + target features (no custom pass needed)
5. Cross-compile, test on QEMU (RVV) or natively (ARM)
6. Numerical validation against generic target output

**Target priority**:
1. ARM NEON (ARMv8-A) — widest hardware base after x86
2. RISC-V RVV — project's original target, requires custom lowering pass
3. ARM SVE — scalable vectors, similar to RVV challenge
4. x86 AVX-512 — incremental over AVX2 (wider vectors, masking)
5. Cortex-M7 — MCU/RTOS target, no SIMD, validates freestanding story

---

## Milestone 7: Additional Kernels

**Goal**: Expand the C API beyond matmul and activations.

**Kernel priority** (by frequency in ML inference):
1. `ks_conv2d` — 2D convolution (NHWC), im2col or direct
2. `ks_softmax` — numerically stable (max-subtract-exp-sum-divide)
3. `ks_layer_norm` / `ks_rms_norm` — normalization
4. `ks_attention` — scaled dot-product attention (composes matmul + softmax)
5. `ks_reduce_sum` / `ks_reduce_max` — reductions along axes
6. `ks_batch_matmul` — batched matmul

Each kernel follows the same pattern:
- Add tile size parameters to target profiles
- Write C header with workspace query
- Implement MLIR lowering (or handwrite reference first, generate later)
- Test against numpy

---

## Milestone 8: Optimization and Hardening

**Goal**: Production-quality library performance and robustness.

**Tasks**:
- Double buffering pass (`--ks-double-buffer`) for memory-bound kernels
- Software prefetch hints for x86 and ARM
- Kernel fusion exploration (matmul + relu, matmul + bias + activation)
- Profile-guided tile size tuning (benchmark harness + parameter sweep)
- Fuzz testing of C API (invalid args, edge cases, large dimensions)
- Memory sanitizer (ASAN/MSAN) CI integration
- Documentation: API reference, integration guide, performance tuning guide

---

## Design Documents

| ID | Title | Scope |
|----|-------|-------|
| [DES-006](docs/design/DES-006-kernel-library-architecture.md) | Kernel Library Architecture | C API, memory mgmt, tiling, packing, target profiles |
| [DES-002](docs/design/DES-002-matmul-kernel.md) | MatMul Kernel (TDD) | MLIR pipeline design (partially superseded by DES-006) |
| [DES-001](docs/design/DES-001-vector-operations-lowering.md) | Vector → RVV Lowering | RVV-specific intrinsic mapping |
| [DES-005](docs/design/DES-005-library-packaging.md) | Library Packaging (v1) | Superseded by DES-006 |

## Target Profile Specification

See [specs/targets/system-description.md](specs/targets/system-description.md) for the
target profile format, validation rules, and provided profiles.
