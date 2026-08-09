# DES-006: Kernel Library Architecture

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-09 |
| **Supersedes** | DES-005 (library packaging), partially DES-002 (matmul pipeline) |
| **Priority** | Critical |

---

## Context

### Problem Statement

KernelSmith's current design targets a compiler pipeline (ks dialect -> LLVM IR -> assembly).
This requires users to understand MLIR, build LLVM, and operate the `ks-opt` tool. That is
a barrier to adoption for the actual consumers of optimized kernels: firmware engineers,
RTOS integrators, ML framework authors, and embedded developers.

The project should instead produce a **RISC-V-first C kernel library** —
precompiled static libraries with stable C headers that any toolchain can link
against. The MLIR compiler becomes an internal build-time tool, not a
user-facing product.

### Goals

1. Ship `libkernelsmith.a` + headers that work with any C99 toolchain
2. Zero runtime dependencies — no malloc, no OS calls, no init/shutdown
3. Prove RISC-V RVV as the first optimized target
4. Predictable memory usage via caller-provided workspace buffers
5. RTOS-compatible: reentrant, no global state, freestanding
6. Keep the design portable enough for later Arm NEON and other targets

### Strategic Positioning

KernelSmith is a **kernel backend**, not a full inference runtime. Its primary
artifact is a static C library that can be called from embedded applications,
test harnesses, and future integrations with runtimes such as llama.cpp/GGML,
ExecuTorch, TFLite Micro, IREE, or ONNX Runtime.

The first optimized product wedge is RISC-V RVV because RVV-capable edge
hardware is growing while the software kernel ecosystem is less mature than Arm.
Arm NEON remains an important secondary target, but it follows the RISC-V
quantized kernel and transformer-demo path.

### Non-Goals

- Runtime JIT compilation
- Dynamic dispatch between targets at runtime
- Autotuning (tile sizes are fixed per target profile at build time)
- Graph-level optimization or operator fusion (single-kernel scope)
- Owning a full LLM or graph runtime
- Maintaining llama.cpp/GGML as the primary product surface

### Background

- DES-005 sketched library packaging but treated it as Phase 4 (after full pipeline)
- DES-002 designed the matmul pipeline but omitted memory management and C interface
- No existing design addresses: workspace memory, data packing, stride support,
  target profiles, or multi-level tiling
- Competing libraries (CMSIS-NN, XNNPACK, ruy) demonstrate the API patterns
  users expect

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Pure C99 public API, no C++ in headers | Must Have |
| REQ-2 | All memory caller-provided (no internal malloc) | Must Have |
| REQ-3 | Workspace query functions for each kernel | Must Have |
| REQ-4 | Stride/leading-dimension parameters for non-contiguous data | Must Have |
| REQ-5 | Static target profile system (build-time target selection) | Must Have |
| REQ-6 | Multi-level tiling driven by target profile cache sizes | Must Have |
| REQ-7 | Data packing for matmul B operand | Must Have |
| REQ-8 | Type variants: f32, f16, i8 (with accumulator types) | Must Have |
| REQ-9 | Tail handling for non-tile-aligned dimensions | Must Have |
| REQ-10 | Alignment requirements documented and queryable | Should Have |
| REQ-11 | In-place operation support where mathematically valid | Should Have |
| REQ-12 | Double buffering / prefetch hints for tiled loops | Nice to Have |
| REQ-13 | Deterministic reduction order (documented per kernel) | Nice to Have |

---

## Design

### 1. System Description Interface (Target Profiles)

A target profile is a static description of the hardware a kernel library is compiled for.
It is a build-time input — one profile per library build. The profile drives tile size
selection, vector width, alignment, and packing decisions.

#### Format

Plain C header, included at build time. No JSON, no parsing, no runtime overhead.

```c
/* target/x86_avx2.h */
#ifndef KS_TARGET_PROFILE_H
#define KS_TARGET_PROFILE_H

#define KS_TARGET_NAME          "x86-avx2"
#define KS_TARGET_ARCH          KS_ARCH_X86_64

/* Vector unit */
#define KS_SIMD_WIDTH_BITS      256
#define KS_SIMD_WIDTH_F32       8       /* 256 / 32 */
#define KS_SIMD_WIDTH_F16       16      /* 256 / 16 */
#define KS_SIMD_WIDTH_I8        32      /* 256 / 8  */

/* Cache hierarchy */
#define KS_L1D_SIZE_KB          32
#define KS_L2_SIZE_KB           256
#define KS_L3_SIZE_KB           8192
#define KS_CACHELINE_BYTES      64

/* Memory alignment */
#define KS_PREFERRED_ALIGN      32      /* AVX2 = 32-byte aligned loads */
#define KS_REQUIRED_ALIGN       1       /* unaligned supported but slower */

/* Matmul tile sizes (derived from cache + vector width) */
/* L2 tiles (outer loop) */
#define KS_MATMUL_TILE_M_L2     128
#define KS_MATMUL_TILE_N_L2     256
#define KS_MATMUL_TILE_K_L2     512

/* L1 tiles (inner loop) */
#define KS_MATMUL_TILE_M_L1     32
#define KS_MATMUL_TILE_N_L1     64
#define KS_MATMUL_TILE_K_L1     128

/* Register tile / micro-kernel */
#define KS_MATMUL_MR            6       /* rows computed per micro-kernel */
#define KS_MATMUL_NR            16      /* cols computed per micro-kernel */

/* Feature flags */
#define KS_HAS_FMA              1
#define KS_HAS_F16C             1       /* f16 <-> f32 conversion */
#define KS_HAS_VNNI             0       /* int8 dot product */

/* Packing */
#define KS_MATMUL_PACK_B        1       /* pack B operand for cache locality */
#define KS_MATMUL_PACK_A        0       /* A packing optional at this size */

#endif /* KS_TARGET_PROFILE_H */
```

#### Provided Profiles

```
target/
  generic.h            -- Portable C, no SIMD, conservative tile sizes (reference)
  riscv_rvv_256.h      -- RISC-V RVV with VLEN=256 (first optimized target)
  riscv_rvv_512.h      -- RISC-V RVV with VLEN=512 (future RVV profile)
  aarch64_neon.h       -- ARMv8-A NEON, 128-bit (secondary target after RVV)
  x86_avx2.h           -- x86-64 with AVX2 + FMA (future)
```

#### How Profiles Drive Codegen

The MLIR pipeline reads profile values as pass options:

```bash
# Build for RISC-V RVV (primary target)
ks-opt input.mlir \
  --ks-tile="l2-tiles=64,64,256 l1-tiles=8,32,256 mr=8,nr=32" \
  --ks-pack="pack-b=true" \
  --ks-vectorize="width=8" \
  --ks-lower-to-rvv \
  --convert-to-llvm="target-triple=riscv64-unknown-linux-gnu"
```

A build script reads the target profile header and translates `#define` values into
pass options. The MLIR passes themselves are target-agnostic — they accept parameters.

#### Design Rationale

**Why C headers, not JSON/YAML/TOML?**
- Zero parse cost — preprocessor evaluates at compile time
- Users can override individual values with `-DKS_MATMUL_TILE_M_L1=16`
- No dependency on any parser library
- Works in freestanding environments (no file I/O needed)
- Can be `#include`d directly into kernel code for compile-time decisions

**Why static, not runtime?**
- RTOS environments often cannot do runtime detection (no cpuid, no /proc)
- Static profiles produce smaller binaries (no dispatch tables)
- Deterministic behavior — same binary always takes the same code path
- Matches how users ship firmware: cross-compile for a known target

---

### 2. C API Design (matmul focus)

#### Public Header: `include/kernelsmith/ks_matmul.h`

```c
#ifndef KERNELSMITH_MATMUL_H
#define KERNELSMITH_MATMUL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * ks_matmul_f32 - General matrix multiplication: C = A * B
 *
 * Computes C[M,N] = A[M,K] * B[K,N]. All matrices are row-major.
 * C must not alias A or B. A and B may alias each other only if the
 * operation is mathematically valid (square, same pointer).
 *
 * Parameters:
 *   A           - Input matrix, M rows x K cols, row-major
 *   lda         - Leading dimension of A (stride between rows, >= K)
 *   B           - Input matrix, K rows x N cols, row-major
 *   ldb         - Leading dimension of B (stride between rows, >= N)
 *   C           - Output matrix, M rows x N cols, row-major (preallocated)
 *   ldc         - Leading dimension of C (stride between rows, >= N)
 *   M           - Number of rows in A and C
 *   N           - Number of columns in B and C
 *   K           - Shared dimension (cols of A, rows of B)
 *   workspace   - Scratch buffer for internal use (may be NULL if ws_size == 0)
 *   ws_size     - Size of workspace in bytes (must be >= ks_matmul_f32_workspace())
 *
 * Returns:
 *   0 on success
 *   KS_ERR_INVALID_ARG   - NULL pointer or invalid dimensions
 *   KS_ERR_WORKSPACE      - workspace too small
 *
 * Thread safety: Reentrant. No global state. Multiple threads may call
 * concurrently with independent buffers.
 *
 * Determinism: Results are bitwise identical across calls with the same
 * inputs, dimensions, and workspace. Reduction order is fixed.
 */
int ks_matmul_f32(
    const float*    A,      size_t lda,
    const float*    B,      size_t ldb,
    float*          C,      size_t ldc,
    size_t M, size_t N, size_t K,
    void* workspace, size_t ws_size
);

int ks_matmul_f16(
    const uint16_t* A,      size_t lda,
    const uint16_t* B,      size_t ldb,
    uint16_t*       C,      size_t ldc,
    size_t M, size_t N, size_t K,
    void* workspace, size_t ws_size
);

/* i8 input, i32 accumulator output */
int ks_matmul_i8(
    const int8_t*   A,      size_t lda,
    const int8_t*   B,      size_t ldb,
    int32_t*        C,      size_t ldc,
    size_t M, size_t N, size_t K,
    void* workspace, size_t ws_size
);

/*
 * Workspace query - returns minimum workspace bytes needed.
 * Returns 0 if the kernel needs no workspace at this size.
 */
size_t ks_matmul_f32_workspace(size_t M, size_t N, size_t K);
size_t ks_matmul_f16_workspace(size_t M, size_t N, size_t K);
size_t ks_matmul_i8_workspace(size_t M, size_t N, size_t K);

/*
 * Preferred memory alignment in bytes for best performance.
 * Pointers not meeting this alignment will still work but may be slower.
 */
size_t ks_matmul_alignment(void);

#ifdef __cplusplus
}
#endif

#endif /* KERNELSMITH_MATMUL_H */
```

#### Common Definitions: `include/kernelsmith/ks_common.h`

```c
#ifndef KERNELSMITH_COMMON_H
#define KERNELSMITH_COMMON_H

/* Version */
#define KS_VERSION_MAJOR  0
#define KS_VERSION_MINOR  1
#define KS_VERSION_PATCH  0

/* Error codes */
#define KS_OK               0
#define KS_ERR_INVALID_ARG  (-1)
#define KS_ERR_WORKSPACE    (-2)
#define KS_ERR_UNSUPPORTED  (-3)

/* Target info (compiled into the library) */
const char* ks_target_name(void);     /* e.g. "x86-avx2" */
size_t      ks_simd_width_f32(void);  /* e.g. 8 */

#endif /* KERNELSMITH_COMMON_H */
```

#### Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Leading dimension params | Yes (`lda`, `ldb`, `ldc`) | Supports non-contiguous tensors (transposed views, submatrices). Standard BLAS convention. |
| Workspace as parameter | Yes (`void* workspace, size_t ws_size`) | No internal malloc. Caller controls memory. RTOS compatible. |
| Return int error code | Yes | No exceptions, no errno. Works in freestanding C. |
| Per-type function names | Yes (`_f32`, `_f16`, `_i8`) | No generics in C. Explicit is better than clever. Linker resolves only what's used. |
| No `alpha`/`beta` scaling | Omitted initially | BLAS has `C = alpha*A*B + beta*C`. We start simple. Add in v0.2 if needed. |
| No transpose flags | Omitted initially | BLAS has `CblasNoTrans`/`CblasTrans`. We start with NT (A normal, B normal). Add in v0.2. |

---

### 3. Memory Management Architecture

#### Principles

1. **No internal allocation.** Kernels never call malloc, calloc, mmap, or any allocator.
2. **Workspace is a byte buffer.** The kernel uses it as a scratch arena — layout is internal.
3. **Workspace is ephemeral.** Contents are undefined after the kernel returns. Caller can
   reuse the same workspace buffer across different kernel calls.
4. **Workspace size is deterministic.** For given dimensions, `_workspace()` always returns
   the same value. It's a pure function of (M, N, K) and the target profile tile sizes.

#### Workspace Composition (matmul)

For a tiled, packed matmul, workspace contains:

```
workspace layout:
  +---------------------------+
  | Packed B panel            |  K * NR * sizeof(element)
  | (reused per M-tile)       |  (NR from target profile)
  +---------------------------+
  | Packed A panel            |  MR * K * sizeof(element) [if KS_MATMUL_PACK_A]
  | (optional, profile flag)  |
  +---------------------------+
  | Alignment padding         |  up to KS_PREFERRED_ALIGN bytes
  +---------------------------+
```

Workspace size computation:

```c
size_t ks_matmul_f32_workspace(size_t M, size_t N, size_t K) {
    size_t pack_b = KS_MATMUL_PACK_B ? (K * KS_MATMUL_NR * sizeof(float)) : 0;
    size_t pack_a = KS_MATMUL_PACK_A ? (KS_MATMUL_MR * K * sizeof(float)) : 0;
    size_t align  = KS_PREFERRED_ALIGN;
    /* Round up each region to alignment boundary */
    pack_b = (pack_b + align - 1) & ~(align - 1);
    pack_a = (pack_a + align - 1) & ~(align - 1);
    return pack_b + pack_a;
}
```

For the `generic` target profile (no packing): workspace is 0 bytes.

#### MLIR Mapping

In the generated MLIR, workspace appears as a memref function argument:

```mlir
func.func @ks_matmul_f32(
    %A: memref<?x?xf32, strided<[?, 1]>>,     // A with lda stride
    %B: memref<?x?xf32, strided<[?, 1]>>,     // B with ldb stride
    %C: memref<?x?xf32, strided<[?, 1]>>,     // C with ldc stride
    %workspace: memref<?xi8>                    // raw byte buffer
) -> i32 {
    // Tiling, packing, compute — all within workspace
    // No memref.alloc anywhere in the function body
}
```

The `--ks-bufferize` pass enforces that no `memref.alloc` ops survive lowering.
Any surviving alloc is a compilation error.

---

### 4. Tiling Strategy

#### Multi-Level Tiling

Matmul uses three levels of tiling, each targeting a cache level:

```
for mc in 0..M step TILE_M_L2:           // L2 tile over M
  for nc in 0..N step TILE_N_L2:         // L2 tile over N
    for kc in 0..K step TILE_K_L2:       // L2 tile over K
      pack_b(B[kc:kc+TILE_K_L2, nc:nc+TILE_N_L2], workspace)
      for mr in mc..mc+TILE_M_L2 step TILE_M_L1:   // L1 tile
        for nr in nc..nc+TILE_N_L2 step TILE_N_L1:  // L1 tile
          micro_kernel(A[mr, kc], packed_B, C[mr, nr])
```

The micro-kernel (innermost computation) processes an MR x NR tile using SIMD.
MR and NR are chosen so that all intermediate values fit in registers.

#### Tile Size Selection

Tile sizes come from the target profile. The rationale for each level:

| Level | Target | Constraint | Example (x86 AVX2, f32) |
|-------|--------|-----------|------------------------|
| L2 | L2 cache | `TILE_M_L2 * TILE_K_L2 * sizeof(elem) < L2_SIZE / 2` | 128 x 512 x 4 = 256KB < 256KB |
| L1 | L1 cache | `TILE_M_L1 * TILE_N_L1 * sizeof(elem) < L1_SIZE / 2` | 32 x 64 x 4 = 8KB < 16KB |
| Register | Registers | `MR * NR <= available_vector_regs * SIMD_WIDTH` | 6 x 16 = 96 values in 12 YMM regs |

**Why / 2?** Both the current tile and the next tile (for prefetch or packing) need
to coexist in cache.

#### Tail Handling

When dimensions are not divisible by tile sizes:

```
M = 100, TILE_M = 64

Iteration 1: m = 0..63   (full tile)
Iteration 2: m = 64..99  (partial tile, 36 rows)
```

MLIR's tiling generates `min(tile_size, remaining)` as the trip count. For the
micro-kernel, partial tiles use one of:

1. **Masked operations** (preferred for RVV, AVX-512): process full vector width,
   mask out invalid lanes
2. **Scalar fallback**: process remaining elements one at a time (generic target)
3. **Padding**: copy partial tile into zero-padded workspace buffer, run full
   micro-kernel, copy results back (used when masking is unavailable)

The target profile `KS_SIMD_WIDTH_F32` determines which strategy applies.

#### MLIR Implementation

The tiling pass applies `linalg.tile` in sequence:

```
Pass 1: --ks-tile-l2 (reads TILE_*_L2 from profile)
  linalg.matmul -> scf.for (L2 tiles) { linalg.matmul on tile }

Pass 2: --ks-pack (reads PACK_A, PACK_B from profile)
  Insert linalg.pack ops for B (and optionally A)

Pass 3: --ks-tile-l1 (reads TILE_*_L1, MR, NR from profile)
  Inner linalg.matmul -> scf.for (L1 tiles) { micro-kernel sized matmul }
```

The passes themselves are target-agnostic. All target-specific knowledge lives in
the profile values passed as options.

---

### 5. Data Packing

#### Why Packing Matters

In a naive tiled matmul, accessing a column-panel of B strides across memory:

```
B is row-major [K rows x N cols]:
  row 0: [b00, b01, b02, ..., b0N]
  row 1: [b10, b11, b12, ..., b1N]
  ...

Accessing B[:, 0:NR] means:
  b00 (cache line 0), b10 (cache line ~N/16), b20 (cache line ~2N/16), ...
  -> one cache miss per row of B
```

Packing rearranges B into tile-contiguous layout:

```
packed_B for panel n=0..NR:
  [b00, b01, ..., b0_NR, b10, b11, ..., b1_NR, b20, ...]
  -> sequential memory access, hardware prefetcher works perfectly
```

#### Performance Impact

| Configuration | Relative perf (typical) |
|--------------|------------------------|
| No tiling | 1x (baseline) |
| Tiled, no packing | 3-5x |
| Tiled + B packing | 8-15x |
| Tiled + A&B packing | 10-20x |

Packing is the single biggest performance optimization after tiling.

#### MLIR Representation

```mlir
// Before packing (tiled matmul, B accessed with strides)
%B_tile = memref.subview %B[%kc, %nc] [%tile_k, %tile_n] [1, 1]
linalg.matmul ins(%A_tile, %B_tile) outs(%C_tile)

// After packing pass
%packed_B = memref.subview %workspace[0] [%pack_size] [1]  // region in workspace
linalg.pack %B_tile ... into %packed_B  // contiguous copy
linalg.matmul ins(%A_tile, %packed_B) outs(%C_tile)
```

#### Packing and Workspace

Pack buffers live inside the caller-provided workspace. The workspace query function
accounts for this. On the `generic` target (KS_MATMUL_PACK_B=0), no packing occurs
and workspace is 0.

---

### 6. Stride Support

#### The Problem

Real-world tensors are frequently non-contiguous:

```c
// A is a submatrix of a larger matrix — rows are not adjacent
float big_matrix[1000][1000];
// User wants matmul on rows 10-74, cols 20-148
// A points to &big_matrix[10][20], lda = 1000 (not 128)
ks_matmul_f32(&big_matrix[10][20], 1000, ...);
```

Without stride parameters, the user must copy data into a contiguous buffer before
calling the kernel. This doubles memory usage and adds latency.

#### MLIR Mapping

Strided access maps directly to MLIR's `memref` with strides:

```mlir
// lda=1000 for a 64x128 submatrix
%A = memref.cast ... : memref<64x128xf32, strided<[1000, 1]>>
```

MLIR's `memref.subview` and strided memref types handle this natively. The lowering
passes access elements through the stride metadata — no special handling needed in
the tiling or vectorization passes.

#### Contiguous Fast Path

When `lda == K` (contiguous), packing can use `memcpy`-style bulk copies instead of
strided element-by-element copies. The micro-kernel can also use aligned vector loads.
The generated code checks this:

```c
if (lda == K) {
    // fast path: sequential vector loads
} else {
    // strided path: gather or scalar load + broadcast
}
```

---

### 7. Type Variants and Codegen

#### Variant Matrix (matmul)

| Function | Input A | Input B | Output C | Accumulator | Notes |
|----------|---------|---------|----------|-------------|-------|
| `ks_matmul_f32` | f32 | f32 | f32 | f32 | Standard |
| `ks_matmul_f16` | f16 | f16 | f16 | f32 internal | Accumulate in f32, truncate |
| `ks_matmul_i8` | i8 | i8 | i32 | i32 | Quantized inference |

#### Quantized Decode Kernel APIs

| Function | Activation | Weight | Output | Accumulator | Layout |
|----------|------------|--------|--------|-------------|--------|
| `ks_dot_i8` | i8 | i8 | i32 scalar | i32 | Contiguous vectors |
| `ks_matvec_i8` | i8 | i8 | i32 vector | i32 | Row-major weights with row stride |
| `ks_dot_w4a8` | i8 | signed int4 | f32 scalar | f32 | Low-nibble-first packed weights |
| `ks_matvec_w4a8` | i8 | signed int4 | f32 vector | f32 | Row-major packed rows plus per-group f32 scales |

W4A8 rows pack two signed int4 weights per byte. The low nibble stores even
`k`, the high nibble stores odd `k`, and each row owns
`ceil(cols / group_size)` f32 scales. The performance path fuses int4 unpack,
dequantization, and dot/GEMV compute instead of materializing dequantized
weights.

#### How Variants Are Generated

One MLIR template, stamped per type:

```mlir
// matmul_template.mlir (parameterized)
func.func @ks_matmul_${SUFFIX}(
    %A: memref<?x?x${IN_TYPE}, strided<[?, 1]>>,
    %B: memref<?x?x${IN_TYPE}, strided<[?, 1]>>,
    %C: memref<?x?x${OUT_TYPE}, strided<[?, 1]>>,
    %workspace: memref<?xi8>
) -> i32 {
    // ... same lowering passes, MLIR handles type-specific ops ...
}
```

Build-time expansion:

```makefile
VARIANTS = f32:f32:float:float  f16:f16:uint16_t:uint16_t  i8:i32:int8_t:int32_t

$(foreach v,$(VARIANTS),\
  $(eval SUFFIX=$(word 1,$(subst :, ,$v))) \
  $(eval ACCU=$(word 2,$(subst :, ,$v))) \
  ... generate and compile variant ... \
)
```

Each variant compiles to a separate `.o` file. All link into one `.a`.

#### MLIR Type Handling

The lowering passes are type-generic:

- `ks.matmul` accepts any supported element type (constrained by TableGen)
- `linalg.matmul` works on any numeric type
- Vectorization picks the right SIMD width from the profile (`KS_SIMD_WIDTH_F32`,
  `KS_SIMD_WIDTH_I8`, etc.)
- LLVM backend emits target-specific instructions per type

The only type-specific logic is:
1. **Accumulator promotion** (f16 inputs -> f32 accumulator -> f16 output):
   insert `arith.extf` / `arith.truncf` around the accumulator
2. **i8 matmul** uses `arith.extsi` + `arith.muli` + `arith.addi` instead of
   `arith.mulf` + `arith.addf`

These are handled in the `--ks-lower-to-linalg` pass based on input/output types.

---

### 8. Double Buffering and Prefetch

#### When It Matters

| Target | Benefit | Approach |
|--------|---------|----------|
| x86 (AVX2/512) | Small (10-20%) | `_mm_prefetch` hints, hardware prefetcher does most work |
| ARM (Cortex-A) | Medium (15-30%) | Software prefetch via `__builtin_prefetch` |
| RISC-V (simple cores) | Large (30-50%) | Explicit double buffering of pack buffers |
| MCU with DMA | Essential | DMA transfer overlapped with compute |

#### Design

Double buffering is an **optional optimization pass** (`--ks-double-buffer`), not part
of the core pipeline. The core pipeline produces correct, well-tiled code without it.

When enabled, the pass transforms the K-loop:

```
// Before: sequential pack + compute
for kc in 0..K step TILE_K:
    pack(B[kc])        // stalls until complete
    compute(A, packed_B, C)

// After: overlapped pack + compute
pack(B[0])  // prime the pipeline
for kc in 0..K-TILE_K step TILE_K:
    pack(B[kc + TILE_K])           // pack NEXT tile into buffer 1
    compute(A, packed_B_buf0, C)   // compute CURRENT tile from buffer 0
    swap(buf0, buf1)
compute(A, packed_B_buf0, C)  // drain: last tile
```

This doubles the packing workspace (two B panels instead of one). The workspace query
accounts for this when double buffering is enabled in the profile:

```c
#define KS_MATMUL_DOUBLE_BUFFER  1  /* in target profile */
```

#### Priority

**Defer to v0.2.** Focus on correct tiling and packing first. Double buffering
is more impactful on RISC-V (simple cores, limited prefetch) than on x86.

---

### 9. Additional Concerns

#### Alignment

The target profile specifies `KS_PREFERRED_ALIGN`. The workspace query rounds
all internal buffer offsets to this alignment. The public API documents:

> For best performance, input and output pointers should be aligned to
> `ks_matmul_alignment()` bytes. Unaligned pointers are supported but may
> incur a performance penalty.

#### In-Place Operations

| Kernel | In-place? | Notes |
|--------|-----------|-------|
| matmul | No | C cannot alias A or B (overwrites during compute) |
| relu | Yes | `ks_relu_f32(buf, buf, n)` is safe |
| gelu | Yes | Safe — element-wise |
| softmax | No | Requires two passes over input (max, then exp) |
| layer_norm | No | Requires mean/variance over input |

Documented in each kernel's header comment. Enforced in debug builds via overlap checks.

#### Thread Safety

All kernels are reentrant by construction:
- No global/static mutable state
- No `init()`/`shutdown()`
- Workspace is caller-provided and per-call
- Multiple threads can call the same kernel concurrently with different buffers

#### Determinism

Floating-point reduction order (sum in matmul's K-dimension) is deterministic for
given (M, N, K, tile_sizes). Changing tile sizes (different target profile) may
change results due to rounding. This is documented.

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Runtime target dispatch | Single binary for all targets | Code size, dispatch overhead, RTOS unfriendly | Rejected |
| JSON target profiles | Human-readable, easy to edit | Needs parser, runtime cost, no freestanding | Rejected |
| **C header target profiles** | Zero cost, overridable, freestanding | Less structured than JSON | **Selected** |
| C++ API as primary | Type safety, overloading | ABI instability, no C interop, RTOS issues | Rejected |
| BLAS-compatible API | Familiar to users | Complex (transpose, alpha/beta, layout), scope creep | Deferred to v0.2 |

---

## Data Flow: From MLIR to Shipped Library

```
Target Profile (C header)
  │
  ├── extracted by build script as pass options
  │
  ▼
MLIR Template (ks.matmul with type params)
  │
  ├── --ks-lower-to-linalg      (ks.matmul -> linalg.matmul, type promotion)
  ├── --ks-tile-l2               (L2-level tiling from profile)
  ├── --ks-pack                  (data packing if profile enables it)
  ├── --ks-tile-l1               (L1-level tiling + micro-kernel sizing)
  ├── --ks-vectorize             (SIMD using profile vector width)
  ├── --one-shot-bufferize       (tensor -> memref, no allocs)
  ├── --ks-alloc-check           (fail if any memref.alloc survives)
  ├── --convert-to-llvm          (target triple from profile)
  │
  ▼
LLVM IR (target-specific)
  │
  ├── llvm-translate -> .ll
  ├── llc -> .o  (or clang -c)
  │
  ▼
Object files (one per kernel variant)
  │
  ├── ar rcs libkernelsmith.a *.o
  │
  ▼
Shipped artifact:
  libkernelsmith.a
  include/kernelsmith/ks_matmul.h
  include/kernelsmith/ks_common.h
```

---

## Test Strategy

### Unit Tests (C)

- [ ] `ks_matmul_f32`: Square (64x64), rectangular (32x128 x 128x64), small (4x4)
- [ ] `ks_matmul_f32`: Non-tile-aligned (17x23 x 23x31)
- [ ] `ks_matmul_f32`: With stride (lda > K, submatrix access)
- [ ] `ks_matmul_f32`: Workspace = NULL when `_workspace()` returns 0 (generic target)
- [ ] `ks_matmul_f32`: Error on workspace too small
- [ ] `ks_matmul_f32`: Error on NULL pointers
- [ ] `ks_matmul_f16`: Correctness vs f32 reference (within f16 tolerance)
- [ ] `ks_matmul_i8`: Correctness vs i32 reference (exact for small values)
- [ ] Workspace query returns deterministic values
- [ ] Alignment query returns profile-consistent value
- [ ] Thread safety: concurrent calls from multiple threads

### Lit Tests (MLIR pipeline)

- [ ] `--ks-lower-to-linalg`: ks.matmul disappears, linalg.matmul appears
- [ ] `--ks-tile-l2`: scf.for loops appear with profile tile sizes
- [ ] `--ks-pack`: linalg.pack ops appear for B operand
- [ ] `--ks-tile-l1`: nested scf.for with MR/NR sizes
- [ ] `--ks-vectorize`: vector.load / vector.fma / vector.store appear
- [ ] `--ks-alloc-check`: pipeline with no memref.alloc (pass succeeds)
- [ ] `--ks-alloc-check`: pipeline with stray alloc (pass fails with error)
- [ ] Tail handling: non-divisible dimensions produce min() bounds

### Numerical Validation

- [ ] f32 matmul vs numpy: max relative error < 1e-5
- [ ] f16 matmul vs numpy: max relative error < 1e-2
- [ ] i8 matmul vs numpy: exact match for values within i32 range
- [ ] Results deterministic across repeated runs

### Cross-Target

- [ ] Build with `generic` profile: compiles, runs, correct results, workspace=0
- [ ] Build with `riscv_rvv_256` profile: compiles, runs on QEMU, correct results (primary)
- [ ] Build with `aarch64_neon` profile: compiles, runs natively or cross-compiled (secondary)

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Packing overhead exceeds benefit for small matrices | Medium | Medium | Skip packing below threshold (e.g., M*N*K < 4096) |
| Multi-level tiling pass complexity | Medium | Medium | Build one level at a time, test each in isolation |
| MLIR bufferization leaves stray allocs | High | Medium | `--ks-alloc-check` pass as hard gate in pipeline |
| Stride support complicates vectorization | Medium | Low | MLIR memref handles strides natively |
| Profile tile sizes suboptimal | Medium | High | Start with literature values, benchmark and revise |
| i8/f16 accumulator promotion bugs | High | Medium | Dedicated lit tests for each type combination |

---

## Open Questions

- [ ] Should we support column-major (Fortran) layout as an option?
- [ ] Should `alpha`/`beta` GEMM scaling be in v0.1 or deferred?
- [ ] Should we provide a `ks_matmul_f32_transb` variant (B transposed) in v0.1?
- [ ] Minimum dimensions: should we special-case 1x1, Mx1, 1xN (vector dot, GEMV)?
- [ ] Should the generic profile use 2-level or 1-level tiling?

---

## Dependencies

- MLIR/LLVM 18+ (lowering infrastructure, bufferization, linalg.pack)
- Host C compiler (for building the generic target library)
- Cross-compiler per target (riscv64-unknown-elf-gcc, aarch64-linux-gnu-gcc, etc.)
- Python 3.10+ (build scripts, test data generation)
- numpy (numerical validation reference)

---

## Implementation Plan

### Stage 1: API and Reference

Establish the C interface and a working library with no MLIR involvement.

- [ ] Write `include/kernelsmith/ks_matmul.h` and `ks_common.h`
- [ ] Write `target/generic.h` profile (portable C, no SIMD)
- [ ] Handwrite `ks_matmul_f32` as plain triple-loop C (reference implementation)
- [ ] Handwrite `ks_matmul_f32_workspace` (returns 0 for generic)
- [ ] Build as `libkernelsmith.a` with CMake
- [ ] Write C test program that links against the library
- [ ] Verify numerical correctness vs numpy
- [ ] Ship: users can `#include` and link today

### Stage 2: MLIR Pipeline Basics

Replace the handwritten reference with MLIR-generated code for the generic target.

- [ ] Implement `--ks-lower-to-linalg` pass (ks.matmul -> linalg.matmul)
- [ ] Implement `--ks-tile-l2` pass (single-level tiling, generic profile sizes)
- [ ] Connect pipeline: MLIR -> LLVM IR -> .o (for host target)
- [ ] Replace handwritten matmul with MLIR-generated version
- [ ] Verify: same tests pass, same numerical results
- [ ] Lit tests for each pass in isolation

### Stage 3: Packing + Multi-Level Tiling + RVV Lowering

Performance-critical optimizations, targeting RISC-V RVV as primary.

- [ ] Write `target/riscv_rvv_256.h` profile (VLEN=256 baseline)
- [ ] Implement `--ks-pack` pass (B operand packing into workspace)
- [ ] Implement `--ks-tile-l1` pass (L1 tiling, MR/NR micro-kernel)
- [ ] Update workspace query to account for pack buffers
- [ ] Implement `--ks-vectorize` pass (SIMD using profile width)
- [ ] Implement `--ks-lower-to-rvv` pass (vector/tensor pipeline to LLVM dialect
      for the RISC-V backend)
- [ ] Cross-compile for riscv64, test on QEMU with multiple VLEN configs
- [ ] Benchmark: generic vs RVV profile, show speedup
- [ ] Stride tests: non-contiguous input matrices

### Stage 4: RISC-V Quantization

INT8 and W4A8 quantized kernels for RISC-V edge inference. Prioritize
dot/GEMV for batch-1 transformer decode before large GEMM.

- [ ] Add `ks.quantize` / `ks.dequantize` ops
- [ ] Add quantization attributes to ks.matmul (scale, zero_point)
- [ ] Implement INT8 lowering: i8 -> i32 accumulate -> requantize -> i8
- [ ] Add quantized dot/GEMV APIs and lowering
- [ ] Add W4A8 fused unpack/dequantize + compute kernels
- [ ] RVV INT8 path: `vwmul.vv` (widening multiply), `vnsra.wi` (narrowing shift)
- [ ] Tighten TableGen type constraints (replace AnyTensor)
- [ ] Test all type variants (f32, i8, W4A8)

### Stage 5: Known-Runtime Transformer Integration Smoke

Prove that a pinned llama.cpp runtime can consume the stable C API before
coupling runtime integration to quantized-layout conversion.

- [ ] Pin a llama.cpp revision and licensed, checksummed CI-sized model
- [ ] Establish deterministic unmodified host output
- [ ] Route one reviewed f32 operation through `libkernelsmith`
- [ ] Prove invocation and deterministic token parity

### Stage 6: Ecosystem and Secondary Target Follow-Ons

Use the proven C kernels in broader environments.

- [ ] Add tested GGML-to-KernelSmith W4A8 conversion
- [ ] Route selected llama.cpp decode kernels through generated RVV objects
- [ ] Validate the runtime in QEMU user-mode before system-mode and gem5
- [ ] Write `target/aarch64_neon.h` profile
- [ ] Profile-driven Arm NEON tile sizes
- [ ] Cross-compile and test natively or on target device

---

## Related Documents

- [DES-001: Vector Operations Lowering](DES-001-vector-operations-lowering.md)
- [DES-002: MatMul Kernel (TDD)](DES-002-matmul-kernel.md) — pipeline design, partially superseded
- [DES-011: RISC-V First Transformer Demo Strategy](DES-011-riscv-first-transformer-demo.md)
- [DES-016: Known-Runtime-First Transformer Integration](DES-016-known-runtime-first-transformer-integration.md)
- ~~DES-005~~ — deleted, was superseded by this document
- [MatMul Specification](../../specs/kernels/matmul.md)
- [RVV Target Specification](../../specs/targets/riscv-rvv.md)
- [ROADMAP.md](../../ROADMAP.md) — updated to reflect library-first approach
