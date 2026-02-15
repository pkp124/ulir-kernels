# DES-008: RISC-V LLM Runtime Stack Integration

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-02-15 |
| **Depends On** | DES-006 (Kernel Library Architecture), DES-001 (Vector → RVV Lowering) |
| **Priority** | High |

---

## Context

### Problem Statement

KernelSmith produces `libkernelsmith.a` — a static C library of optimized kernel
functions (matmul, attention, softmax, activations, normalization). These kernels
are the computational primitives needed for LLM inference, but they do not
constitute a complete inference system.

Running an LLM end-to-end requires orchestration above the kernel level:

- **Model weight loading** — deserializing multi-GB parameter files
- **KV-cache management** — growing/shrinking attention cache per sequence
- **Token sampling** — top-k, top-p, temperature, repetition penalty
- **Memory planning** — allocating and reusing buffers across layers
- **Execution scheduling** — ordering kernel calls for a forward pass
- **Tokenization** — text ↔ token ID conversion

KernelSmith explicitly does not provide these (see DES-006 Non-Goals). A runtime
stack is needed to compose KernelSmith kernels into a functioning LLM inference
engine on RISC-V hardware.

### Goals

1. Define integration patterns for KernelSmith kernels into LLM runtime frameworks
2. Identify the runtime stack best suited for RISC-V deployment scenarios
3. Specify the backend interface between runtime and KernelSmith C API
4. Guide implementation of target profiles for RISC-V RVV hardware

### Non-Goals

- Building a new runtime framework from scratch (use existing ones)
- Adding graph-level optimization to KernelSmith (stays single-kernel scope)
- Runtime JIT compilation (KernelSmith is ahead-of-time only)
- Dynamic dispatch between targets (static profile per build)

### Background

- DES-006 established the C99 API pattern: caller-provided workspace, no malloc,
  reentrant, static target profiles
- DES-001 designed the vector → RVV intrinsic lowering pipeline
- The RISC-V RVV ecosystem is maturing: QEMU supports RVV 1.0, hardware is
  shipping (SiFive P670/P870, SpacemiT K1, Kendryte K230)
- Several lightweight LLM runtimes exist that accept custom compute backends

---

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Integration must preserve KernelSmith's C99 API contract (no internal malloc, reentrant, workspace-based) | Must Have |
| REQ-2 | Runtime must support RISC-V Linux (rv64gcv) as a first-class target | Must Have |
| REQ-3 | Runtime must handle KV-cache management for autoregressive LLM decoding | Must Have |
| REQ-4 | Runtime must support quantized weight formats (at least INT8, INT4) | Must Have |
| REQ-5 | Integration should not require patching the upstream runtime (plugin/backend API) | Should Have |
| REQ-6 | Runtime should support bare-metal / RTOS RISC-V deployment | Should Have |
| REQ-7 | Runtime should allow model-level operator fusion across KernelSmith kernels | Nice to Have |

---

## Design

### 1. Runtime Stack Evaluation

Three runtime architectures are evaluated for integration with KernelSmith
on RISC-V. Each is assessed against the requirements above.

#### Option A: llama.cpp / GGML Backend (Recommended — Near-Term)

**Architecture**:
```
User application
    ↓
llama.cpp (model logic, sampling, KV-cache, tokenizer)
    ↓
GGML (tensor graph, memory allocator, backend dispatch)
    ↓
ggml-kernelsmith backend (new)
    ↓
libkernelsmith.a (KS kernels, RVV-optimized)
    ↓
RISC-V RVV hardware
```

**Why llama.cpp/GGML**:

| Criterion | Assessment |
|-----------|------------|
| LLM-specific features | Best-in-class: KV-cache, sampling, GGUF model format, quantization (Q4_0 through Q8_0), RoPE, GQA/MQA, speculative decoding |
| RISC-V support | Existing rv64gcv backend in GGML (basic); KernelSmith replaces its compute kernels |
| Integration effort | Low — `ggml_backend` C API aligns with KernelSmith's C99 API |
| Community | Largest open-source LLM inference community; active RISC-V contributors |
| Quantization | Native — GGML's block quantization formats are the industry standard for edge LLM |
| Binary size | Minimal — single static binary, no framework dependencies |
| RTOS potential | Partial — llama.cpp assumes Linux libc, but GGML core is portable |

**Integration Point: `ggml_backend`**

GGML's backend API dispatches tensor operations to pluggable implementations.
KernelSmith registers as a custom backend:

```c
// ggml-kernelsmith.h — KernelSmith backend for GGML

#include "ggml-backend.h"
#include <kernelsmith/ks_matmul.h>
#include <kernelsmith/ks_activations.h>

// Backend initialization
ggml_backend_t ggml_backend_kernelsmith_init(void);

// Backend buffer type (workspace management)
ggml_backend_buffer_type_t ggml_backend_kernelsmith_buffer_type(void);
```

**Operation Mapping**:

| GGML Operation | KernelSmith Kernel | Notes |
|----------------|-------------------|-------|
| `GGML_OP_MUL_MAT` | `ks_matmul_f32` / `ks_matmul_f16` / `ks_matmul_i8` | Core compute — 80%+ of LLM FLOPS |
| `GGML_OP_SOFT_MAX` | `ks_softmax` | Attention softmax |
| `GGML_OP_SILU` | `ks_silu_f32` | LLaMA activation |
| `GGML_OP_GELU` | `ks_gelu_f32` | GPT-2/BERT activation |
| `GGML_OP_RMS_NORM` | `ks_rms_norm` | LLaMA normalization |
| `GGML_OP_NORM` | `ks_layer_norm` | GPT-2 normalization |
| `GGML_OP_ADD` | (scalar loop / RVV) | Element-wise, trivial |
| `GGML_OP_ROPE` | (custom or fallback) | KernelSmith doesn't have RoPE yet |

**Workspace Management Bridge**:

GGML manages its own memory arena. The KernelSmith workspace is allocated from
GGML's scratch buffer pool:

```c
static void ks_backend_compute(ggml_backend_t backend,
                                struct ggml_tensor * dst) {
    struct ggml_tensor * src0 = dst->src[0];
    struct ggml_tensor * src1 = dst->src[1];

    switch (dst->op) {
    case GGML_OP_MUL_MAT: {
        size_t M = src0->ne[1], K = src0->ne[0];
        size_t N = src1->ne[1];

        // Query workspace from KernelSmith
        size_t ws_size = ks_matmul_f32_workspace(M, N, K);

        // Allocate from GGML's scratch buffer
        void * workspace = ggml_backend_scratch_alloc(backend, ws_size,
                                                       ks_matmul_alignment());

        ks_matmul_f32(
            (const float *)src0->data, src0->nb[1] / sizeof(float),
            (const float *)src1->data, src1->nb[1] / sizeof(float),
            (float *)dst->data,        dst->nb[1]  / sizeof(float),
            M, N, K,
            workspace, ws_size
        );
        break;
    }
    case GGML_OP_SILU:
        ks_silu_f32((const float *)src0->data,
                    (float *)dst->data,
                    ggml_nelements(src0));
        break;
    // ... other operations
    }
}
```

**Quantization Bridging**:

GGML uses block quantization formats (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0). These
are dequantized to f16/f32 before kernel invocation, or KernelSmith's `ks_matmul_i8`
handles the int8 path directly:

```
GGML Q4_0 weights → dequantize to f16 → ks_matmul_f16 (with f32 accumulator)
GGML Q8_0 weights → cast to i8        → ks_matmul_i8  (with i32 accumulator)
```

For maximum performance, a future KernelSmith `ks_matmul_q4_0` could operate
directly on GGML's block format, avoiding the dequantization step. This is
deferred to Milestone 5+.

---

#### Option B: IREE HAL Backend (Recommended — Long-Term)

**Architecture**:
```
Model (StableHLO / TOSA / torch-mlir)
    ↓ iree-compile
IREE VM module (ahead-of-time compiled)
    ↓
IREE runtime (VM executor, HAL device management, buffer views)
    ↓
HAL device: kernelsmith (new)
    ↓
libkernelsmith.a (KS kernels, RVV-optimized)
    ↓
RISC-V RVV hardware
```

**Why IREE**:

| Criterion | Assessment |
|-----------|------------|
| MLIR alignment | Best — same compiler infrastructure, shared type system, composable passes |
| Graph optimization | Strong — stream-level fusion, buffer planning, tiling at dispatch level |
| RISC-V support | Active — LLVM CPU backend with rv64gcv support, ahead-of-time compilation |
| Integration effort | Medium — HAL driver API is well-documented but has a learning curve |
| Quantization | Good — supports mixed-precision via StableHLO quantization |
| Deployment model | Ahead-of-time compiled VM modules, no JIT, deterministic memory |

**Integration Point: IREE HAL Driver**

IREE's Hardware Abstraction Layer (HAL) defines a device driver interface.
KernelSmith registers as a custom HAL device that intercepts compute dispatches:

```c
// iree_hal_kernelsmith_driver.h

iree_status_t iree_hal_kernelsmith_driver_create(
    iree_string_view_t identifier,
    iree_allocator_t host_allocator,
    iree_hal_driver_t** out_driver);

// The driver creates a device that routes dispatches to KS kernels
// based on the dispatch key (operation name + type signature)
```

**Why Long-Term**: IREE's compiler could eventually invoke KernelSmith's MLIR
passes directly in its compilation pipeline (not just call the C functions at
runtime). This would enable cross-kernel fusion at the MLIR level — e.g., fusing
`ks.matmul` + `ks.relu` into a single tiled loop nest — which is impossible when
kernels are opaque C function calls.

---

#### Option C: Custom Thin Runtime (For Bare-Metal / RTOS)

**Architecture**:
```
Model weights (flat binary, mmap-able)
    ↓
ks_llm_runner (minimal C runtime)
    ├── Layer scheduler (static execution plan)
    ├── Memory planner (pre-computed buffer layout)
    ├── KV-cache manager (ring buffer)
    └── Token sampler (greedy / top-k)
    ↓
libkernelsmith.a (KS kernels, RVV-optimized)
    ↓
RISC-V RVV hardware (bare-metal or RTOS)
```

**When to use**: Deeply embedded RISC-V targets running a fixed model architecture
(e.g., a specific TinyLLaMA or Phi variant). No OS, no dynamic memory, no
filesystem beyond model weight flash storage.

**Scope of custom runtime**:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Layer scheduler | Low | Static loop: for each layer, call fixed kernel sequence |
| Memory planner | Medium | Pre-compute buffer offsets at compile time, single flat allocation |
| KV-cache | Medium | Ring buffer with fixed max sequence length |
| Weight loader | Low | mmap or flash-read into pre-assigned buffer |
| Token sampler | Low | Greedy or temperature-scaled argmax |
| Tokenizer | Medium | BPE lookup table, no regex |

**Total code**: ~2000-4000 lines of C. No dependencies beyond libc (or freestanding).

This option is deferred until KernelSmith has a working RVV target (Milestone 6+)
and a specific embedded RISC-V board is selected for validation.

---

### 2. Recommended Integration Path

```
Phase 1 (Now → Milestone 4):
    Focus on libkernelsmith.a correctness and x86 performance.
    No runtime integration yet — validate kernels in isolation.

Phase 2 (Milestone 5-6):
    Implement RISC-V RVV target profile + lowering pass.
    Write ggml-kernelsmith backend (Option A).
    Validate: run LLaMA-7B on QEMU rv64gcv via llama.cpp.

Phase 3 (Milestone 7+):
    Explore IREE HAL integration (Option B) for graph-level optimization.
    Build custom thin runtime (Option C) for a specific embedded board.

Phase 4 (Post-1.0):
    IREE compiler integration — KernelSmith passes inside IREE's pipeline.
    Cross-kernel fusion at MLIR level.
```

---

### 3. RISC-V RVV Target Profile

To support the runtime integration, KernelSmith needs a concrete RVV target
profile. The following profile targets RISC-V cores with VLEN=256 (e.g.,
SiFive P670-class, SpacemiT K1):

**Profile**: `target/riscv_rvv_256.h`

```c
/* RISC-V RVV VLEN=256 target profile */

#define KS_TARGET_NAME          "riscv-rvv-256"
#define KS_TARGET_ARCH          KS_ARCH_RISCV64

/* Vector unit — VLEN=256 bits */
#define KS_SIMD_WIDTH_BITS      256
#define KS_SIMD_WIDTH_F32       8       /* 256 / 32 */
#define KS_SIMD_WIDTH_F16       16      /* 256 / 16 */
#define KS_SIMD_WIDTH_I8        32      /* 256 / 8  */
#define KS_VECTOR_IS_SCALABLE   1       /* VLA: true VLEN may differ at runtime */

/* Cache hierarchy (typical dual-issue RV64 core) */
#define KS_L1D_SIZE_KB          32
#define KS_L2_SIZE_KB           256
#define KS_L3_SIZE_KB           0       /* many RV64 cores lack L3 */
#define KS_CACHELINE_BYTES      64

/* Memory alignment */
#define KS_PREFERRED_ALIGN      32      /* VLEN/8 = 32 bytes */
#define KS_REQUIRED_ALIGN       1       /* RVV supports unaligned */

/* Matmul tile sizes (conservative for 256KB L2, 32KB L1) */
/* L2 tiles */
#define KS_MATMUL_TILE_M_L2     64
#define KS_MATMUL_TILE_N_L2     64
#define KS_MATMUL_TILE_K_L2     256

/* L1 tiles */
#define KS_MATMUL_TILE_M_L1     8
#define KS_MATMUL_TILE_N_L1     32
#define KS_MATMUL_TILE_K_L1     256

/* Register tile / micro-kernel */
/* With LMUL=2: 16 f32 elements per vregs pair, 4 pairs for accumulators */
#define KS_MATMUL_MR            4       /* rows per micro-kernel */
#define KS_MATMUL_NR            16      /* cols = 2 × VLMAX (LMUL=2) */

/* Feature flags */
#define KS_HAS_FMA              1       /* vfmacc.vv */
#define KS_HAS_F16C             1       /* vfwcvt.f.f.v (f16→f32) */
#define KS_HAS_VNNI             0       /* no dot-product extension yet */
#define KS_HAS_RVV              1
#define KS_RVV_VLEN_MIN         256     /* minimum VLEN this profile targets */
#define KS_RVV_LMUL_DEFAULT     2       /* default LMUL for compute-bound ops */

/* Packing */
#define KS_MATMUL_PACK_B        1
#define KS_MATMUL_PACK_A        0

/* Double buffering — beneficial on simple in-order RV64 cores */
#define KS_MATMUL_DOUBLE_BUFFER 0       /* defer to Milestone 8 */
```

**Tile size rationale**:

| Parameter | Value | Constraint |
|-----------|-------|-----------|
| TILE_M_L2=64 | 64 × 256 × 4 = 64KB | < L2/2 = 128KB |
| TILE_N_L2=64 | Matches M for square tiles | Balanced iteration count |
| TILE_K_L2=256 | 64 × 256 × 4 = 64KB | Fits in L2 alongside B panel |
| TILE_M_L1=8 | 8 × 32 × 4 = 1KB | < L1/2 = 16KB |
| TILE_N_L1=32 | 2 × VLMAX at LMUL=2 | Two vector register groups |
| MR=4 | 4 rows of accumulators | 4 × 2 = 8 vregs for C tile |
| NR=16 | VLMAX at LMUL=2 (SEW=32) | Full vector register group |

---

### 4. LLM Forward Pass Mapping

A typical decoder-only transformer layer maps to KernelSmith kernels as follows:

```
┌─────────────────────────────────────────────────────┐
│ Transformer Decoder Layer                           │
│                                                     │
│  input ──→ [ks_rms_norm] ──→ normed                │
│                                                     │
│  normed ──→ [ks_matmul W_Q] ──→ Q                  │
│  normed ──→ [ks_matmul W_K] ──→ K → KV-cache       │
│  normed ──→ [ks_matmul W_V] ──→ V → KV-cache       │
│                                                     │
│  Q, K, V ──→ [ks_attention] ──→ attn_out           │
│       (internally: matmul + softmax + matmul)       │
│                                                     │
│  attn_out ──→ [ks_matmul W_O] ──→ projected        │
│  projected + input ──→ [add] ──→ residual1          │
│                                                     │
│  residual1 ──→ [ks_rms_norm] ──→ normed2           │
│  normed2 ──→ [ks_matmul W_gate] ──→ gate           │
│  normed2 ──→ [ks_matmul W_up]   ──→ up             │
│  gate ──→ [ks_silu] ──→ activated                   │
│  activated * up ──→ [element-wise mul]              │
│  result ──→ [ks_matmul W_down] ──→ ff_out          │
│  ff_out + residual1 ──→ [add] ──→ output           │
│                                                     │
└─────────────────────────────────────────────────────┘
```

**Kernel call count per layer** (LLaMA-style architecture):
- `ks_matmul`: 7 calls (Q, K, V, O projections + gate, up, down MLP)
- `ks_rms_norm`: 2 calls (pre-attention, pre-MLP)
- `ks_silu`: 1 call (MLP activation)
- `ks_attention`: 1 call (or decomposed into 2× matmul + softmax)
- Element-wise add/mul: 3 calls (residuals + gated MLP)

**Compute dominance**: `ks_matmul` accounts for ~90% of FLOPS. This is
why matmul optimization (tiling, packing, RVV vectorization) is the
highest-priority path for RISC-V performance.

---

### 5. Memory Layout for LLM Inference

The runtime is responsible for buffer allocation. KernelSmith kernels operate
on caller-provided pointers. A typical memory layout for a 7B-parameter LLM:

```
Memory Map (rv64gcv, 8GB DRAM):

┌──────────────────────────────────────┐ 0x0000_0000
│ Model weights (mmap'd, read-only)    │
│ ~7GB for 7B params @ Q4_0            │
│ ~14GB for 7B params @ f16            │
├──────────────────────────────────────┤
│ KV-cache                             │
│ n_layers × 2 × max_seq × d_head     │
│ ~2GB for 32 layers, 4096 seq, f16    │
├──────────────────────────────────────┤
│ Activation buffers (reusable)        │
│ 2-3 buffers of (max_batch × d_model) │
│ ~50MB for batch=1, d=4096, f32       │
├──────────────────────────────────────┤
│ KernelSmith workspace                │
│ Shared across all kernel calls       │
│ ~256KB (dominated by matmul packing) │
├──────────────────────────────────────┤
│ Runtime overhead (tokenizer, etc.)   │
│ ~10MB                                │
└──────────────────────────────────────┘
```

**Key insight**: The KernelSmith workspace is tiny compared to model weights
and KV-cache. A single workspace buffer (sized for the largest matmul in the
model) can be reused across all kernel calls — the workspace is ephemeral
by contract (DES-006).

---

### 6. Performance Considerations for RISC-V

#### Memory Bandwidth

RISC-V platforms typically have lower memory bandwidth than x86/ARM server
chips. LLM inference during decoding is memory-bandwidth-bound (loading weights
for each token). Optimizations:

1. **Weight quantization**: Q4_0/Q8_0 reduces memory traffic 2-8×
2. **KV-cache in f16**: Halves cache memory vs f32
3. **Operator fusion**: Avoids writing/reading intermediate activations
   (requires graph-level runtime, favors IREE long-term)
4. **Double buffering**: Overlaps weight loads with compute (DES-006 §8,
   especially impactful on in-order RV64 cores)

#### Vector Length Agnostic Execution

KernelSmith's RVV lowering (DES-001) generates VLA code that adapts to
runtime VLEN. This means the same `libkernelsmith.a` works across:
- VLEN=128 (minimum spec-compliant)
- VLEN=256 (SpacemiT K1, SiFive P670)
- VLEN=512+ (future high-performance cores)

The target profile's tile sizes should be tuned for the minimum expected
VLEN, with the VLA micro-kernel automatically utilizing wider vectors
when available.

---

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Build full runtime in KernelSmith | Complete control, tight integration | Massive scope increase, duplicates existing work | Rejected |
| Target only IREE | Best MLIR alignment | Steep learning curve, smaller community for LLM-specific features | Deferred to Phase 3 |
| Target only llama.cpp | Fastest path to working LLM | No graph-level optimization, limited to GGML's operation set | Selected for Phase 2 |
| **Layered approach: llama.cpp now, IREE later** | Practical near-term, optimal long-term | Two integration efforts | **Selected** |
| ExecuTorch (Meta) | Good edge deployment story | Less mature RISC-V, heavier PyTorch dependency | Monitored, not targeted |
| TVM runtime | Good RISC-V support | Heavy framework, different compilation model | Not aligned with KernelSmith |
| MNN / NCNN | Lightweight, mobile-focused | Limited LLM support, custom backends harder to add | Not targeted |

---

## Test Strategy

### Phase 2 Validation (ggml-kernelsmith backend)

- [ ] Build llama.cpp with ggml-kernelsmith backend on rv64gcv (QEMU)
- [ ] Run LLaMA-2-7B (Q4_0) inference: correct token generation
- [ ] Run LLaMA-2-7B (Q8_0) inference: correct token generation
- [ ] Perplexity benchmark: ggml-kernelsmith vs ggml-default on WikiText-2
- [ ] Token throughput: tok/s on QEMU rv64gcv (VLEN=256)
- [ ] Token throughput: tok/s on real hardware (if available)
- [ ] Memory usage: peak RSS during inference
- [ ] All KernelSmith C API tests still pass (regression)

### Correctness Criteria

- Perplexity within 0.1 of reference GGML backend (f32 path)
- Identical token generation for greedy decoding (deterministic)
- No memory leaks (Valgrind/ASAN clean)
- Thread safety: concurrent inference sessions with independent contexts

---

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| GGML backend API changes upstream | Medium | Medium | Pin to GGML release tag, track breaking changes |
| RVV QEMU performance not representative of real hardware | Medium | High | Validate on real RV64 board when available; use cycle-accurate sim |
| KernelSmith kernel coverage gaps vs GGML op set | High | Medium | Fall back to GGML's default implementation for unsupported ops |
| Quantization format mismatch (GGML Q4 vs KS i8) | Medium | Medium | Implement dequant-then-compute bridge; add native Q4 kernel later |
| RISC-V hardware availability | High | Low | QEMU + SiFive dev boards; SpacemiT K1 boards are shipping |

---

## Open Questions

- [ ] Should KernelSmith add a `ks_matmul_q4_0` that operates directly on GGML's
      block-quantized format, or always dequantize first?
- [ ] Should the ggml-kernelsmith backend live in the KernelSmith repo or be
      contributed upstream to llama.cpp?
- [ ] What is the minimum RVV VLEN that KernelSmith should support? (128 is spec
      minimum, but 256 is the practical floor for useful LLM throughput)
- [ ] Should KernelSmith add RoPE (rotary positional embedding) as a kernel, or
      leave it to the runtime?

---

## Dependencies

- KernelSmith Milestone 4+ (vectorized matmul with packing)
- KernelSmith Milestone 6 (RISC-V RVV target, `--ks-lower-to-rvv` pass)
- llama.cpp / GGML (ggml-backend API, GGUF model format)
- RISC-V cross-compilation toolchain (riscv64-unknown-linux-gnu-gcc)
- QEMU rv64gcv (RVV 1.0 support)
- LLVM 18+ with RISC-V V extension codegen

---

## Implementation Plan

### Stage 1: RVV Target Profile (Milestone 6 prerequisite)

- [x] Write `target/riscv_rvv_256.h` profile header
- [ ] Write `target/riscv_rvv_128.h` profile header (minimum VLEN)
- [ ] Validate profile: build libkernelsmith.a with RVV profile, run on QEMU
- [ ] Benchmark: RVV matmul vs generic on QEMU

### Stage 2: GGML Backend Skeleton

- [ ] Create `backends/ggml-kernelsmith/` directory
- [ ] Implement `ggml_backend_kernelsmith_init()`
- [ ] Map `GGML_OP_MUL_MAT` → `ks_matmul_f32`
- [ ] Map `GGML_OP_SILU` → `ks_silu_f32`
- [ ] Map `GGML_OP_RMS_NORM` → `ks_rms_norm`
- [ ] Fallback: unsupported ops → GGML CPU backend
- [ ] Build and link against llama.cpp

### Stage 3: Validation

- [ ] Run LLaMA-2-7B Q4_0 on x86 (KernelSmith backend vs default)
- [ ] Cross-compile for rv64gcv, run on QEMU
- [ ] Perplexity comparison
- [ ] Token throughput measurement

### Stage 4: IREE Exploration (Post-Milestone 7)

- [ ] Prototype IREE HAL driver for KernelSmith
- [ ] Compile simple model (single matmul) through IREE → KS backend
- [ ] Evaluate graph-level fusion opportunities

---

## Related Documents

- [DES-006: Kernel Library Architecture](DES-006-kernel-library-architecture.md) — C API, memory management, target profiles
- [DES-001: Vector → RVV Lowering](DES-001-vector-operations-lowering.md) — RVV intrinsic lowering pipeline
- [DES-002: MatMul Kernel](DES-002-matmul-kernel.md) — matmul pipeline design
- [RVV Target Specification](../../specs/targets/riscv-rvv.md) — RVV ISA mapping
- [Attention Specification](../../specs/kernels/attention.md) — attention kernel design
- [ROADMAP.md](../../ROADMAP.md) — milestone plan
