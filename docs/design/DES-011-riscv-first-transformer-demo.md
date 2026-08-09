# DES-011: RISC-V First Transformer Demo Strategy

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Superseded in part by DES-016 |
| **Author** | KernelSmith Team |
| **Created** | 2026-06-06 |
| **Related** | DES-006, DES-009, DES-010, DES-016, ROADMAP.md |

> **Sequencing update (2026-08-09):** DES-016 supersedes this document's
> custom-runner-first ordering. The kernel set, static-library boundaries, and
> QEMU correctness principles remain relevant, but the first demo now uses a
> bounded llama.cpp integration smoke before quantized RVV and system
> simulation work.

## Context

### Problem Statement

KernelSmith is a compiler-assisted C kernel library, not a full model runtime.
The project needs an end-to-end demo that proves generated RISC-V RVV kernels
are useful for quantized inference without forcing the project to absorb the
complexity of a production LLM runtime too early.

### Background

The edge inference ecosystem is moving toward:

- ahead-of-time compiled runtimes with static memory planning;
- quantized CPU kernels for INT8, INT4, and mixed-precision inference;
- fused dequantization plus matrix multiply/vector dot kernels;
- backend integration into runtimes such as ExecuTorch, XNNPACK, TFLite Micro,
  IREE, ONNX Runtime, and llama.cpp/GGML.

RISC-V RVV is a credible differentiation target because the software ecosystem
is less mature than Arm NEON while RVV-capable edge boards and SoCs are growing.
Arm remains important, but KernelSmith's first strategic wedge is RISC-V.

## Requirements

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Keep the shipped artifact a static C library plus headers. | Must Have |
| REQ-2 | Make RISC-V RVV the first optimized target and demo platform. | Must Have |
| REQ-3 | Prove quantized transformer inference with no internal allocation. | Must Have |
| REQ-4 | Prioritize batch-1 decode kernels: quantized dot/GEMV before large GEMM. | Must Have |
| REQ-5 | Fuse INT4/INT8 dequantization with compute; avoid materializing dequantized weights. | Must Have |
| REQ-6 | Keep llama.cpp/GGML integration as a follow-on credibility demo. | Should Have |
| REQ-7 | Preserve a small, auditable demo that can run under QEMU. | Should Have |

## Design

### Overview

KernelSmith should use a two-stage demo strategy:

1. **Minimal llama2.c-style runner first.** A small C example owns model loading,
   token generation, memory planning, and calls into KernelSmith kernels.
2. **llama.cpp/GGML integration second.** Once the kernel ABI and RVV quantized
   kernels are proven, integrate selected kernels into a larger runtime for
   ecosystem credibility and comparative benchmarks.

This keeps the first milestone focused on KernelSmith's core value: generated,
workspace-safe, quantized RISC-V kernels.

### Product Positioning

KernelSmith is not intended to compete with full deployment stacks. Its primary
role is:

> A RISC-V-first, compiler-assisted, static C kernel backend for quantized edge
> inference.

Framework integrations are downstream consumers of the C kernels, not the
primary artifact.

### Demo Kernel Set

The first transformer demo should prioritize decode-path kernels:

| Kernel | Purpose | Notes |
|--------|---------|-------|
| `ks_matvec_w4a8` | INT4 weights x INT8 activations matrix-vector | Primary batch-1 decode kernel. |
| `ks_dot_w4a8` | Quantized dot product primitive | Useful for testing and GEMV decomposition. |
| `ks_matmul_w4a8` | Small-batch/prefill path | Follows GEMV once quantized packing is stable. |
| `ks_rms_norm_f32` | Transformer normalization | Required by llama-style blocks. |
| `ks_softmax_f32` | Attention probabilities | Numerically stable max-subtract form. |
| `ks_silu_f32` | Feed-forward activation | Already in C API for f32. |
| `ks_add_f32` | Residual connection | Simple but required for end-to-end blocks. |

The quantized compute path should unpack/dequantize inside the dot/GEMV/GEMM
loop. Standalone dequantization ops remain useful for tests and IR semantics,
but they are not the performance path for LLM weights.

### Data Flow

```
TinyStories-style model
  -> scripts/quantize_model.py
  -> W4A8 KernelSmith model file
  -> examples/runner.c
  -> libkernelsmith.a
  -> qemu-riscv64 / RVV board
```

### llama2.c-Style Runner Scope

The runner should stay intentionally small:

- single-file or small multi-file C example;
- static workspace planning;
- no dynamic dispatch;
- no dependency on MLIR or C++;
- model format tailored to the demo, not a general interchange format;
- deterministic greedy decoding path for validation.

### llama.cpp/GGML Follow-On Scope

The later integration should be a backend proof of concept that replaces
selected quantized dot/GEMV/GEMM kernels. It should not block the first demo.

The integration should compare:

- llama.cpp scalar baseline;
- llama.cpp existing RVV kernels where available;
- KernelSmith-generated RVV kernels.

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Full llama.cpp integration first | Recognizable ecosystem demo; direct comparison to an industry baseline | Large surface area, upstream churn, GGML format complexity, harder to isolate KernelSmith value | Rejected for first demo |
| Minimal llama2.c-style runner first | Small, auditable, QEMU-friendly, proves C ABI and generated kernels directly | Less ecosystem credibility than llama.cpp | **Selected** |
| Build a custom full runtime | Maximum control | Duplicates work outside KernelSmith's kernel-library mission | Rejected |
| ExecuTorch/TFLite integration first | Strong production relevance | Requires delegate/runtime integration before kernels are proven | Deferred |

## Test Strategy

- Lit tests for quantized ops, verifiers, and lowering patterns.
- C API tests for `ks_dot_w4a8`, `ks_matvec_w4a8`, and required f32 helper ops.
- NumPy reference validation for quantized dot/GEMV/GEMM.
- QEMU multi-VLEN correctness for generated RVV binaries.
- Demo-level deterministic token test for a tiny prompt/model pair.
- Benchmark output for scalar generic vs RVV generated kernels.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Demo becomes a runtime project | High | Medium | Keep runner minimal and document non-goals. |
| GEMM-only work misses batch-1 decode bottlenecks | High | Medium | Prioritize dot/GEMV before W4A8 GEMM. |
| llama.cpp integration churn consumes kernel work | Medium | Medium | Defer until the standalone C ABI is stable. |
| QEMU benchmark results misrepresent real hardware | Medium | Medium | Treat QEMU as correctness-first; validate performance on boards when available. |
| INT4 format choices diverge from GGML/KleidiAI conventions | Medium | Medium | Document packing and keep conversion scripts explicit. |

## Open Questions

Draft [DES-015](DES-015-quantized-layout-abi-and-integration-conversion-paths.md)
proposes native symmetric per-group W4A8, per-group positive weight scales,
per-tensor activation scale, and a KernelSmith-specific model container.
Its GGML path decodes and requantizes weights; it is not a lossless repack.

- [ ] Which first INT4 format should be used: simple symmetric W4A8 or a
      GGML-compatible block format? (DES-015 proposes native symmetric W4A8.)
- [ ] Should scales be per-channel, per-block, or both in the first ABI?
      (DES-015 proposes per-group weights and per-tensor activations.)
- [ ] Should the first model file format be KernelSmith-specific or a restricted
      importer from llama2.c checkpoints? (DES-015 proposes KernelSmith-specific
      first, with a requantizing GGML importer for M8.)
- [ ] What RVV board should serve as the first performance reference?

## Dependencies

- Generated C library path for RISC-V RVV object files.
- Workspace materialization and allocation checking.
- Quantized dot/GEMV/GEMM dialect and lowering support.
- RISC-V cross-compiler and QEMU correctness harness.

## Implementation Plan

### Phase 1: RISC-V generated-kernel foundation

- [ ] Wire generated RVV objects into `libkernelsmith.a`.
- [ ] Add `--ks-alloc-check` as a hard pipeline gate.
- [ ] Integrate `--ks-materialize-pack-workspace` into the RVV build path.
- [ ] Validate generated kernels under QEMU.

### Phase 2: Quantized decode kernels

- [ ] Add quantized dot/GEMV C APIs.
- [ ] Define W4A8 packing and scale layout.
- [ ] Lower fused unpack/dequantize plus compute for RVV.
- [ ] Validate against NumPy references.

### Phase 3: Minimal transformer runner

- [ ] Add `examples/runner.c`.
- [ ] Add `scripts/quantize_model.py`.
- [ ] Add `scripts/demo_rvv.sh`.
- [ ] Run deterministic prompt validation under QEMU.

### Phase 4: llama.cpp/GGML integration proof

- [ ] Map KernelSmith quantized layouts to GGML-compatible call shapes.
- [ ] Replace selected dot/GEMV/GEMM kernels in a focused integration branch.
- [ ] Benchmark against llama.cpp scalar and RVV baselines.
