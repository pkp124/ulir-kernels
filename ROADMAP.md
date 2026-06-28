# KernelSmith Roadmap

> **Vision**: Ship a RISC-V-first C kernel library (`libkernelsmith.a` +
> headers) that provides optimized quantized ML inference kernels for **edge,
> embedded, and physical AI** devices. RISC-V RVV is the first optimized target;
> ARM NEON is a secondary port once the RVV path is proven.
> No runtime dependencies. No MLIR knowledge required by users. Works with any
> toolchain, any RTOS, any task execution engine.
>
> **Domain focus**: Quantized inference on resource-constrained RISC-V edge
> SoCs, SBCs, robotics/physical-AI devices, and embedded systems. Quantization
> is a core feature, not an afterthought. Phone-class ARM devices remain relevant
> but are not the first strategic wedge.
>
> **Architecture reference**: [DES-006](docs/design/DES-006-kernel-library-architecture.md)
> **Demo strategy**: [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md)

## Product Positioning

KernelSmith is a **kernel backend**, not a full model deployment framework.
MLIR is internal build-time machinery; users consume a static C library with
stable headers and caller-provided workspace. Framework integrations such as
llama.cpp/GGML, ExecuTorch, TFLite Micro, or IREE are downstream consumers of
the C kernels.

The first end-to-end proof is a minimal llama2.c-style RISC-V transformer
runner. A full llama.cpp/GGML integration is a follow-on credibility demo after
the generated RVV kernel ABI, quantized layouts, and workspace model are stable.

---

## Milestone 0: Dialect Infrastructure (Complete)

**Status**: Done

- KS dialect with 13 operations defined in TableGen
- Verifiers for 9 operations (matmul, batch_matmul, conv2d, attention,
  softmax, layer_norm, rms_norm, reduce_sum, reduce_max)
- `ks-opt` CLI tool (parse/print + pass driver)
- Lit tests for parse round-trip and verifier errors
- Build system (CMake + LLVM/MLIR 21 integration)

---

## Milestone 1: C API + Reference Library (Complete)

**Status**: Done

**Goal**: Ship a working `libkernelsmith.a` with stable C headers. Handwritten
reference implementations. No MLIR in the critical path yet.

**Why this first**: Users can start integrating against the API immediately.
The MLIR-generated implementations replace the reference code incrementally
in later milestones — behind the same stable interface.

**Deliverables**:
- `include/kernelsmith/ks_matmul.h` — matmul C API (f32 baseline)
- `include/kernelsmith/ks_activations.h` — relu, gelu, silu C API
- `include/kernelsmith/ks_common.h` — error codes, version, target info
- `target/generic.h` — portable C target profile (no SIMD, no packing)
- Handwritten reference `ks_matmul_f32` (triple-loop, single-level tiling)
- Handwritten reference `ks_relu_f32`, `ks_gelu_f32`, `ks_silu_f32`
- `libkernelsmith.a` built with CMake
- C test program linking against the library, validated against numpy
- Workspace query functions (return 0 for generic profile)

**Tasks**:
1. Write public C headers following [DES-006 API design](docs/design/DES-006-kernel-library-architecture.md#2-c-api-design-matmul-focus)
2. Write `target/generic.h` target profile
3. Implement reference `ks_matmul_f32` (tiled triple loop, stride support)
4. Implement reference activations (element-wise scalar C)
5. CMake: build as static library, install headers
6. C test: correctness vs numpy for matmul (square, rectangular, non-aligned dims)
7. C test: correctness for activations

---

## Milestone 2: MLIR Lowering — Activations

**Status**: Partial. The pass and lit tests are implemented; generated object
integration into `libkernelsmith.a` is still pending.

**Goal**: Replace handwritten activation functions with MLIR-generated code.
Proves the pass infrastructure works end-to-end.

**Deliverables**:
- `KSLowerActivationsPass` (`--ks-lower-activations`)
- Pass registration infrastructure (`Passes.h`, `Passes.td`)
- Generated `.o` files replacing handwritten activations
- Lit tests for each transformation
- C library tests still pass (same API, MLIR-generated implementation)

**Tasks**:
1. Create pass infrastructure: `include/KernelSmith/Passes/Passes.h`, `Passes.td`
2. Implement `--ks-lower-activations`:
   - `ks.relu` -> `arith.maxf(input, zero)`
   - `ks.gelu` -> `math.erf` + arith
   - `ks.silu` -> `math.exp` + arith (`x * sigmoid(x)`)
3. Tighten TableGen type constraints: `AnyTensor` -> `KS_FloatTensor` / `KS_NumericTensor`
4. Pipeline: MLIR -> bufferize -> LLVM IR -> .o (for host target)
5. Replace handwritten activation .o files with generated ones
6. Verify: same C tests pass, same numerical results
7. Lit tests: `tests/lit/Passes/lower-activations.mlir`

---

## Milestone 3: MLIR Lowering — MatMul (Generic Target)

**Status**: Current / partial. `--ks-lower-to-linalg`, `--ks-tile`, and
`--ks-alloc-check` are implemented and tested. Bufferization, generated object
integration, and C library replacement remain open.

**Goal**: Replace handwritten matmul with MLIR-generated code. Single-level tiling,
no packing, generic target profile. Proves the full ks -> linalg -> LLVM pipeline.

**Deliverables**:
- `KSLowerToLinalgPass` (`--ks-lower-to-linalg`)
- `KSTilePass` (`--ks-tile`) with profile-driven tile sizes
- ✓ `KSAllocCheckPass` (`--ks-alloc-check`) — fail if any `memref.alloc` survives
- Bufferization config: all buffers are function arguments, zero internal malloc
- Generated matmul for generic target
- C library tests still pass

**Tasks**:
1. Implement `--ks-lower-to-linalg` (ks.matmul -> linalg.matmul + tensor.empty)
2. Add accumulator type attribute to ks.matmul for mixed-precision (f16->f32, i8->i32)
3. Implement `--ks-tile` (single-level, reads tile sizes from pass options)
4. Configure one-shot-bufferize: function-argument buffers only
5. ✓ Implement `--ks-alloc-check` (reject stray memref.alloc as hard error)
6. Build script: read target/generic.h, translate to pass options
7. Pipeline: ks.matmul -> linalg -> tile -> bufferize -> LLVM IR -> .o
8. Replace handwritten matmul with generated version
9. Verify: same C tests pass
10. Lit tests for each pass in isolation

---

## Milestone 4: RISC-V RVV Target (Primary) — Validation Complete

**Goal**: End-to-end optimized matmul and activations for RISC-V RVV. This is
the project's primary hardware target. Multi-level tiling, B packing, and
RVV-specific vectorization.

**Why RVV first**: Edge RISC-V SoCs (e.g., T-Head C908/C910, SiFive X280,
Kendryte K230) are the primary deployment targets. RVV's vector-length-agnostic
(VLA) model benefits from explicit vector-dialect lowering, profile-driven
tiling, and packing; relying on scalar loops plus autovectorization is
insufficient for high-performance VLA code.

**Design**: [DES-009](docs/design/DES-009-m4-rvv-lowering.md)

**Deliverables** (completed ✓ / pending …):
- ✓ `target/riscv_rvv_256.h` — RVV target profile (VLEN=256, LMUL=4, f32)
- ✓ `KSPackPass` (`--ks-pack`) — B operand packing into `[N/NR, K, NR]` layout
- ✓ Multi-level tiling via two `--ks-tile` invocations (L2 then register)
- ✓ `KSVectorizePass` (`--ks-vectorize`) — linalg → vector dialect
- ✓ `KSLowerToRVVPass` (`--ks-lower-to-rvv`) — full LLVM dialect lowering pipeline
- ✓ `scripts/compile-rvv.sh` — driver script (ks-opt + mlir-translate + llc)
- ✓ Lit tests: pack, vectorize, lower-to-rvv
- ✓ QEMU runner updated for multi-VLEN correctness + benchmark
- ✓ QEMU correctness validation with golden-backed host/RVV comparisons
- ✓ Benchmark: generic/reference vs RVV report in a reproducible JSON format
- ✓ Tail handling policy for packed non-divisible N via clear diagnostics

**Tasks**:
1. ✓ Write `target/riscv_rvv_256.h` target profile
2. ✓ Implement `--ks-pack` (linalg.pack for B operand)
3. ✓ Extend `--ks-tile` to multi-level (L2 + register tile via two invocations)
4. ✓ Implement `--ks-vectorize` (linalg → vector.contract + transfer ops)
5. ✓ Implement `--ks-lower-to-rvv` (full bufferize→cf→vector→LLVM pipeline)
6. ✓ Set up cross-compilation: `scripts/compile-rvv.sh` driver
7. ✓ QEMU test runner: multi-VLEN correctness and benchmark support
8. ✓ Lit tests for all three new passes
9. ✓ Run QEMU correctness tests on actual RISC-V binary
10. ✓ Benchmark reporting for generic/reference vs RVV paths

---

## Milestone 5: RISC-V Quantization Foundation

**Goal**: Establish the quantized arithmetic and ABI needed for RISC-V edge
inference, with batch-1 transformer decode as the first optimization target.

**Why now**: Edge inference is quantization-first. INT8 remains the baseline for
general edge models, while on-device LLM inference depends on INT4 or other
low-bit weight formats. For batch-1 transformer decode, quantized dot/GEMV is
often more important than large GEMM because the workload is memory-bandwidth
bound.

**Deliverables**:
- `ks.quantize` / `ks.dequantize` ops for semantic tests and explicit conversion
- Quantized dot/GEMV C APIs:
  - `ks_dot_i8`
  - `ks_matvec_i8`
  - `ks_dot_w4a8`
  - `ks_matvec_w4a8`
- `ks_matmul_i8` for small-batch/prefill and non-transformer workloads
- Fused RVV lowering for unpack/dequantize + dot/GEMV/GEMM
- Block or per-channel scale layout documented in the public ABI
- NumPy references for i8 and W4A8 dot/GEMV/GEMM

**Tasks**:
1. ✓ Add `ks.quantize` and `ks.dequantize` ops (TableGen + verifier + lit tests)
2. Add accumulator/quantization metadata needed for i8 and W4A8 lowering
3. Define C APIs and packed layouts for i8 and W4A8 dot/GEMV
4. Implement INT8 lowering: i8 -> i32 accumulate -> requantize where needed
5. Implement W4A8 fused unpack/dequantize + i32/f32 accumulation for RVV
6. Add RVV i8/W4A8 tile parameters and pack factors to target profiles
7. Add NumPy validation for quantized dot/GEMV/GEMM
8. Add lit tests for quantized lowering and diagnostics

---

## Milestone 6: Transformer Minimum Kernel Set

**Goal**: Add the small set of non-matmul kernels needed for a minimal
llama-style transformer block on RISC-V RVV.

**Why before broad operator coverage**: The first end-to-end proof is a
quantized transformer demo, not a general vision model suite. The decode path
needs normalization, softmax, residual add, activation, and quantized GEMV/dot.

**Design**: [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md)

**Deliverables**:
- `ks.add` / `ks.mul` element-wise ops and C APIs
- Lowering and C API for `ks.rms_norm`
- Numerically stable `ks.softmax` lowering and C API
- RVV-friendly `ks_silu_f32` path for transformer feed-forward layers
- C API smoke tests and NumPy validation for each transformer helper kernel
- QEMU correctness for the transformer helper kernels where toolchains exist

**Tasks**:
1. Add `ks.add` and `ks.mul` ops with broadcasting rules, verifiers, and lit tests
2. Add C APIs for `ks_add_f32`, `ks_mul_f32`, `ks_rms_norm_f32`, and `ks_softmax_f32`
3. Lower add/mul/silu to vectorizable linalg or loop forms
4. Lower RMSNorm with f32 accumulation and explicit epsilon behavior
5. Lower softmax using max-subtract-exp-sum-divide for numerical stability
6. Add functional validator coverage for transformer helper kernels
7. Add QEMU tests for generated RVV helper kernels

---

## Milestone 7: Minimal RISC-V Transformer Demo

**Goal**: Demonstrate a quantized llama-style transformer running end-to-end on
RISC-V RVV using KernelSmith-generated kernels.

**Why this demo first**: A minimal llama2.c-style runner is small, auditable,
QEMU-friendly, and aligned with KernelSmith's static C library objective. It
proves the C ABI, workspace model, quantized layouts, and RVV generated kernels
without taking on the full llama.cpp/GGML runtime surface.

**Design**: [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md)

**Deliverables**:
- Minimal `examples/runner.c` inspired by llama2.c
- KernelSmith-specific quantized model format for the demo
- `scripts/quantize_model.py` for W4A8 conversion
- `scripts/demo_rvv.sh` for cross-compilation and QEMU execution
- Deterministic prompt validation
- Correctness and benchmark comparison against scalar generic kernels

**Integration points**:

```
runner component       -> KernelSmith C API
--------------------      --------------------------------------------
attention/FFN matvec   -> ks_matvec_w4a8() / ks_dot_w4a8()
prefill matmul         -> ks_matmul_w4a8()
rmsnorm                -> ks_rms_norm_f32()
softmax                -> ks_softmax_f32()
silu activation        -> ks_silu_f32()
residual add           -> ks_add_f32()
```

**Demo model**: TinyStories-class model, quantized to W4A8 with group/block
scales. The model should be small enough for QEMU correctness runs and useful
enough to produce recognizable text on RVV hardware.

**Tasks**:
1. Define the demo model container and W4A8 packing format
2. Add quantization script for weights and scale metadata
3. Implement the minimal runner with static workspace planning
4. Replace decode-path compute with KernelSmith C API calls
5. Add deterministic prompt/token validation
6. Add QEMU run script and benchmark output
7. Document limitations and follow-on llama.cpp/GGML integration path

**Demo invocation**:

```bash
# Quantize model (on host)
python scripts/quantize_model.py \
  --input tinystories-15m.bin --output tinystories-15m.ksmodel \
  --scheme w4a8 --group-size 64

# Cross-compile runner + libkernelsmith for RVV
riscv64-unknown-linux-gnu-gcc -o ks-run examples/runner.c \
  -Lbuild-rvv/lib -lkernelsmith -march=rv64gcv

# Generate text on QEMU
qemu-riscv64 -cpu rv64,v=true,vlen=256 \
  ./ks-run tinystories-15m.ksmodel \
  -p "Once upon a time" -n 128
```

**Validation**: Compare deterministic greedy output against a Python or scalar C
reference using the same quantized model. Track tokens/sec separately for QEMU
and real RVV hardware; QEMU is primarily a correctness target.

---

## Milestone 8: llama.cpp / GGML Integration Proof

**Goal**: Demonstrate that KernelSmith's generated RVV kernels can be consumed
by a widely recognized LLM runtime without making llama.cpp the primary product.

**Why after the minimal runner**: llama.cpp/GGML brings model loading, GGUF
formats, threading, many quantization formats, and upstream churn. Integrating
too early would obscure whether KernelSmith's own kernel ABI and codegen are
correct. After the small runner proves the kernels, llama.cpp provides an
industry baseline and ecosystem credibility.

**Deliverables**:
- Focused integration branch or example that replaces selected GGML RVV kernels
- Mapping between KernelSmith W4A8/i8 layouts and GGML-compatible call shapes
- Benchmarks against llama.cpp scalar and existing RVV baselines
- Documentation of what remains outside KernelSmith's scope

**Tasks**:
1. Identify GGML quantized dot/GEMV/GEMM entry points that match KernelSmith kernels
2. Add layout conversion or direct packing support where needed
3. Replace a small set of RVV kernels behind build flags
4. Benchmark prompt processing and decode throughput
5. Compare accuracy/perplexity against the unmodified runtime
6. Document upstreamability gaps and API changes needed for stable integration

---

## Milestone 9: RISC-V Operator Coverage and Hardening

**Goal**: Broaden RISC-V edge model coverage and harden the generated-kernel
library after the transformer path is proven.

**Tasks**:
- RISC-V vision operator track:
  - `ks.depthwise_conv2d` op (NHWC input, HW1C filter)
  - direct depthwise lowering (no im2col for memory-constrained targets)
  - RVV vectorization along the channel dimension
  - `ks.avg_pool2d` / `ks.max_pool2d`
  - regular `ks.conv2d` lowering (direct or im2col + matmul based on profile)
- Double buffering pass (`--ks-double-buffer`) for memory-bound kernels
- Software prefetch hints for RVV and ARM
- Kernel fusion (matmul + bias + activation, conv + bn + relu)
- Profile-guided tile size tuning (benchmark harness + parameter sweep)
- Fuzz testing of C API (invalid args, edge cases, large dimensions)
- Memory sanitizer (ASAN/MSAN) CI integration
- Static memory analysis: worst-case workspace bounds for RTOS environments
- Bare-metal validation (no libc dependency in generated code)
- Documentation: API reference, integration guide, performance tuning guide

---

## Milestone 10: Frontend Ingestion — KernelSmith as a Lowering Target

**Status**: Planned

**Goal**: Make the `ks` dialect and the linalg/vector path beneath it a
consumable backend target for higher-level kernel frontends — TileLang-style
tile DSLs, Torch-MLIR, IREE, and other linalg/StableHLO producers — so
KernelSmith delivers RVV-quality codegen without owning an authoring language.

**Why**: The kernel-DSL community has converged on MLIR progressive lowering as
the shared substrate. KernelSmith's differentiation is RVV/edge/quantized
codegen and a zero-dependency C ABI, not a new authoring surface. The
highest-leverage way to align with the ecosystem is to *consume* its frontends
rather than compete with them. KernelSmith stays a **backend**: frontends own
the user-facing authoring experience; KernelSmith owns RVV lowering, packing,
quantized layouts, and the static C ABI.

**Design**: DES-0xx (planned) — ingestion boundary and supported-op contract

**Deliverables**:
- Documented ingestion contract: which inputs KernelSmith accepts (`ks` ops plus
  a supported subset of `linalg` on tensors) and which it rejects
- A `linalg` ingestion / raising path so external `linalg` producers reach the
  RVV pipeline without first lowering through `ks` ops
- TileLang-style tile-IR bridge investigation: map tile-level constructs onto
  `ks` tiling/packing (or onto `linalg` + the existing `ks` passes)
- Conformance lit tests: representative frontend output → RVV lowering
- Design doc defining the ingestion boundary and op contract

**Tasks**:
1. Enumerate the input op contract (`ks` + supported `linalg`) and document it
2. Add/verify a direct `linalg`-on-tensors entry point into the tile/pack/vectorize pipeline
3. Prototype a TileLang-style tile-IR → `ks`/`linalg` bridge on one matmul kernel
4. Add conformance lit tests for at least one external-frontend lowering path
5. Write DES-0xx recording the ingestion boundary and supported-op contract

---

## Milestone 11: User-Expressible Schedules

**Status**: Planned

**Goal**: Give advanced users an optional, declarative way to express schedules
(tile sizes, packing, loop order, unroll factors, vectorization width / LMUL
hints) that drives the existing pass pipeline — without requiring MLIR knowledge
and without replacing the default profile-driven path.

**Why**: Algorithm/schedule separation (Halide → TVM TensorIR → TileLang) is the
central idea of the modern kernel-DSL community. KernelSmith already separates
compute (`ks` ops) from transforms (`--ks-tile`/`--ks-pack`/`--ks-vectorize`
passes); today the schedule is *implicit* in the target-profile tables. Making
it *explicit and user-overridable* is the natural alignment step and unlocks
per-kernel tuning plus a concrete search space for autotuning (Milestone 9).

**Positioning / non-goals**: This is a schedule *specification* layer, **not** a
general imperative kernel-authoring DSL. The default C-library consumer still
writes zero MLIR and zero schedules — schedules are opt-in for performance
engineers and for the autotuner. KernelSmith does not become Triton/TIR; it
gains an explicit schedule surface over the transforms it already has.

**Design**: DES-0xx (planned) — schedule model vs TVM TensorIR / TileLang;
records the "schedule spec, not authoring DSL" decision

**Deliverables**:
- Schedule recipe format (declarative; YAML/TOML or a thin Python API) keyed per
  op/target, expressing: L2 tile sizes, register tile sizes, pack on/off +
  layout, loop order, unroll factors, and vectorization width / LMUL hints
- Schedule → pass-pipeline compiler: a recipe lowers to concrete
  `--ks-tile`/`--ks-pack`/`--ks-vectorize` options, overriding profile defaults
  only where the recipe specifies them
- Schedule validation against the target profile (reject tile sizes exceeding
  VLMAX, illegal pack factors, etc.) with clear diagnostics
- Round-trip and negative lit/unit tests for schedule application
- Integration hook for Milestone 9 autotuning: the schedule space is the search space
- Design doc comparing the schedule model to TVM TensorIR / TileLang

**Tasks**:
1. Define the schedule recipe schema and its mapping to existing pass options
2. Implement the recipe → pass-pipeline compiler (recipe overrides profile defaults)
3. Add schedule validation against target profiles with actionable diagnostics
4. Add round-trip and negative tests for schedule application
5. Expose the schedule space to the autotuner harness (Milestone 9)
6. Write DES-0xx comparing the model to TVM TensorIR / TileLang and recording the decision

---

## Future Targets (Not Prioritized)

These targets may be added after the core edge pipeline is proven:

| Target | Notes |
|--------|-------|
| ARM NEON | Secondary production port for phones, Raspberry Pi, and Arm SBCs after RVV transformer kernels are proven. |
| ARM SVE | Scalable vectors, similar to RVV. Relevant for server ARM (Graviton). |
| x86 AVX2/AVX-512 | Server/desktop. Existing `target/x86_avx2.h` profile can be used. |
| ARM Cortex-M | MCU/RTOS. Scalar only, validates freestanding story for TinyML. |
| NPU/DSP offload | Vendor-specific. Would require new dialect extensions. |

---

## Design Documents

| ID | Title | Scope |
|----|-------|-------|
| [DES-006](docs/design/DES-006-kernel-library-architecture.md) | Kernel Library Architecture | C API, memory mgmt, tiling, packing, target profiles |
| [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md) | RISC-V First Transformer Demo Strategy | Minimal llama2.c-style demo first; llama.cpp/GGML integration later |
| [DES-012](docs/design/DES-012-riscv-simulation-verification.md) | RISC-V Simulation Verification | QEMU-first generated-kernel verification with Spike/gem5/Renode follow-ons |
| [DES-002](docs/design/DES-002-matmul-kernel.md) | MatMul Kernel (TDD) | MLIR pipeline design (partially superseded by DES-006) |
| [DES-001](docs/design/DES-001-vector-operations-lowering.md) | Vector → RVV Lowering | RVV-specific intrinsic mapping |
| ~~DES-005~~ | ~~Library Packaging (v1)~~ | Deleted — was superseded by DES-006 |

## Target Profile Specification

See [specs/targets/system-description.md](specs/targets/system-description.md) for the
target profile format, validation rules, and provided profiles.
