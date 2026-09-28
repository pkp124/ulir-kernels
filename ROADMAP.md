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
> **Demo strategy**: [DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md)
> supersedes the sequencing in [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md)

## Where the project is

M0, M1, M4, M5, and M6 are done. M2 and M3 have working passes
(`--ks-lower-activations`, `--ks-lower-to-linalg`, `--ks-tile`,
`--ks-alloc-check`). The handwritten matmul and activation objects in
`libkernelsmith.a` have not been replaced by generated objects. That swap is
open and is not the active milestone.

**Active work is M7.** [DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md)
and [TASK-025](tasks/TASK-025.md) specify a llama.cpp host smoke that routes
one f32 operation through the public C API. The design is under review.
[TASK-026](tasks/TASK-026.md) (pin the runtime and a CI-sized model) has not
started.

The dialect has 20 operations. The [README kernel table](README.md#supported-kernels)
is the list of what parses, verifies, lowers, and has a C API. Day-to-day
status is [tasks/MILESTONES.md](tasks/MILESTONES.md).

## Product Positioning

KernelSmith is a **kernel backend**, not a full model deployment framework.
MLIR is internal build-time machinery; users consume a static C library with
stable headers and caller-provided workspace. Framework integrations such as
llama.cpp/GGML, ExecuTorch, TFLite Micro, or IREE are downstream consumers of
the C kernels.

The first end-to-end proof is a narrow llama.cpp integration that routes a
reviewed f32 operation through the public KernelSmith C API. Quantized W4A8/RVV
integration follows after that seam is proven, and configurable QEMU
system-mode/gem5 simulation follows the stable runtime workload.

---

## Milestone 0: Dialect Infrastructure (Complete)

**Status**: Done

- KS dialect, initially 13 operations in TableGen. Later milestones added
  elementwise, quantization, and transformer ops. The dialect now has 20
  operations; the current table is in [README.md](README.md#supported-kernels)
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

**Status**: Partial. `--ks-lower-to-linalg`, `--ks-tile`, and
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

**Status**: Done for the decode-oriented dot/GEMV foundation. Quantized GEMM
remains future work.

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
- Fused lowering for INT8 and W4A8 dot/GEMV
- Block or per-channel scale layout documented in the public ABI
- NumPy references for i8 and W4A8 dot/GEMV
- Generated INT8 RVV objects integrated behind the C API

**Tasks**:
1. ✓ Add `ks.quantize` and `ks.dequantize` ops (TableGen + verifier + lit tests)
2. ✓ Add accumulator/quantization metadata needed for i8 and W4A8 lowering
3. ✓ Define C APIs and packed layouts for i8 and W4A8 dot/GEMV
4. ✓ Implement INT8 lowering: i8 -> i32 accumulation
5. ✓ Implement W4A8 fused unpack/dequantize and f32 accumulation lowering
6. ✓ Add RVV i8/W4A8 tile parameters and pack factors to target profiles
7. ✓ Add NumPy validation for quantized dot/GEMV
8. ✓ Add lit tests for quantized lowering and diagnostics

---

## Milestone 6: Transformer Minimum Kernel Set

**Status**: Done.

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
1. ✓ Add `ks.add` and `ks.mul` ops with broadcasting rules, verifiers, and lit tests
2. ✓ Add C APIs for `ks_add_f32`, `ks_mul_f32`, `ks_rms_norm_f32`, and `ks_softmax_f32`
3. ✓ Lower add/mul/silu to vectorizable linalg or loop forms
4. ✓ Lower RMSNorm with f32 accumulation and explicit epsilon behavior
5. ✓ Lower softmax using max-subtract-exp-sum-divide for numerical stability
6. ✓ Add functional validator coverage for transformer helper kernels
7. ✓ Add QEMU tests for generated RVV helper kernels

---

## Milestone 7: Known-Runtime Transformer Integration Smoke

**Goal**: Run a pinned transformer model in llama.cpp while routing one
observable f32 operation through the existing KernelSmith public C API.

**Why this first**: This is the fastest route that proves an established runtime
can consume KernelSmith. It separates runtime integration risk from unresolved
GGML-to-KernelSmith quantization conversion and RVV optimization.

**Design**:
[DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md)

**Deliverables**:
- Pinned llama.cpp revision and licensed, checksummed CI-sized model
- Reproducible unmodified host baseline with deterministic greedy output
- Build-time opt-in for one reviewed KernelSmith f32 operation
- Non-zero KernelSmith invocation report and operation-level comparison
- Deterministic token parity between baseline and KS-enabled configurations
- Recommendation for a permanent backend or narrower CPU adapter

**Tasks**:
1. `TASK-025`: adopt and review the known-runtime-first strategy
2. `TASK-026`: pin the runtime/model baseline and identify candidate seams
3. `TASK-027`: route one f32 operation through KernelSmith on the host

**Claim boundary**: This milestone proves runtime consumption only. It does not
claim quantized compatibility or RVV acceleration.

---

## Milestone 8: Quantized RVV Runtime Integration

**Goal**: Route a llama.cpp quantized decode dot/GEMV path through the native
KernelSmith W4A8 ABI and execute the RVV implementation under QEMU user-mode.

**Why after the host smoke**: Runtime integration must be proven before adding
lossy format conversion, target-specific objects, and RISC-V cross-build
failures.

**Design**:
[DES-015](docs/design/DES-015-quantized-layout-abi-and-integration-conversion-paths.md),
[DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md)

**Deliverables**:
- Ratified and versioned native quantized layout
- Tested GGML `Q4_0` decode and requantization into native W4A8
- W4A8 RVV object selection behind `ks_dot_w4a8` / `ks_matvec_w4a8`
- Quantized llama.cpp decode adapter with invocation and conversion metadata
- Host numerical/token comparison against the pinned baseline
- RISC-V QEMU user-mode validation at VLEN 256 and 512
- Machine-readable correctness, error, and benchmark reports

**Tasks**:
1. `TASK-028`: ratify and version the native quantized layout
2. `TASK-029`: convert GGML `Q4_0` weights to native W4A8
3. `TASK-030`: integrate W4A8 RVV objects behind the public C API
4. `TASK-031`: route llama.cpp quantized decode through KernelSmith
5. `TASK-032`: validate the quantized runtime under QEMU user-mode

**Claim boundary**: Conversion is lossy and is not native GGML format
compatibility. QEMU timing is simulation data, not hardware performance.

---

## Milestone 9: Configurable RISC-V System Simulation

**Goal**: Run the stable runtime workload in a configurable single-node RVV
system, first with QEMU system-mode and then with gem5 full-system.

**Why after runtime correctness**: Boot, image, device, and microarchitectural
modeling failures should not be mixed with runtime or quantization integration
failures.

**Design**:
[DES-012](docs/design/DES-012-riscv-simulation-verification.md),
[DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md)

**Deliverables**:
- Simulator-neutral system configuration for hart count, ISA, VLEN/ELEN,
  memory, caches, and devices
- One configurable RV64GCV QEMU system node running the M8 workload
- Capability diagnostics for simulator-specific unsupported fields
- gem5 full-system adapter running the same image and workload
- Functional cross-simulator comparison and stable gem5 statistics report

**Tasks**:
1. `TASK-033`: define the configuration and bring up one QEMU RVV system node
2. `TASK-034`: map the workload and configuration to gem5 full-system

Additional harts, accelerators, and heterogeneous nodes follow after the
single-node path is stable.

---

## Milestone 10: RISC-V Operator Coverage and Hardening

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

The full index, with status, is [docs/design/README.md](docs/design/README.md).

| ID | Title | Scope |
|----|-------|-------|
| [DES-006](docs/design/DES-006-kernel-library-architecture.md) | Kernel Library Architecture | C API, memory mgmt, tiling, packing, target profiles |
| [DES-011](docs/design/DES-011-riscv-first-transformer-demo.md) | RISC-V First Transformer Demo Strategy | Original custom-runner-first strategy; sequencing superseded by DES-016 |
| [DES-016](docs/design/DES-016-known-runtime-first-transformer-integration.md) | Known-Runtime-First Transformer Integration | llama.cpp host smoke, quantized RVV integration, then system simulation |
| [DES-015](docs/design/DES-015-quantized-layout-abi-and-integration-conversion-paths.md) | Quantized Layout ABI and Integration Conversion Paths | Proposed native layout versioning; requantized GGML import; future IRON investigation |
| [DES-012](docs/design/DES-012-riscv-simulation-verification.md) | RISC-V Simulation Verification | QEMU-first generated-kernel verification with Spike/gem5/Renode follow-ons |
| [DES-002](docs/design/DES-002-matmul-kernel.md) | MatMul Kernel (TDD) | MLIR pipeline design (partially superseded by DES-006) |
| [DES-001](docs/design/DES-001-vector-operations-lowering.md) | Vector → RVV Lowering | RVV-specific intrinsic mapping |
| ~~DES-005~~ | ~~Library Packaging (v1)~~ | Deleted — was superseded by DES-006 |

## Target Profile Specification

See [specs/targets/system-description.md](specs/targets/system-description.md) for the
target profile format, validation rules, and provided profiles.
