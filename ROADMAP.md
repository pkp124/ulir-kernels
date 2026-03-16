# KernelSmith Roadmap

> **Vision**: Ship a C kernel library (`libkernelsmith.a` + headers) that provides
> optimized ML inference kernels for **edge, embedded, and physical AI** devices.
> Primary target: RISC-V RVV. Secondary target: ARM NEON.
> No runtime dependencies. No MLIR knowledge required by users. Works with any
> toolchain, any RTOS, any task execution engine.
>
> **Domain focus**: Quantized inference on resource-constrained hardware — phones,
> SBCs, MCUs, and emerging RISC-V edge SoCs. Quantization is a core feature,
> not an afterthought.
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

**Goal**: Replace handwritten matmul with MLIR-generated code. Single-level tiling,
no packing, generic target profile. Proves the full ks -> linalg -> LLVM pipeline.

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

## Milestone 4: RISC-V RVV Target (Primary) — In Progress

**Goal**: End-to-end optimized matmul and activations for RISC-V RVV. This is
the project's primary hardware target. Multi-level tiling, B packing, and
RVV-specific vectorization.

**Why RVV first**: Edge RISC-V SoCs (e.g., T-Head C908/C910, SiFive X280,
Kendryte K230) are the primary deployment targets. RVV's vector-length-agnostic
(VLA) model requires a custom lowering pass — LLVM autovectorization is
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
- … QEMU correctness validation (requires cross-compiler + qemu-user in CI)
- … Benchmark: generic vs RVV (demonstrating speedup on QEMU)
- … Stride and tail handling (non-divisible N)

**Tasks**:
1. ✓ Write `target/riscv_rvv_256.h` target profile
2. ✓ Implement `--ks-pack` (linalg.pack for B operand)
3. ✓ Extend `--ks-tile` to multi-level (L2 + register tile via two invocations)
4. ✓ Implement `--ks-vectorize` (linalg → vector.contract + transfer ops)
5. ✓ Implement `--ks-lower-to-rvv` (full bufferize→cf→vector→LLVM pipeline)
6. ✓ Set up cross-compilation: `scripts/compile-rvv.sh` driver
7. ✓ QEMU test runner: multi-VLEN correctness and benchmark support
8. ✓ Lit tests for all three new passes
9. … Run QEMU correctness tests on actual RISC-V binary (needs CI cross-compiler)
10. … Benchmark speedup measurement (QEMU VLEN=256 vs generic)

---

## Milestone 5: Quantization Foundation

**Goal**: INT8 quantized matmul — the single most important operation for edge
inference. This is what makes KernelSmith relevant to the edge AI domain.

**Why now**: Every serious edge framework (TFLite, XNNPACK, QNN, ArmNN) is
quantization-first. Deploying f32 models on edge devices is a non-starter for
production workloads. INT8 symmetric quantization covers the majority of
deployed edge models today.

**Deliverables**:
- `ks.quantize` / `ks.dequantize` ops in the KS dialect
- `ks_matmul_i8` — INT8 input, INT32 accumulator, requantize to INT8 output
- Quantization-aware lowering: i8 multiply -> i32 accumulate -> shift/round -> i8
- RVV-optimized quantized matmul (using widening multiply instructions)
- C API: `ks_matmul_i8(input_i8, weight_i8, scale, zero_point, output_i8, ...)`
- Numerical validation against numpy quantized reference

**Tasks**:
1. Add `ks.quantize` and `ks.dequantize` ops (TableGen + verifier + lit tests)
2. Add quantization attributes to `ks.matmul` (scale, zero_point, per-channel)
3. Implement INT8 lowering in `--ks-lower-to-linalg`:
   - `linalg.quantized_matmul` or manual `arith.extsi` + `linalg.matmul` + requantize
4. INT8-specific tiling (profile parameters for i8 tile sizes)
5. RVV INT8 path: `vwmul.vv` (widening multiply), `vnsra.wi` (narrowing shift)
6. Write C header `ks_matmul.h` i8 variant with scale/zero_point parameters
7. Numpy quantized reference for test validation
8. Lit tests for quantized lowering pipeline

---

## Milestone 6: Edge Operator Coverage

**Goal**: Add the operators that edge inference models actually need beyond
matmul and activations. Prioritized by frequency in deployed edge models
(MobileNet, EfficientNet, tiny transformers).

**Deliverables**:
- New ops with full lowering + RVV optimization + tests
- C API headers for each new kernel

**Operators (in priority order)**:

### 6a: Depthwise Conv2d
The most critical missing op. MobileNet/EfficientNet use depthwise separable
convolutions as their primary building block. More important than regular conv2d
for edge vision models.
- `ks.depthwise_conv2d` op (NHWC input, HW1C filter)
- Lowering: direct convolution (no im2col — memory-constrained)
- RVV vectorization along the channel dimension

### 6b: Element-wise Add / Multiply
Residual connections in every modern architecture. Trivial to implement but
essential for end-to-end model support.
- `ks.add`, `ks.mul` ops (element-wise, broadcasting)
- Fuse with preceding matmul/conv where possible

### 6c: Softmax + Layer Norm / RMS Norm
Already defined as ops (Milestone 0) but need lowering paths.
- Numerically stable softmax (max-subtract-exp-sum-divide)
- Layer norm / RMS norm with f32 accumulation even for i8/f16 inputs

### 6d: Pooling
Average and max pooling for vision model downsampling.
- `ks.avg_pool2d`, `ks.max_pool2d` ops (NHWC)

### 6e: Conv2d (regular)
Already defined as an op. Add lowering path (im2col + matmul or direct).

### 6f: Reduce / Batch MatMul
Already defined as ops. Add lowering paths.

---

## Milestone 7: INT4 and Mixed Precision

**Goal**: INT4 weights x INT8 activations — where on-device LLM inference lives.
Combined with M5 (INT8) and M6 (edge ops), this completes the kernel library
needed for a real-world quantized transformer demo on RISC-V RVV.

**Why before ARM NEON**: The MVP is a complete quantized transformer running
end-to-end on a RISC-V RVV platform. That requires INT4 weights (the standard
for on-device LLMs), not a second architecture. Depth on one target with full
quantization coverage is more compelling than breadth across two targets.

**Deliverables**:
- INT4 weight dequantization (W4A8 scheme)
- `ks_matmul_w4a8` — INT4 weight, INT8 activation, INT32 accumulator
- Block-wise quantization support (group size 32/64/128)
- RVV-optimized: unpack INT4 -> INT8, then widening multiply
- **MVP demo**: quantized LLM inference on RISC-V RVV via QEMU

**Tasks**:
1. INT4 storage format (packed 2 values per byte, little-endian nibble order)
2. Dequantization op: `ks.dequantize_i4` (INT4 -> INT8 with scale per block)
3. W4A8 matmul lowering (dequant + INT8 matmul fused in inner loop)
4. RVV: 4-bit unpack using shift/mask, then `vwmul.vv`
5. Block-wise scale handling (one scale per group of 32-128 weights)
6. Test against reference W4A8 matmul (numpy)
7. MVP demo (see below)

### MVP Demo: Quantized Transformer on RISC-V RVV

**Approach**: Fork [llama2.c](https://github.com/karpathy/llama2.c) (Karpathy's
~700-line C inference engine) and replace its compute calls with KernelSmith's
C API. llama2.c provides tokenizer, model loading, KV cache management, and
sampling — we provide the optimized kernels.

**Why llama2.c**: Pure C, no dependencies, freestanding-compatible, proven with
small models. Matches KernelSmith's "no runtime dependencies" philosophy. The
alternative (writing a custom runtime) duplicates work for no added value at
the MVP stage.

**Demo model**: TinyStories-15M (~8MB quantized W4A8). Small enough for QEMU,
generates coherent English text, proves the full pipeline works.

**Integration points** — replace these llama2.c functions with `ks_*` calls:

```
llama2.c function     →  KernelSmith C API
─────────────────────    ─────────────────────────────
matmul()              →  ks_matmul_w4a8()  (INT4 weights × INT8 activations)
rmsnorm()             →  ks_rms_norm_f32()
softmax()             →  ks_softmax_f32()
silu activation       →  ks_silu_f32()
residual add          →  ks_add_f32()
```

**New artifacts** (in `examples/`):

| File | Lines | Purpose |
|------|-------|---------|
| `examples/runner.c` | ~500 | llama2.c fork with KernelSmith kernel calls |
| `scripts/quantize_model.py` | ~200 | Convert llama2.c .bin weights to W4A8 .ksmodel |
| `scripts/demo_rvv.sh` | ~20 | Cross-compile + QEMU run script |

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

**Validation**: Compare token-by-token output against the original llama2.c
running with f32 weights on the same model. Allow tolerance for quantization
error (expect ~95%+ token match on greedy decoding for a 15M model).

---

## Milestone 8: ARM NEON Target (Secondary)

**Goal**: Same C API, optimized for ARMv8-A NEON. Covers phones, Raspberry Pi,
Jetson, and similar SBCs — the widest deployed edge hardware base. Port the
full kernel library (including INT8/INT4) to a second architecture.

**Why now**: The full kernel library and quantization stack are proven on RVV.
ARM NEON has fixed 128-bit vectors, so LLVM's autovectorizer handles it
reasonably well. No custom lowering pass is needed — target features and
profile-driven tile sizes are sufficient. This makes it a fast port.

**Deliverables**:
- `target/aarch64_neon.h` — ARM NEON target profile
- NEON-optimized matmul (f32, i8, W4A8) via LLVM autovectorization + profile tuning
- NEON-optimized activations, edge ops, and normalization
- INT8: leverage NEON `smull`/`smlal` (widening multiply-accumulate)
- INT4: NEON unpack + widening multiply
- Native or cross-compiled test suite
- Benchmark: generic vs NEON
- Transformer demo replayed on ARM NEON target

**Tasks**:
1. Write `target/aarch64_neon.h` target profile:
   - 128-bit SIMD (4xf32, 8xf16, 16xi8)
   - 32KB L1D, 256KB-1MB L2 (device-dependent)
   - Tile sizes tuned for Cortex-A55/A76 class cores
2. Set LLVM target triple `aarch64-none-linux-gnu` + `+neon,+fp-armv8`
3. Profile-driven tile sizes (no custom pass — LLVM autovectorizes)
4. Cross-compile and test natively or on target device
5. Port all quantized kernels (INT8 matmul, W4A8 matmul) to NEON
6. Benchmark against generic target, compare with XNNPACK/ArmNN on same hardware
7. Optional: `+dotprod` variant (`sdot` instruction for INT8, Cortex-A76+)

---

## Milestone 9: Optimization and Hardening

**Goal**: Production-quality library performance and robustness for edge deployment.

**Tasks**:
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
| ARM SVE | Scalable vectors, similar to RVV. Relevant for server ARM (Graviton). |
| x86 AVX2/AVX-512 | Server/desktop. Existing `target/x86_avx2.h` profile can be used. |
| ARM Cortex-M | MCU/RTOS. Scalar only, validates freestanding story for TinyML. |
| NPU/DSP offload | Vendor-specific. Would require new dialect extensions. |

---

## Design Documents

| ID | Title | Scope |
|----|-------|-------|
| [DES-006](docs/design/DES-006-kernel-library-architecture.md) | Kernel Library Architecture | C API, memory mgmt, tiling, packing, target profiles |
| [DES-002](docs/design/DES-002-matmul-kernel.md) | MatMul Kernel (TDD) | MLIR pipeline design (partially superseded by DES-006) |
| [DES-001](docs/design/DES-001-vector-operations-lowering.md) | Vector → RVV Lowering | RVV-specific intrinsic mapping |
| ~~DES-005~~ | ~~Library Packaging (v1)~~ | Deleted — was superseded by DES-006 |

## Target Profile Specification

See [specs/targets/system-description.md](specs/targets/system-description.md) for the
target profile format, validation rules, and provided profiles.
