# DES-015: Quantized Layout ABI and Integration Conversion Paths

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-07-02 |
| **Approved** | |
| **Approver** | |
| **Related** | DES-006, DES-011, DES-014, specs/kernels/quantization.md, ROADMAP.md §M5–M8 |

## Context

### Problem Statement

KernelSmith stays scoped as a kernel backend: it produces quantized RISC-V RVV
kernels and ships them behind a stable C ABI (`DES-006`, `DES-011`). The
roadmap goal is a minimal quantized LLM running on a RISC-V simulation platform
(`ROADMAP.md` M7), followed by *showcase integrations* with external systems —
llama.cpp/GGML (M8), and later tile DSLs / spatial toolchains such as TileLang
and MLIR-AIE/IRON.

The single decision that determines whether those future integrations are cheap
adapters or expensive rewrites is the **quantized data layout ABI**: how packed
INT4/INT8 weights, their scale/zero-point metadata, and activations are laid out
in memory. Today this layout is defined implicitly across
`specs/kernels/quantization.md` and `DES-014`. This document promotes it to an
explicit, versioned contract and specifies the conversion path from each target
integration format to the KernelSmith-native layout — without changing project
scope and without committing to implementing any integration yet.

The goal here is *design closure*, not code: decide the native format, answer the
open format questions in `DES-011`, and write down the conversion math so that a
future integration is a documented repacking step rather than a redesign.

### Background

- `DES-011` left three open questions unanswered: simple symmetric W4A8 vs a
  GGML-compatible block format; per-channel vs per-block scales; and whether the
  first model file is KernelSmith-specific or a llama2.c importer.
- `DES-014` and `specs/kernels/quantization.md` already define the current native
  layout: signed INT4 weights (two per byte, adjacent nibbles), per-group f32
  scales, INT8 activations with a per-tensor scale/zero-point, f32 accumulation.
- GGML/llama.cpp uses **block-wise** formats: `Q4_0` (32-weight blocks, one f16
  scale, symmetric), `Q4_K` (256-weight super-blocks, hierarchical 6-bit
  scales+mins), and quantizes activations on the fly to `Q8_0`/`Q8_1`
  (32-element blocks, per-block f16 scale). Its `vec_dot` accumulates per block,
  multiplying per-block weight and activation scales.
- MLIR-AIE/IRON's stable operator library is bf16/fp16; its distinguishing
  requirement is not a bit format but **tile-panel layouts streamed through
  ObjectFIFOs** into L1 scratchpads. Interop there is about re-tiling and element
  type, not nibble packing.

## Requirements

From specification: `specs/kernels/quantization.md`.

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Define one canonical, versioned KernelSmith-native quantized layout for W4A8 and INT8. | Must Have |
| REQ-2 | Express the layout as a *logical math contract* plus one or more *physical views*, so external formats are documented (de)serializations of the same contract. | Must Have |
| REQ-3 | Specify the offline conversion math from GGML `Q4_0`/`Q8_0` to the native layout. | Must Have |
| REQ-4 | Document where the native layout and GGML semantics diverge and how each divergence is resolved. | Must Have |
| REQ-5 | Specify how native packed weights map to an IRON/MLIR-AIE tile-panel layout for streaming. | Should Have |
| REQ-6 | Answer the `DES-011` open format questions (native W4A8 form, scale granularity, model file source). | Should Have |
| REQ-7 | Keep integrations as downstream conversions; no runtime format dispatch inside kernels. | Must Have |
| REQ-8 | Change no public C ABI or scope; this is a contract and conversion specification. | Must Have |

## Design

### Overview

Split the quantized ABI into two layers:

1. **Logical contract** — the dequantized math each kernel computes, expressed in
   unpacked logical indices. This is already written in
   `specs/kernels/quantization.md` and does not change.
2. **Physical views** — concrete byte layouts that serialize the logical
   contract. KernelSmith defines exactly one *native* physical view (the one its
   generated kernels consume). Every external format (GGML `Q4_0`, `Q8_0`,
   `Q4_K`; an IRON tile panel) is a *foreign view* with a documented, offline
   conversion to/from the native view.

Kernels only ever see the native view. Integrations are repackers that run at
model-load or build time, never on the hot path. This keeps KernelSmith a kernel
backend while making "showcase" integrations thin.

### Component Design

#### Native W4A8 view (canonical)

This ratifies the layout already specified in `specs/kernels/quantization.md`
§"W4A8 Packed Weight Layout" as the versioned native contract
(`KS_QUANT_LAYOUT_VERSION = 1`):

- **Weights**: signed INT4, two's-complement, two per byte, *adjacent* packing.
  `byte[k/2]` low nibble = `weight[k]` (even `k`), high nibble = `weight[k+1]`
  (odd `k`). Rows independent; a packed row is `ceil(cols/2)` bytes with an
  explicit `packed_stride_bytes` between rows.
- **Weight scales**: f32, row-major, `ceil(cols/group_size)` per row; scale index
  for element `k` is `k / group_size`. Default `group_size = 64`.
- **Activations**: INT8, with a single per-tensor `input_scale` (f64 attr / f32
  runtime) and `input_zero_point`.
- **Accumulation**: f32, fused unpack + dequant + dot (no materialized weights).
- **Symmetry**: `weight_zero_point` present for metadata completeness, expected
  `0` (symmetric W4A8).

#### Native INT8 view

Per `DES-014`: row-major INT8 weights `[rows, cols]`, INT8 activations, exact i32
accumulation, `input_zero_point` / `weight_zero_point`, requantization out of
scope for v1. This is the native view for `ks_dot_i8` / `ks_matvec_i8`.

#### Divergence table: native W4A8 vs GGML

This is the load-bearing part of the document — each row is a divergence that a
GGML converter must resolve.

| Aspect | KernelSmith native | GGML `Q4_0` | Resolution in conversion |
|---|---|---|---|
| Nibble order | Adjacent: `byte[k/2]` holds `k`, `k+1` | Interleaved: `qs[j]` holds `j` (low) and `j+16` (high) within a 32-block | De-interleave: gather GGML `(j, j+16)` into native `(2j, 2j+1)` |
| INT4 encoding | Signed two's-complement nibble | Unsigned nibble with implicit `-8` bias | `w_native = ggml_nibble - 8` (numerically identical values, different bits) |
| Weight group size | `group_size` (default 64) | Fixed 32 per block | Set native `group_size = 32` for lossless reuse, or requantize |
| Weight scale dtype | f32 per group | f16 (`d = max/-8`) per block | Widen f16 → f32; scale sign convention already folded into values |
| Weight symmetry | Symmetric (zp = 0) | `Q4_0` symmetric; `Q4_1`/`Q4_K` add min/offset | `Q4_0` maps directly; `Q4_1`/`Q4_K` need affine `w = a*q + b` support (out of v1) |
| Activation scale | One per-tensor f32 | Per-block (32) f16 via on-the-fly `Q8_0` | **Semantic gap — see below** |
| Accumulation | Single f32 accumulate | Per-block accumulate, scaled per block | Follows from activation-scale granularity |

The **activation-scale granularity** is the only *semantic* (not merely
byte-level) divergence. GGML computes `sum_blocks( d_w[b] * d_a[b] *
sum_i(q_w * q_a) )` with a per-32-block activation scale `d_a[b]`, whereas the
native W4A8 contract applies one activation scale for the whole vector. Two ways
to close it, decided per use case:

- **Weight-only reuse (recommended for M8 showcase)**: import GGML `Q4_0`
  *weights* into the native view, but quantize activations with KernelSmith's own
  per-tensor scheme. Numerically this is KernelSmith's kernel, not a bit-exact
  llama.cpp reproduction — acceptable for a showcase and clearly documented.
- **Block-faithful path (future)**: extend the native contract with an optional
  activation group size (mirroring weight groups) so it can match GGML's
  per-block dot exactly. Tracked as an open question, not v1.

#### GGML → native conversion (offline)

For `Q4_0` weights with `QK4_0 = 32`, per block `b` and intra-block index
`j ∈ [0, 16)`:

```text
native_group_size := 32                       # match GGML block to avoid requant
w_lo := (qs[j] & 0x0F) - 8                     # logical weight at k = 32*b + j
w_hi := (qs[j] >> 4)   - 8                     # logical weight at k = 32*b + j + 16
scale[b] := fp16_to_fp32(block[b].d)
# Re-emit into native adjacent packing at logical positions (32*b + j) and (32*b + j + 16):
native_pack(logical_k = 32*b + j,      value = w_lo)
native_pack(logical_k = 32*b + j + 16, value = w_hi)
native_scales[row][ (32*b + *) / native_group_size ] := scale[b]
```

`native_pack` writes to the adjacent-nibble native layout: value at logical `k`
goes to `byte[k/2]` low nibble if `k` even, high nibble if `k` odd. Because GGML
orders logical elements `j` then `j+16`, the converter must scatter by logical
index, not copy bytes. `Q8_0` activations convert analogously (widen f16 scale,
copy int8 `qs`); under weight-only reuse the activation blocks are re-quantized
by KernelSmith instead.

#### Native → IRON/MLIR-AIE tile panel (offline)

IRON interop is layout/dtype, not nibble format. The native packed weight matrix
`[rows, ceil(cols/2)]` plus per-group scales is re-tiled into `NR`-wide column
panels — the same panel concept KernelSmith's `--ks-pack` already produces
(`[N/NR, K, NR]`, `DES-006` §5, `DES-009`). An IRON adapter:

1. Selects an element type IRON supports for the compute core (bf16/fp16 today;
   an INT4/INT8 core kernel if/when available).
2. Repacks native panels into the AIE tile shape consumed by an ObjectFIFO,
   emitting weights + scales as separate FIFO-fed buffers staged through the L2
   mem tile into L1.
3. Leaves orchestration (workers, ObjectFIFO wiring, data movement) entirely to
   IRON.

The takeaway is that KernelSmith's existing panel-packing layout is already
structurally compatible with a streaming spatial consumer; IRON integration is a
re-tiling shim, and the native view needs no change to support it.

### Interface

No public C ABI changes. The contract is documented and versioned; a build-time
macro records it:

```c
/* target profile / ks_common.h */
#define KS_QUANT_LAYOUT_VERSION 1   /* native W4A8/INT8 physical view */
```

Conversions live outside the kernel library, e.g. `scripts/convert_ggml.py` and
a future `integrations/iron/` adapter — both downstream consumers of the stable
layout, consistent with `DES-011`'s two-stage strategy.

### Data Flow

```
Logical math contract (specs/kernels/quantization.md)
  |
  |  serialize
  v
Native physical view (KS_QUANT_LAYOUT_VERSION = 1)  <-- kernels consume this only
  ^                          ^
  | offline convert          | offline re-tile
  |                          |
GGML Q4_0/Q8_0            IRON tile panel (ObjectFIFO-fed)
(weight-only reuse)       (bf16/fp16 or int core)
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Adopt GGML `Q4_0`/`Q4_K` as the native layout | Byte-compatible with llama.cpp; free model files | Interleaved nibbles + f16 block scales + per-block activation scales complicate RVV lowering and verification; k-quants are complex; ties native format to upstream churn | Rejected |
| Keep native layout implicit, convert ad hoc per integration | No upfront design | Each integration re-derives packing; silent divergence risk; the exact failure `DES-011` warns about | Rejected |
| **Native logical contract + one native view + documented foreign conversions** | Simple, verifiable native kernels; integrations are shims; scope unchanged | Not bit-exact with GGML without a converter | **Selected** |

### Rationale for Chosen Approach

The native layout is chosen for kernel simplicity and testability (adjacent
nibbles and f32 group scales lower cleanly to RVV, per `DES-014`), while a small
set of documented offline converters preserves integration ambitions. Framing
external formats as *views of one logical contract* is the mechanism that keeps
each showcase integration a repacking adapter rather than a fork of the kernel
path.

## Test Strategy

### Unit Tests
- [ ] Round-trip: native pack → unpack recovers logical INT4/INT8 values.
- [ ] GGML `Q4_0` block → native view → dequantized weights match GGML
      `dequantize_row_q4_0` within f16→f32 tolerance (converter correctness).
- [ ] `group_size = 32` native path matches GGML block grouping element-for-element.

### Lit Tests
- [ ] Existing `ks.dot_w4a8` / `ks.matvec_w4a8` parse/verify/lower tests stay green
      (no dialect change expected).

### Edge Cases
- [ ] `cols` not a multiple of 2 (odd tail nibble) and not a multiple of
      `group_size` (partial final scale group).
- [ ] GGML block boundary (32) vs native group boundary (64) misalignment.

### Integration Tests
- [ ] Weight-only GGML import produces coherent greedy tokens in the M7 runner
      (correctness-of-integration, not bit-exact llama.cpp parity).

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Activation-scale granularity gap misread as bit-format bug | Medium | Medium | Document weight-only vs block-faithful paths explicitly (this doc) |
| k-quant (`Q4_K`) affine form unsupported by symmetric native view | Medium | Medium | Restrict v1 GGML interop to `Q4_0`/`Q8_0`; defer affine `w=a*q+b` |
| Native `group_size=64` vs GGML 32 forces requantization/accuracy loss | Medium | Medium | Support `group_size=32` for lossless GGML reuse |
| IRON lacks a stable INT4/INT8 core kernel | Medium | High | Scope IRON showcase to supported dtype first; layout shim is dtype-agnostic |
| Layout version drift without a bump | High | Low | Gate any physical-view change on `KS_QUANT_LAYOUT_VERSION` increment |

## Open Questions

Resolving the `DES-011` open questions:

- [x] **Native W4A8 form**: symmetric per-group signed INT4 with adjacent
      nibble packing (native), *not* a GGML-compatible block format. GGML support
      is an offline converter.
- [x] **Scale granularity**: per-group (default 64) for weights; per-tensor for
      activations in v1, with optional per-group activations as a future
      extension for block-faithful GGML parity.
- [x] **First model file**: KernelSmith-specific container (`DES-011` M7); a
      GGML weight importer is added for the M8 showcase.
- [ ] Should v1 default `group_size` be 32 (GGML-friendly) or 64 (fewer scales)?
- [ ] Add optional per-group activation scale to the native contract for exact
      GGML `vec_dot` parity, or accept weight-only reuse for the showcase?
- [ ] Which IRON compute dtype anchors the first spatial showcase?

## Dependencies

- `specs/kernels/quantization.md`: logical math contract (unchanged).
- `DES-006` §5 (packing) and `DES-009`: native panel layout reused for IRON.
- `DES-011`: two-stage integration strategy (minimal runner, then llama.cpp).
- `DES-014`: INT8 native view and lowering.
- External: GGML `Q4_0`/`Q8_0` reference (`ggml-quants.c`); MLIR-AIE/IRON
  ObjectFIFO model.

## Implementation Plan

This document is design-only; implementation is deferred and sequenced behind the
kernel roadmap.

### Phase 1: Ratify contract (this doc)
- [ ] Approve native layout as `KS_QUANT_LAYOUT_VERSION = 1`.
- [ ] Add `KS_QUANT_LAYOUT_VERSION` macro to the profile / `ks_common.h`.
- [ ] Cross-link this doc from `specs/kernels/quantization.md` and `DES-011`.

### Phase 2: GGML converter (M8 showcase)
- [ ] `scripts/convert_ggml.py`: `Q4_0`/`Q8_0` → native view.
- [ ] Converter correctness tests vs GGML dequant reference.

### Phase 3: IRON layout shim (later showcase)
- [ ] `integrations/iron/` adapter: native panels → ObjectFIFO tile layout.
- [ ] Decide anchor compute dtype and target device.

---

## Review History

### Review 1 (pending)
**Reviewer**: TBD
**Decision**: Pending

**Feedback**:
- Pending review.

**Resolution**:
- Pending review.
