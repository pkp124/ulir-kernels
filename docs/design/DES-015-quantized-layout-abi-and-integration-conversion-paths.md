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
in memory. Today this layout exists across `specs/kernels/quantization.md`,
`DES-014`, the public C API, and target profiles, but it has no explicit layout
version.

This document proposes a versioned native contract and a conversion policy for
GGML `Q4_0`. It does not ratify `KS_QUANT_LAYOUT_VERSION = 1` or change the
public ABI; those require design approval followed by an implementation task.
IRON/MLIR-AIE notes are non-normative because no compatible quantized compute
path has been validated.

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
- MLIR-AIE/IRON uses target-specific tile layouts streamed through ObjectFIFOs.
  No KernelSmith quantized layout has been validated against that execution
  model, so compatibility remains an open research question.

## Requirements

From specification: `specs/kernels/quantization.md`.

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Propose one canonical, versioned KernelSmith-native quantized layout for W4A8 and INT8. | Must Have |
| REQ-2 | Express the layout as a *logical math contract* plus a native physical view, with explicit conversion policies for foreign formats. | Must Have |
| REQ-3 | Specify the offline conversion math from GGML `Q4_0`/`Q8_0` to the native layout. | Must Have |
| REQ-4 | Document where the native layout and GGML semantics diverge and how each divergence is resolved. | Must Have |
| REQ-5 | Record the unresolved constraints for a future IRON/MLIR-AIE integration without claiming compatibility. | Should Have |
| REQ-6 | Propose answers to the `DES-011` open format questions (native W4A8 form, scale granularity, model file source). | Should Have |
| REQ-7 | Keep integrations as downstream conversions; no runtime format dispatch inside kernels. | Must Have |
| REQ-8 | Change no public C ABI in this design-only PR. | Must Have |

## Design

### Overview

Split the quantized ABI into two layers:

1. **Logical contract** — the dequantized math each kernel computes, expressed in
   unpacked logical indices. This is already written in
   `specs/kernels/quantization.md` and does not change.
2. **Physical layout** — the concrete byte layout consumed by KernelSmith
   kernels. Foreign formats are decoded and converted into this native layout.
   Conversion may require requantization when the foreign arithmetic contract
   cannot be represented exactly.

Kernels only ever see the native layout. Importers run at model-load or build
time, never on the hot path. `Q4_K` and other affine formats remain unsupported
until a separate design defines their conversion policy.

### Component Design

#### Native W4A8 layout (proposed versioned contract)

The proposed v1 contract assigns a version to the layout already implemented
and specified in `specs/kernels/quantization.md` §"W4A8 Packed Weight Layout".
The version identifier is not added by this design-only change:

- **Weights**: signed INT4, two's-complement, two per byte, *adjacent* packing.
  `byte[k/2]` low nibble = `weight[k]` (even `k`), high nibble = `weight[k+1]`
  (odd `k`). Rows independent; a packed row is `ceil(cols/2)` bytes with an
  explicit `packed_stride_bytes` between rows.
- **Weight scales**: positive finite f32 values, row-major,
  `ceil(cols/group_size)` per row; scale index for element `k` is
  `k / group_size`. The RVV profile recommends `group_size = 64`.
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
| Nibble order | Adjacent: `byte[k/2]` holds `k`, `k+1` | Interleaved: `qs[j]` holds `j` (low) and `j+16` (high) within a 32-block | Scatter by logical index, then pack adjacent native indices `(2p, 2p+1)` |
| INT4 encoding | Signed two's-complement nibble | Unsigned nibble with implicit `-8` bias | Decode to logical f32 values, then requantize to native signed INT4 |
| Weight group size | Runtime `group_size`; RVV profile recommends 64 | Fixed 32 per block | Use native group size 32 for importer-local error bounds or regroup to the target profile |
| Weight scale | Positive finite f32 per group | Signed f16 `d = max/-8` per block | Direct widening is invalid when `d <= 0`; decode and requantize with a positive native scale |
| Weight symmetry | Symmetric (zp = 0) | `Q4_0` uses signed scale plus biased nibbles; `Q4_1`/`Q4_K` add min/offset | `Q4_0` requires requantization; `Q4_1`/`Q4_K` remain out of scope |
| Activation scale | One per-tensor f32 | Per-block (32) f16 via on-the-fly `Q8_0` | **Semantic gap — see below** |
| Accumulation | Single f32 accumulate | Per-block accumulate, scaled per block | Follows from activation-scale granularity |

There are two arithmetic divergences, not just byte-order differences:

1. GGML `Q4_0` permits a signed block scale. KernelSmith's C API rejects
   non-positive W4A8 scales, so direct nibble repacking plus f16-to-f32 scale
   widening is not a valid general conversion.
2. GGML computes `sum_blocks(d_w[b] * d_a[b] * sum_i(q_w * q_a))` with a
   per-32-block activation scale, whereas KernelSmith applies one activation
   scale for the whole vector.

The proposed M8 path is therefore an **offline requantized weight import**:
decode `Q4_0` blocks to f32, requantize them to KernelSmith signed INT4 with
positive scales, and quantize activations using KernelSmith's per-tensor scheme.
This is neither byte-exact nor numerically identical to llama.cpp.

- **Block-faithful path (future)**: extend the native contract with an optional
  activation group size and a representation for GGML's signed block-scale
  convention. This requires a separate ABI design.

#### GGML → native conversion (offline)

For each `Q4_0` block with `QK4_0 = 32`, first decode GGML's logical f32
weights, then requantize into a positive-scale native group:

```text
ggml_d := fp16_to_fp32(block[b].d)
for j in [0, 16):
  decoded[32*b + j]      := ((qs[j] & 0x0F) - 8) * ggml_d
  decoded[32*b + j + 16] := ((qs[j] >> 4)   - 8) * ggml_d

for each native group g:
  lo := min(decoded[g])
  hi := max(decoded[g])
  native_scale[g] := max(hi / 7, -lo / 8)
  if native_scale[g] == 0:
    native_scale[g] := 1
  for each value v in decoded[g]:
    q := clamp(round(v / native_scale[g]), -8, 7)
    native_pack(q)
```

`native_pack` writes to the adjacent-nibble native layout: value at logical `k`
goes to `byte[k/2]` low nibble if `k` is even and the high nibble otherwise.
The zero-block scale is set to `1` because the current C API requires positive
scales. Importer tests must measure dequantized error; byte-level round-trip is
not an acceptance criterion.

GGML `Q8_0` activations likewise cannot be copied directly into the native
per-tensor activation contract. A compatible importer decodes the Q8 blocks to
f32 and requantizes the complete activation tensor using one positive
`input_scale` and one `input_zero_point`.

#### Future IRON/MLIR-AIE exploration (non-normative)

No INT4/INT8 IRON kernel or ObjectFIFO layout has been validated against the
KernelSmith W4A8 ABI. The existing `--ks-pack` layout is for matrix-multiplication
panels and must not be assumed compatible with row-major packed GEMV weights.
A future design must select a target device and compute dtype, define the
ObjectFIFO tile layout, and measure any conversion cost before claiming a
re-tiling-only integration.

### Interface

This PR makes no public C ABI changes. If the proposal is approved, a follow-up
task will add and test a build-time version macro:

```c
/* target profile / ks_common.h */
#define KS_QUANT_LAYOUT_VERSION 1   /* proposed native W4A8/INT8 layout */
```

The macro does not exist yet. Conversions remain outside the kernel library,
for example a future `scripts/convert_ggml.py`, consistent with `DES-011`'s
two-stage strategy.

### Data Flow

```
Logical math contract (specs/kernels/quantization.md)
  |
  |  serialize
  v
Proposed native physical layout v1  <-- kernels consume this only
  ^
  | offline decode + requantize
  |
GGML Q4_0 weights
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Adopt GGML `Q4_0`/`Q4_K` as the native layout | Byte-compatible with llama.cpp; free model files | Interleaved nibbles + f16 block scales + per-block activation scales complicate RVV lowering and verification; k-quants are complex; ties native format to upstream churn | Rejected |
| Keep native layout implicit, convert ad hoc per integration | No upfront design | Each integration re-derives packing; silent divergence risk; the exact failure `DES-011` warns about | Rejected |
| **Native logical contract + one native layout + documented foreign conversions** | Simple, verifiable native kernels; scope unchanged | GGML import requires lossy requantization under the current positive-scale ABI | **Proposed** |

### Rationale for Chosen Approach

The native layout prioritizes kernel simplicity and testability: adjacent
nibbles and positive f32 group scales lower cleanly to RVV. Importers absorb
foreign-format complexity offline. This keeps format-specific dispatch out of
the kernel path but does not imply lossless interoperability.

## Test Strategy

### Unit Tests
- [ ] Round-trip: native pack → unpack recovers logical INT4/INT8 values.
- [ ] GGML `Q4_0` block → decode → native requantize produces bounded
      dequantized error against `dequantize_row_q4_0`.
- [ ] Cover positive, negative, and zero GGML block scales.
- [ ] Verify native scales are positive and finite after conversion.

### Lit Tests
- [ ] Existing `ks.dot_w4a8` / `ks.matvec_w4a8` parse/verify/lower tests stay green
      (no dialect change expected).

### Edge Cases
- [ ] `cols` not a multiple of 2 (odd tail nibble) and not a multiple of
      `group_size` (partial final scale group).
- [ ] GGML block boundary (32) vs native group boundary (64) regrouping.

### Integration Tests
- [ ] Requantized GGML weight import produces coherent greedy tokens in the M8
      integration (correctness-of-integration, not bit-exact llama.cpp parity).

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Signed GGML block scales copied into the positive-scale native ABI | High | High | Decode and requantize; test positive, negative, and zero blocks |
| Activation-scale granularity gap misread as bit-format bug | Medium | Medium | Document requantized vs block-faithful paths explicitly |
| k-quant (`Q4_K`) affine form unsupported by symmetric native view | Medium | Medium | Restrict v1 GGML interop to `Q4_0`/`Q8_0`; defer affine `w=a*q+b` |
| Native group size 64 vs GGML 32 compounds requantization error | Medium | Medium | Measure both group sizes in converter tests |
| IRON compatibility is inferred without a validated kernel/layout | Medium | High | Require a separate target-specific design and prototype |
| Layout changes before versioning is implemented | High | Medium | Add the macro and ABI tests in a follow-up before declaring v1 ratified |

## Open Questions

Proposed answers to the `DES-011` open questions:

- [ ] **Native W4A8 form**: symmetric per-group signed INT4 with adjacent
      nibble packing (native), *not* a GGML-compatible block format. GGML support
      is an offline decode-and-requantize importer.
- [ ] **Scale granularity**: per-group (RVV recommendation 64) for weights;
      per-tensor for activations in v1, with optional per-group activations as a
      future extension for block-faithful GGML parity.
- [ ] **First model file**: KernelSmith-specific container (`DES-011` M7); a
      GGML weight importer is proposed for the M8 showcase.
- [ ] Should the v1 target-profile recommendation remain 64 or use 32 for
      lower-error GGML imports?
- [ ] Is requantized import sufficient for M8, or does exact GGML parity justify
      a separate block-faithful ABI?
- [ ] Which IRON compute dtype anchors the first spatial showcase?

## Dependencies

- `specs/kernels/quantization.md`: logical math contract (unchanged).
- `DES-006` §5 and `DES-009`: existing matmul packing, to be evaluated rather
  than assumed reusable for IRON.
- `DES-011`: two-stage integration strategy (minimal runner, then llama.cpp).
- `DES-014`: INT8 native view and lowering.
- External: GGML `ggml-quants.c` at
  `30bf8685ed4eb0a47f2b06229543327749904150`
  (`d = max / -8`, dequantization `(q - 8) * d`); MLIR-AIE/IRON ObjectFIFO
  model.

## Implementation Plan

This document is design-only; implementation is deferred and sequenced behind the
kernel roadmap.

### Phase 1: Review the proposal (this doc)
- [ ] Approve or revise the proposed native layout v1.
- [ ] Keep specification cross-links explicitly marked as draft.

### Phase 2: Implement versioning
- [ ] Create a task for `KS_QUANT_LAYOUT_VERSION` in `ks_common.h`.
- [ ] Add ABI and layout helper tests before declaring v1 ratified.

### Phase 3: GGML converter (M8 showcase)
- [ ] `scripts/convert_ggml.py`: `Q4_0`/`Q8_0` → native view.
- [ ] Converter correctness tests vs GGML dequant reference.

### Phase 4: IRON investigation (separate design)
- [ ] Select an anchor compute dtype and target device.
- [ ] Prototype and validate a quantized ObjectFIFO tile layout.

---

## Review History

### Review 1 (2026-08-08)
**Reviewer**: Cursor technical review
**Decision**: Revision required; remains draft

**Feedback**:
- GGML `Q4_0` scales can be negative, while the KernelSmith C API requires
  positive W4A8 scales. Direct nibble repacking and scale widening is invalid.
- The layout version macro is not implemented and must not be described as
  ratified.
- IRON compatibility has not been validated.

**Resolution**:
- Replaced lossless-repack claims with explicit f32 decode and positive-scale
  requantization.
- Marked versioning as a proposal with a separate implementation phase.
- Moved IRON content to a non-normative future investigation.
