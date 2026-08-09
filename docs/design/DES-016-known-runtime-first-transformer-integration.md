# DES-016: Known-Runtime-First Transformer Integration

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Under Review |
| **Author** | KernelSmith Team |
| **Created** | 2026-08-09 |
| **Approved** | |
| **Approver** | |
| **Related** | DES-006, DES-011, DES-012, DES-015, ROADMAP.md M7-M9 |

## Context

### Problem Statement

The post-M6 roadmap selected a custom llama2.c-style runner before integrating
with a recognized runtime. That is the shortest route to any end-to-end
transformer executable, but it does not directly prove that the KernelSmith
library can be consumed by an established runtime.

The desired proof has changed: first demonstrate a real runtime executing a
transformer while routing observable work through KernelSmith, then add the
native quantized RVV path, and only afterward introduce configurable full-system
simulation. The roadmap must distinguish those claims so that a fast integration
smoke is not mislabeled as quantized acceleration.

### Background

- `DES-011` chose a custom runner first and llama.cpp/GGML second. This document
  supersedes that sequencing while retaining its kernel-library boundaries.
- M5 and M6 provide the public decode-oriented C APIs, transformer helpers,
  generated RVV infrastructure, golden references, and QEMU user-mode tests.
- The current generated-object integration is stronger for INT8 than W4A8.
  W4A8 RVV object replacement remains an implementation gap.
- `DES-015` proposes, but does not ratify, a native W4A8 layout version.
  GGML `Q4_0` requires decode and requantization rather than direct repacking.
- `DES-012` already places QEMU system-mode and gem5 after user-mode
  correctness. This document preserves that tiering.

## Requirements

From `include/kernelsmith/`, `specs/kernels/quantization.md`, and
`specs/targets/riscv-rvv.md`:

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Use a recognized transformer runtime for the first integration demo. | Must Have |
| REQ-2 | Prove with instrumentation that at least one runtime operation executes through the KernelSmith public C API. | Must Have |
| REQ-3 | Pin the runtime revision, model artifact, license, and checksum for reproducibility. | Must Have |
| REQ-4 | Establish host output/token parity before RISC-V cross-compilation. | Must Have |
| REQ-5 | Keep the initial smoke independent of unresolved GGML-to-KernelSmith quantization conversion. | Must Have |
| REQ-6 | Do not claim quantized or RVV acceleration until the corresponding path is executed and measured. | Must Have |
| REQ-7 | Add quantized decode through a tested conversion contract and W4A8 RVV implementation. | Must Have |
| REQ-8 | Run the stable runtime workload in QEMU user-mode before system-mode QEMU or gem5. | Must Have |
| REQ-9 | Keep runtime-specific code outside the stable KernelSmith C ABI. | Should Have |
| REQ-10 | Prefer an upstreamable backend or adapter after the feasibility seam is proven. | Should Have |

## Design

### Overview

The integration proceeds through three independently verifiable milestones:

1. **Known-runtime smoke.** Pin llama.cpp and a CI-sized model, establish an
   unmodified baseline, and route one f32 operation through KernelSmith. Exact
   token parity and an invocation counter prove integration correctness.
2. **Quantized RVV decode.** Ratify the native layout, test GGML conversion,
   integrate W4A8 RVV objects, and route a decode dot/GEMV path through
   KernelSmith under QEMU user-mode.
3. **Configurable system simulation.** Define a simulator-neutral system
   configuration, bring up one RVV node in QEMU system-mode, then map the same
   workload and configuration to gem5 full-system.

### Component Design

#### Feasibility gate

The first implementation task is deliberately a spike. It must:

- pin a llama.cpp commit rather than track a moving branch;
- select a small model that llama.cpp already supports and record its license
  and cryptographic checksum;
- build and run an unmodified deterministic greedy baseline;
- map candidate f32 operations to existing KernelSmith APIs;
- select one narrow seam and add a build-time opt-in;
- report a non-zero KernelSmith invocation count;
- preserve baseline tokens and numerically compare the replaced operation.

An internal compile-time hook is acceptable for the spike. It is not accepted
as the permanent integration if it requires broad GGML graph, allocator, or
scheduler changes. A full `ggml_backend_t` implementation is deferred until the
spike establishes that backend ownership is justified.

The spike must stop and return to design review if:

- no existing public KernelSmith API matches a runtime operation without
  changing semantics;
- the model is too large for repeatable host CI;
- deterministic parity cannot be separated from runtime nondeterminism; or
- the hook requires ownership of GGML tensor buffers or the whole graph.

#### Claim levels

Each stage has an explicit claim:

| Stage | Allowed claim |
|-------|---------------|
| Host f32 smoke | llama.cpp executes an operation through `libkernelsmith` |
| Host quantized decode | Tested GGML data is converted and executed through the native KS W4A8 API |
| QEMU user-mode | The same runtime and KS quantized path execute on RISC-V Linux |
| RVV evidence | Generated or target-specific W4A8 code executes RVV instructions |
| System simulation | The stable workload boots and runs in a configurable simulated system |

Passing an f32 smoke does not establish quantized compatibility or RVV
acceleration. QEMU timing is correctness evidence, not hardware performance.

#### Quantization boundary

The quantized milestone is gated by `DES-015` review. Conversion must decode
GGML blocks to a numerical reference and requantize to the native KernelSmith
layout. Tests must measure reconstruction error and runtime output impact.
Direct nibble reuse or claims of bit-exact `Q4_0` compatibility are prohibited.

The first optimized runtime seam is decode-oriented dot/GEMV. Quantized GEMM,
prefill optimization, and additional GGML quantization types are separate work.

#### System simulation boundary

Kernel target profiles remain compile-time kernel tuning inputs. A later system
configuration describes harts, ISA extensions, VLEN/ELEN, memory, caches, and
devices. Simulator adapters report unsupported properties rather than silently
ignoring them.

QEMU user-mode remains the required fast correctness gate. QEMU system-mode
validates boot and OS/platform integration. gem5 validates cache and
microarchitectural behavior and is not a pull-request gate.

### Interface

The spike uses the existing public C ABI. Runtime-specific adapters are opt-in:

```text
unmodified llama.cpp baseline
  + pinned runtime/model manifest
  + KernelSmith integration build flag
  + selected operation adapter
  + invocation/result report
```

No new KernelSmith public API is added solely to mirror an internal llama.cpp
type. Public API changes require their own design review.

### Data Flow

```
Pinned model
  -> llama.cpp model loader / tokenizer / graph
  -> selected operation adapter
  -> KernelSmith public C API
  -> scalar baseline, then quantized RVV implementation
  -> deterministic tokens and integration report
  -> QEMU user-mode
  -> QEMU system-mode
  -> gem5 full-system
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Custom llama2.c-style runner first | Smallest implementation; direct C ABI; QEMU-friendly | Does not prove consumption by a recognized runtime; duplicates loading/tokenizer logic | Deferred as an optional ABI fixture |
| Full llama.cpp backend first | Cleaner long-term runtime abstraction | Requires buffer, scheduler, and graph ownership before the seam is proven | Deferred until after the spike |
| Narrow llama.cpp integration spike | Fast recognized-runtime proof; preserves runtime model loading and tokenizer | Initial hook may not be upstreamable | **Selected** |
| ExecuTorch, TFLite, ONNX Runtime, or IREE first | Recognized deployment frameworks | Delegate/backend surface exceeds the current kernel proof | Deferred |
| QEMU system-mode or gem5 before runtime integration | Establishes platform model early | Adds boot, image, and device failures without proving runtime consumption | Deferred |

### Rationale for Chosen Approach

The selected sequence optimizes for the stated outcome rather than for the
smallest standalone program. It proves runtime consumption early while
preserving strict gates around quantization and RVV claims. The feasibility
spike limits exposure to upstream churn and can fail cheaply before a permanent
backend is designed.

## Test Strategy

### Unit Tests
- [ ] Validate pinned runtime/model manifests and checksums.
- [ ] Compare selected-operation outputs with and without the KS adapter.
- [ ] Test GGML decode and KS requantization against numerical references.
- [ ] Reject incompatible layout versions and group sizes.

### Integration Tests
- [ ] Build and run an unmodified host runtime baseline.
- [ ] Build the KS-enabled host runtime and require a non-zero invocation count.
- [ ] Require deterministic greedy token parity for the f32 smoke.
- [ ] Compare quantized tokens and numerical error against the pinned baseline.
- [ ] Cross-build and run the quantized runtime under QEMU user-mode.
- [ ] Require RVV instruction evidence before labeling the path accelerated.
- [ ] Reuse the stable workload for later QEMU system-mode and gem5 tests.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Upstream llama.cpp churn breaks an internal hook | High | High | Pin a commit; keep the spike narrow; design a backend only after validation |
| A token match hides that KS was never called | High | Medium | Require invocation counters and operation-level result comparison |
| f32 smoke is misreported as quantized RVV acceleration | High | Medium | Enforce the claim-level table in reports and documentation |
| GGML and native W4A8 layouts are treated as byte-compatible | High | Medium | Require decode/requantize tests from DES-015 |
| CI model is too large or license is unsuitable | High | Medium | Make model selection and artifact policy an explicit spike output |
| Full backend work consumes the project before kernels are proven | High | Medium | Use a build-time hook for the spike and define stop criteria |
| System simulation delays runtime correctness | Medium | Medium | Keep QEMU system-mode and gem5 in the later milestone |

## Open Questions

- [ ] Which pinned llama.cpp revision has the narrowest maintainable candidate
      seam?
- [ ] Which licensed, CI-sized model provides deterministic greedy output and
      remains practical under RISC-V QEMU?
- [ ] Which f32 operation is the first spike target: matmul, normalization, or
      another existing public API?
- [ ] Does the validated seam justify a full GGML backend or a smaller CPU
      adapter?
- [ ] What error and perplexity thresholds are acceptable after `Q4_0` to
      native W4A8 requantization?

## Dependencies

- M6 completion: public transformer helper APIs and generated-helper QEMU path.
- `DES-006`: stable C library boundary.
- `DES-012`: simulator tiering.
- `DES-015`: proposed quantized layout and GGML conversion contract.
- An externally pinned llama.cpp source revision and model artifact.

## Implementation Plan

### Phase 1: Known-runtime smoke
- [ ] `TASK-026`: pin and validate the llama.cpp/model baseline.
- [ ] `TASK-027`: route one f32 runtime operation through KernelSmith.

### Phase 2: Quantized RVV runtime path
- [ ] `TASK-028`: ratify and version the native quantized layout.
- [ ] `TASK-029`: implement tested GGML-to-KS W4A8 conversion.
- [ ] `TASK-030`: integrate W4A8 RVV objects behind the public API.
- [ ] `TASK-031`: route llama.cpp quantized decode through KernelSmith.
- [ ] `TASK-032`: validate the runtime under QEMU user-mode.

### Phase 3: Configurable system simulation
- [ ] `TASK-033`: define the system configuration and bring up one QEMU RVV
      system node.
- [ ] `TASK-034`: map the same workload and configuration to gem5 full-system.

---

## Review History

### Review 1 (2026-08-09)
**Reviewer**: Project direction review
**Decision**: Under Review

**Feedback**:
- Prioritize a working transformer in a recognized runtime.
- Take the faster host/user-mode route before system simulation.
- Retain QEMU system-mode and gem5 as later configurable-system work.

**Resolution**:
- Replaced custom-runner-first sequencing with a bounded llama.cpp feasibility
  spike.
- Split integration correctness, quantized RVV acceleration, and system
  simulation into separate milestones and claim levels.
