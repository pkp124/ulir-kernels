# DES-013: Golden Reference Verification Infrastructure

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Implemented |
| **Author** | KernelSmith Team |
| **Created** | 2026-06-07 |
| **Related** | DES-007, DES-009, DES-012, TASK-006, TASK-010, TASK-011, TASK-012 |

> **Current status (2026-09-28):** TASK-010, TASK-011, and TASK-012 are
> complete, including INT8 and W4A8 smoke cases. The open questions later in
> this file are still open. How to add a case is in the testing guide.

## Context

### Problem Statement

KernelSmith now has a QEMU user-mode smoke path for generated RVV objects, but
the first case validates against constants embedded in the MLIR test program.
That proves the simulator wiring works; it does not scale to matmul, packed
layouts, dynamic shapes, quantized arithmetic, or fixed-point rounding behavior.

The project needs a target-independent golden-reference verification system:

1. Generate deterministic inputs and expected outputs from NumPy or a widely
   trusted AI framework.
2. Verify host/x86 KernelSmith output against that golden reference.
3. Verify RISC-V RVV simulator output against the same golden reference, and
   optionally against the host output.
4. Record dtype, shape, tolerance, quantization parameters, rounding mode, and
   saturation policy so fixed-point failures are diagnosable.

### Background

Existing KernelSmith pieces:

- `tests/test_data_generator.py` creates simple binary inputs and references.
- `tests/functional_validator.py` compares host outputs with NumPy-derived
  references.
- `tests/riscv_runner.py` builds descriptor-backed RVV ELFs and runs them under
  QEMU user-mode.
- `TASK-006` requires QEMU correctness against C or NumPy references.
- `DES-012` defines QEMU-first simulation tiers, but not the golden data model.

Relevant open-source patterns:

- IREE separates compiler lit tests from runtime/e2e checks. Its e2e and
  external suites compile programs, run them on selected backends/devices, and
  compare outputs using explicit expected-output files or assertions.
- ONNX provides `onnx.reference.ReferenceEvaluator`, a pure Python reference
  runtime for operator/model outputs.
- TVM and NumPy-centric test flows commonly generate framework outputs and use
  `numpy.testing.assert_allclose` or project wrappers with explicit tolerances.

KernelSmith should follow the same split: lit tests validate IR shape and
diagnostics, while functional tests validate executable outputs against stable
golden data.

## Requirements

From specifications and roadmap:
`specs/targets/riscv-rvv.md`, `ROADMAP.md` M4/M5, `TASK-006`.

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Generate deterministic golden inputs and expected outputs with NumPy first, and optionally Torch/ONNX for complex ops. | Must Have |
| REQ-2 | Store golden metadata: shape, dtype, seed, op attributes, tolerance, quantization parameters, rounding mode, saturation policy, and content hashes. | Must Have |
| REQ-3 | Verify host/x86 KernelSmith outputs against golden references before target-specific simulation. | Must Have |
| REQ-4 | Verify RISC-V RVV QEMU outputs against the same golden references and supported VLEN profiles. | Must Have |
| REQ-5 | Support fixed-point comparisons with exact integer checks where required and explicit tolerances where rounding policy permits. | Must Have |
| REQ-6 | Emit machine-parseable reports for pass/fail, max error, mismatch count, dtype/shape, and failing indices. | Must Have |
| REQ-7 | Keep smoke tests small enough for PR CI and allow broader randomized/nightly matrices later. | Should Have |
| REQ-8 | Keep the case format target-neutral so ARM NEON and future RISC-V profiles can reuse the same golden data. | Should Have |

## Design

### Overview

Golden-reference verification is organized around a target-neutral case
descriptor and a generated artifact bundle:

```text
case descriptor
  -> golden generator (NumPy/Torch/ONNX)
  -> golden bundle: inputs + expected outputs + metadata
  -> host/x86 execution
  -> compare host output vs golden
  -> RISC-V RVV execution under QEMU
  -> compare RISC-V output vs golden and optionally host output
```

The golden bundle is the contract. Every executable target consumes the same
inputs and is judged against the same expected outputs and comparison policy.

### Component Design

#### Case Descriptor

Descriptors live under `tests/functional/cases/` or `tests/golden/cases/` once
implemented. JSON is sufficient initially and keeps CI dependency-light.

Example:

```json
{
  "name": "matmul_f32_16x32x24",
  "kernel": "matmul",
  "source": "tests/functional/mlir/matmul_f32_16x32x24.mlir",
  "generator": {
    "kind": "numpy",
    "function": "matmul",
    "seed": 42
  },
  "inputs": [
    {"name": "A", "shape": [16, 32], "dtype": "float32"},
    {"name": "B", "shape": [32, 24], "dtype": "float32"}
  ],
  "outputs": [
    {"name": "C", "shape": [16, 24], "dtype": "float32"}
  ],
  "compare": {
    "mode": "allclose",
    "rtol": 0.00001,
    "atol": 0.00001,
    "strict_shape": true,
    "strict_dtype": true
  },
  "targets": ["host_reference", "riscv_rvv_256"],
  "vlens": [256, 512]
}
```

Quantized cases add explicit arithmetic policy:

```json
{
  "compare": {
    "mode": "quantized_exact",
    "accumulator_dtype": "int32",
    "input_zero_point": 0,
    "weight_zero_point": 0,
    "scale": 0.03125,
    "rounding": "nearest_even",
    "saturation": "int8"
  }
}
```

#### Golden Generator

The generator creates deterministic artifacts:

```text
tests/golden/generated/<case>/
  manifest.json
  input_A.npy
  input_B.npy
  expected_C.npy
```

`manifest.json` records:

- case name and schema version;
- generator backend and version;
- seed and input distribution;
- shape and dtype of each tensor;
- compare policy;
- quantization policy;
- SHA-256 for every generated file.

For in-repo smoke tests, generated arrays may be small and committed. Larger or
randomized suites should be regenerated in CI or run nightly.

#### Host Execution Layer

Host verification runs first. It should support:

1. Current scalar C library/reference kernels.
2. Future generated x86/LLVM objects.
3. Output capture to `.npy` or raw binary plus metadata.

Host failures mean the case, ABI, or golden data is broken and RISC-V should
not run for that case in CI.

#### RISC-V Execution Layer

RISC-V verification reuses the existing `scripts/compile-rvv.sh` and
`tests/riscv_runner.py` concepts but changes the harness contract:

- load or embed the same generated input data;
- call the generated RVV kernel;
- write output bytes or compute comparison metrics;
- print stable result markers;
- run under each valid VLEN for the target profile.

The preferred long-term model is to export actual output files and compare them
on the host after QEMU exits. A smaller embedded-data mode is acceptable for PR
smoke tests if it is generated from the golden manifest, not handwritten in
MLIR.

#### Comparator

The comparator is a Python tool shared by host and RISC-V flows:

- `exact`: byte-for-byte or integer equality.
- `allclose`: NumPy-style `rtol`/`atol` with strict shape/dtype checks.
- `quantized_exact`: exact integer comparison with mismatch reporting.
- `dequantized_allclose`: dequantize actual and expected using manifest policy,
  then compare in float for diagnostics.

Reports include:

- `PASS` / `FAIL`;
- max absolute error and max relative error;
- mismatch count;
- first N failing indices with actual/expected values;
- file hashes for actual and expected output.

### Interface

Initial command shape:

```bash
python tests/golden/generate.py --case tests/golden/cases/matmul_f32.json
python tests/verify.py --case tests/golden/cases/matmul_f32.json --target host_reference
python tests/verify.py --case tests/golden/cases/matmul_f32.json --target riscv_rvv_256 --vlens 256 512
```

CI may wrap these through:

```bash
./scripts/run-tests.sh --golden
./scripts/run-tests.sh --riscv-functional
```

The existing `tests/riscv_runner.py` can either call into `tests/verify.py` or
be replaced by it once host and target verification share one descriptor schema.

### Data Flow

```
Descriptor
  -> Golden generator
  -> inputs.npy + expected.npy + manifest.json
  -> Host executable -> host_actual.npy
  -> Comparator(host_actual, expected)
  -> RVV executable under QEMU -> rvv_actual.npy or metrics
  -> Comparator(rvv_actual, expected)
  -> Optional Comparator(rvv_actual, host_actual)
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Embedded constants in MLIR | Simple, no runtime file IO | Does not scale; easy to handwrite wrong expectations; poor for quantized cases | Keep only for tiny smoke tests |
| C harness computes scalar reference on target | Independent of host artifacts | Duplicates reference logic in C; can hide target ABI errors; slow for larger cases | Use selectively for sanity checks |
| NumPy/Torch-generated golden bundles | Reproducible, target-neutral, good diagnostics | Requires schema and artifact management | **Selected** |
| Compare only RISC-V vs x86 output | Catches target divergence | If x86 is wrong, both can agree incorrectly | Use as secondary check |
| Vendor external model suites early | Broad coverage | Too much infrastructure before kernel ABI stabilizes | Defer to nightly/future |

### Rationale for Chosen Approach

Golden bundles establish one source of truth across host, RISC-V, and future
targets. They also force the project to specify fixed-point arithmetic policy
before validating quantized kernels, which prevents ambiguous "close enough"
behavior from creeping into INT8/W4A8 work.

## Test Strategy

### Unit Tests
- [ ] Descriptor schema validation.
- [ ] Golden manifest hash validation.
- [ ] Comparator exact/allclose/quantized modes.
- [ ] Failure reports include first mismatching indices.

### Lit Tests
- Existing lit tests remain responsible for parser, verifier, and lowering IR
  shape. Golden-reference tests do not replace lit tests.

### Edge Cases
- [ ] Non-divisible dimensions and tails.
- [ ] Strided inputs/outputs where ABI supports them.
- [ ] NaN/Inf policy for floating point.
- [ ] Quantized saturation boundaries.
- [ ] Rounding tie cases.
- [ ] Zero scale or invalid quantization metadata diagnostics.

### Integration Tests
- [ ] Host matmul f32 output vs NumPy golden.
- [ ] RISC-V matmul f32 output vs the same NumPy golden at VLEN 256 and 512.
- [ ] RISC-V ReLU/activation output vs generated golden constants.
- [ ] INT8 dot/GEMV exact accumulator test once quantization metadata lands.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Golden generator bug becomes source of truth | High | Medium | Keep simple NumPy code reviewed with known small hand-checked cases |
| Artifact churn bloats repository | Medium | Medium | Commit only small smoke bundles; regenerate larger cases in CI/nightly |
| QEMU file IO complicates static harnesses | Medium | Medium | Start with embedded generated arrays, then add output-file capture |
| Fixed-point tolerances become vague | High | Medium | Require explicit rounding/saturation policy in descriptor |
| Host and target harness ABIs diverge | High | Medium | Generate harness bindings from the same descriptor where possible |

## Open Questions

- [ ] Should small golden `.npy` files be committed, or should CI always
      regenerate them and validate hashes?
- [ ] Should RISC-V harnesses write `.npy`, raw binary, or only metrics in PR CI?
- [ ] Which framework should be the first non-NumPy reference for attention:
      PyTorch, ONNX ReferenceEvaluator, or both?
- [ ] Should descriptor schema validation use JSON Schema or a Python dataclass
      validator initially?

## Dependencies

- `TASK-006`: existing RVV validation milestone.
- `TASK-010`: golden generator and descriptor schema.
- `TASK-011`: host/x86 execution comparator.
- `TASK-012`: RISC-V output comparison against golden references.
- `tests/functional_validator.py`: existing comparison utilities.
- `tests/riscv_runner.py`: existing QEMU execution path.
- `scripts/compile-rvv.sh`: generated RVV object pipeline.

## Implementation Plan

### Phase 1: Golden Data Contract (`TASK-010`)
- [x] Define descriptor schema and manifest format.
- [x] Add deterministic NumPy generators for matmul and ReLU.
- [x] Add comparator library with exact, allclose, and quantized modes.
- [x] Add unit tests for schema, manifest, hashes, and comparator failures.

### Phase 2: Host Reference Execution (`TASK-011`)
- [x] Add host harness path that consumes descriptor inputs.
- [x] Dump host outputs in the shared artifact format.
- [x] Compare host outputs against golden references in CTest/CI.

### Phase 3: RISC-V Golden Verification (`TASK-012`)
- [x] Update RISC-V runner/harness contract to consume descriptor inputs.
- [x] Run RVV outputs under QEMU at VLEN 256 and 512.
- [x] Compare RVV outputs against golden references.
- [x] Compare RVV outputs against host outputs where useful.

### Phase 4: Quantized Expansion
- [x] Add quantized descriptor fields for scale, zero point, rounding, and
      saturation.
- [x] Add INT8/W4A8 NumPy reference cases for dot/GEMV, plus an INT8 matmul
      smoke case.
- [x] M5 quantized lowering is gated on those golden comparisons.

---

## Review History

### Review 1 (pending)
**Reviewer**: TBD
**Decision**: Pending

**Feedback**:
- Pending review.

**Resolution**:
- Pending review.
