# DES-012: RISC-V Simulation Verification

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft |
| **Author** | KernelSmith Team |
| **Created** | 2026-06-06 |
| **Related** | DES-006, DES-007, DES-009, DES-011, TASK-006 |

## Context

### Problem Statement

KernelSmith can lower RVV-targeted kernels to LLVM and object code, but the
project still needs a reliable simulation strategy that proves those generated
objects execute correctly on RISC-V. The immediate gap is M4 verification:
generated RVV binaries should run under simulation, compare against references,
and produce reproducible correctness and benchmark output.

### Background

The repository already has the pieces for a QEMU-first flow:

- `scripts/compile-rvv.sh` lowers KernelSmith MLIR to RVV object code with a
  `+zvl256b` target baseline.
- `tests/qemu_runner.py` runs RISC-V Linux ELFs on `qemu-riscv64` across VLENs.
- `.github/workflows/ci-rvv-sim.yml` sketches a build-then-simulate workflow.
- `scripts/setup-rvv-sim.sh` installs QEMU user-mode, a RISC-V Linux
  cross-compiler, and optional Spike.

The `pkp124/riscv` project is useful as a reference for platform taxonomy,
CI artifacts, and simulator setup. It should not be vendored wholesale because
its core flow is a bare-metal system exploration stack, while KernelSmith's
near-term need is generated kernel verification through linked C harnesses.

## Requirements

From specification: `specs/targets/riscv-rvv.md`

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | Run generated RVV kernel binaries in simulation as part of M4 validation. | Must Have |
| REQ-2 | Treat QEMU user-mode as the required fast correctness simulator. | Must Have |
| REQ-3 | Make VLEN testing profile-aware; do not run VLEN values below the artifact's `zvl` baseline. | Must Have |
| REQ-4 | Compare simulator results against scalar C or NumPy-derived references. | Must Have |
| REQ-5 | Emit stable machine-parseable output for correctness, timing, and diagnostics. | Must Have |
| REQ-6 | Keep Spike as an optional reference ISA simulator until harness ABI support is settled. | Should Have |
| REQ-7 | Add system-mode QEMU, gem5, and Renode only where they validate properties user-mode cannot. | Should Have |
| REQ-8 | Preserve local reproducibility through scripts before relying on CI-only behavior. | Must Have |

## Design

### Overview

Simulation verification is organized as a tiered matrix. Lower tiers are fast,
required, and run frequently. Higher tiers are slower or validate different
system properties, so they start as manual or scheduled checks.

| Tier | Platform | Purpose | CI Policy |
|------|----------|---------|-----------|
| 1 | QEMU user-mode (`qemu-riscv64`) | Fast Linux ELF correctness for generated kernels | Required PR gate for RVV changes |
| 2 | Spike | Reference ISA behavior and instruction traces | Manual or scheduled until harness ABI is stable |
| 3 | QEMU system-mode | Bare-metal or freestanding smoke tests | Future, after generated library ABI integration |
| 4 | gem5 | Cache and microarchitectural analysis | Scheduled/manual performance investigation |
| 5 | Renode | SoC/peripheral modeling | Future, only for RTOS or board-level integration |

### Scope

The first implementation scope is intentionally narrow:

1. Compile KernelSmith MLIR to RVV object code.
2. Link that object with a small C harness into a static RISC-V Linux ELF.
3. Run the ELF under QEMU user-mode at VLEN 256 and 512.
4. Compare results inside the harness against a scalar reference.
5. Print stable output:
   - `PASS` or `FAIL`
   - `MAX_ABS_ERROR: <value>` for floating-point kernels
   - `TIME_NS: <value>` for benchmark-enabled harnesses

This scope directly addresses `TASK-006` without requiring a bare-metal runtime,
proxy kernel, platform HAL, or RTOS memory map.

### Simulator Runner Interface

`tests/qemu_runner.py` should evolve into a generic simulator runner rather than
embedding QEMU-only assumptions in test orchestration.

Initial command shape:

```bash
python tests/riscv_runner.py \
  --sim qemu-user \
  --binary build-rvv/bin/matmul_f32_rvv \
  --profile riscv_rvv_256 \
  --vlens 256 512
```

Future command shape:

```bash
python tests/riscv_runner.py \
  --sim spike \
  --binary build-rvv/bin/matmul_f32_rvv \
  --abi pk \
  --vlen 256
```

Runner responsibilities:

- check simulator availability and print actionable install guidance;
- reject incompatible simulator/profile/VLEN combinations before execution;
- run each case with a timeout;
- capture stdout/stderr and exit code;
- validate required output markers;
- optionally normalize outputs for cross-simulator comparison.

### Test Case Descriptors

Generated-kernel tests should be declared by small descriptors so CI can build a
matrix without hardcoding every command in YAML.

Suggested JSON fields:

```json
{
  "name": "matmul_f32_128x256x128",
  "input_mlir": "tests/riscv/matmul_f32_128x256x128.mlir",
  "harness": "tests/riscv/harnesses/matmul_f32_harness.c",
  "profile": "riscv_rvv_256",
  "simulators": ["qemu-user"],
  "vlens": [256, 512],
  "benchmark": true
}
```

JSON keeps the first implementation dependency-free in Python. YAML or CMake
integration can be revisited if the matrix grows.

### VLEN and Profile Policy

The current RVV driver emits objects for a `+zvl256b` baseline. Therefore:

- VLEN 256 and 512 are valid for `riscv_rvv_256` artifacts.
- VLEN 128 must be rejected for those artifacts.
- A future `riscv_rvv_128` profile can enable VLEN 128 testing.

This prevents false failures from running a binary on a simulator configuration
that violates the compiled artifact's architectural minimum.

### CI Rollout

The CI shape should be:

1. Build `ks-opt`.
2. Compile each RVV test case to an object with `scripts/compile-rvv.sh`.
3. Link each object with its C harness into a static RISC-V Linux ELF.
4. Upload ELF/log artifacts.
5. Run QEMU user-mode at VLEN 256 and 512.
6. Upload simulator logs.

Spike should remain a `workflow_dispatch` option until a dedicated ABI path is
landed. gem5 and Renode should not be PR gates for M4 because their value is not
fast generated-kernel correctness.

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| Vendor the `pkp124/riscv` platform stack | Rich simulator examples and CI patterns | Bare-metal HAL scope does not match immediate KernelSmith kernel verification | Rejected |
| QEMU system-mode first | Closer to bare-metal and RTOS goals | Requires linker scripts, UART/exit handling, and memory-map work before testing generated kernels | Deferred |
| Spike first | Strong ISA reference behavior | Requires proxy-kernel or HTIF-compatible harnesses and is slower to set up | Deferred |
| QEMU user-mode first | Fast, CI-friendly, matches current Linux ELF tooling | Does not validate bare-metal or peripheral assumptions | **Selected** |
| gem5 as a PR gate | Microarchitectural statistics | Slow, complex, and RVV support may lag | Rejected for PR gates |

## Test Strategy

### Local Checks

- `python tests/riscv_runner.py --help`
- `scripts/compile-rvv.sh <case.mlir> <case.o>`
- `riscv64-linux-gnu-gcc -static <case.o> <harness.c> -o <case>`
- `python tests/riscv_runner.py --sim qemu-user --binary <case> --vlens 256 512`

### CI Checks

- QEMU user-mode correctness for generated RVV matmul at VLEN 256 and 512.
- Benchmark output for at least one representative matmul shape.
- Artifact upload for ELFs and simulator logs.

### Future Checks

- Spike cross-check for the same harness output.
- QEMU system-mode freestanding smoke test once generated kernels are wired into
  the static library path.
- gem5 scheduled runs for cache and timing analysis.

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| QEMU accepts code that Spike rejects | Medium | Medium | Add Spike as manual reference once ABI path is stable |
| Benchmarks from QEMU are treated as hardware truth | Medium | Medium | Label QEMU as correctness-first; use real boards or gem5 for performance investigation |
| VLEN/profile mismatch creates noisy failures | High | Medium | Enforce profile-aware VLEN validation in the runner |
| CI setup becomes too slow | Medium | Medium | Keep QEMU as the only required PR simulator initially |
| Harnesses diverge from public C APIs | Medium | Medium | Link through the same generated object and C API path once library integration lands |

## Open Questions

- [ ] Should simulator test descriptors live under `tests/riscv/` or
      `tests/sim/` once non-RVV RISC-V tests exist?
- [ ] Should Spike use `pk` with Linux-style static ELFs or a small HTIF harness?
- [ ] Which RVV board should serve as the first hardware comparison target?
- [ ] Should gem5 track only scalar/generic kernels until upstream RVV support is
      sufficient for generated vector kernels?

## Dependencies

- `TASK-006`: M4 RVV correctness and benchmark validation.
- `scripts/compile-rvv.sh`: generated RVV object pipeline.
- `tests/qemu_runner.py`: current QEMU-only runner.
- `scripts/setup-rvv-sim.sh`: local simulator dependency setup.
- RISC-V Linux cross-compiler and QEMU user-mode packages.

## Implementation Plan

### Phase 1: QEMU User-Mode Correctness

- [ ] Rename or replace `tests/qemu_runner.py` with a profile-aware
      `tests/riscv_runner.py`.
- [ ] Change default `riscv_rvv_256` VLEN coverage to 256 and 512.
- [ ] Add a generated matmul RVV harness with scalar reference comparison.
- [ ] Link the generated object and harness into a static RISC-V Linux ELF.
- [ ] Run the harness under QEMU user-mode in CI.

### Phase 2: Benchmark Reporting

- [ ] Standardize `TIME_NS`, `MAX_ABS_ERROR`, and `PASS` output markers.
- [ ] Add benchmark mode to the runner.
- [ ] Upload benchmark logs as CI artifacts.

### Phase 3: Spike Reference Path

- [ ] Decide between `pk` and HTIF harness support.
- [ ] Add manual Spike workflow dispatch.
- [ ] Normalize QEMU and Spike result lines for cross-simulator comparison.

### Phase 4: System and Analysis Platforms

- [ ] Add QEMU system-mode only when freestanding or RTOS validation is needed.
- [ ] Add gem5 scheduled jobs for performance analysis after correctness is stable.
- [ ] Add Renode only for board/peripheral validation scenarios.

---

## Review History

### Review 1 (pending)

**Reviewer**: TBD
**Decision**: Pending

**Feedback**:
- Pending review.

**Resolution**:
- Pending review.
