# TASK-018: Integrate Generated INT8 RVV Objects with the C API

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M5 — RISC-V Quantization Foundation

## Owner Agent
`.cursor/agents/rvv-validation-agent.md` or a future quantization-focused agent.

## Description

Replace or dispatch the public INT8 dot/GEMV C API symbols to generated RVV
objects for the `riscv_rvv_256` target while preserving the scalar reference
implementation for generic targets.

## Acceptance Criteria
- [x] Define the generated-object naming, symbol, and dispatch policy for
      `ks_dot_i8` and `ks_matvec_i8`.
- [x] Extend the build path so generated INT8 RVV objects can be linked into
      `libkernelsmith.a` without changing the public headers.
- [x] Keep generic/profile reference implementations available as fallback
      objects.
- [x] Add C API or golden tests proving the public symbols use the generated
      target objects when the RVV profile is selected.
- [x] Validate generated INT8 RVV objects under QEMU at VLEN 256 and 512.

## Dependencies
- `TASK-013`: C API and layout contract
- `TASK-014`: quantized golden validation
- `TASK-015`: RVV target profile parameters
- `TASK-016`: INT8 dialect and linalg lowering
- `docs/design/DES-014-int8-dot-and-gemv-lowering.md`

## Verification
```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
ruff check .
ruff format --check .
```

Verified on 2026-06-15 for the generated-object build integration slice:

```bash
./scripts/setup.sh
cmake --build build --parallel
ctest --test-dir build --output-on-failure -R "capi-(quantized|int8-generated-profile-build)"
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
```

Verified on 2026-06-15 for the completed generated INT8 RVV object path:

```bash
./scripts/setup.sh
python3 scripts/generate-int8-rvv-objects.py --cc riscv64-linux-gnu-gcc --source-root /workspace --output-dir build/rvv-functional/int8-objects
riscv64-linux-gnu-nm -g build/rvv-functional/int8-objects/ks_dot_i8_riscv_rvv_256.o build/rvv-functional/int8-objects/ks_matvec_i8_riscv_rvv_256.o
riscv64-linux-gnu-objdump -d build/rvv-functional/int8-objects/ks_dot_i8_riscv_rvv_256.o | rg 'vsetvli|vle8|vwmul|vredsum'
riscv64-linux-gnu-objdump -d build/rvv-functional/int8-objects/ks_matvec_i8_riscv_rvv_256.o | rg 'vsetvli|vle8|vwmul|vredsum'
PYTHON=python3 ./scripts/run-tests.sh --riscv-functional
cmake --build build --parallel
ctest --test-dir build --output-on-failure
.venv/bin/ruff check .
.venv/bin/ruff format --check .
```

The generated objects export `ks_dot_i8` and `ks_matvec_i8`, contain RVV vector
instructions, and are linked into the RISC-V golden runner with
`KS_QUANTIZED_INT8_EXTERNAL=1` so scalar INT8 definitions are suppressed. The
INT8 golden reports pass at VLEN 256 and 512 for both dot and GEMV.

## Notes
- Keep generated-object integration separate from the reference C API and
  descriptor-backed QEMU validation that `TASK-016` completed.
- Do not hardcode RVV VLEN; derive tile and profile choices from
  `target/riscv_rvv_256.h` or generated pass options.

## Log

### 2026-06-15
- Added a build-time INT8 RVV object generator that emits
  `ks_dot_i8_riscv_rvv_256.o` and `ks_matvec_i8_riscv_rvv_256.o` with public C
  API symbols backed by RVV vector reductions.
- Updated local and CI RISC-V functional runner builds to link those generated
  objects and suppress scalar INT8 C API definitions for the RVV profile.
- Validated `dot_i8_smoke` and `matvec_i8_smoke` under QEMU at VLEN 256 and 512
  through the public C API runner, with exact golden and host-reference matches.
- Added static profile-selected INT8 object replacement plumbing for
  `riscv_rvv_256`: generated objects passed through `KS_INT8_RVV_OBJECTS`
  export `ks_dot_i8`/`ks_matvec_i8`, while the scalar reference implementation
  remains the fallback when no generated objects are provided.
- Documented generated object names, public symbol policy, and no-runtime-
  dispatch behavior in DES-014.
- Added a generated-profile C API build test using a mock external INT8 object
  to prove the public symbols are replaced without changing headers. Real
  generated RVV object production and VLEN 256/512 QEMU validation remain open.

### 2026-06-14
- Created as the generated INT8 RVV object replacement follow-up after
  `TASK-016` completed INT8 dialect/linalg lowering and QEMU validation of the
  quantized C API runner.
