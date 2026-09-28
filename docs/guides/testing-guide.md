# Testing Guide

KernelSmith tests run through CTest. Start there. The sections below are the
extra commands for MLIR, the C library, golden numbers, and QEMU.

## Run the suite

```bash
make test
```

That is `ctest --test-dir build --output-on-failure`. Narrower targets:

| Command | What it runs |
|---|---|
| `make lit` | MLIR FileCheck tests (`check-kernelsmith-lit`) |
| `make unit` | C++ GoogleTest (`kernelsmith-unit`) |
| `make capi` | C smoke tests and NumPy checks |
| `make lint` | `ruff check` and `ruff format --check` |
| `make verify` | lint, build, and test |

`./scripts/run-tests.sh` wraps the same groups:

```bash
./scripts/run-tests.sh --lit
./scripts/run-tests.sh --unit
./scripts/run-tests.sh --integration
./scripts/run-tests.sh --riscv-functional
```

`--all` runs lit, unit, and integration. It does not cross-compile or boot
QEMU. Use `--riscv-functional` for the golden RVV run, and install
`qemu-riscv64` plus `riscv64-linux-gnu-gcc` first
(`./scripts/setup-rvv-sim.sh` on a machine that does not already have them).

One lit file:

```bash
cmake --build build --target check-kernelsmith-lit
./build/bin/ks-opt tests/lit/Dialect/Kernel/matmul.mlir
```

The lit substitution is `%ks-opt`. Tests live in `tests/lit/Dialect/Kernel/`
and `tests/lit/Passes/`.

## Lit tests

A parse test checks that the printer emits the operation you wrote:

```mlir
// RUN: ks-opt %s | FileCheck %s

// CHECK-LABEL: func @test_relu
func.func @test_relu(%input: tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: ks.relu
  %output = ks.relu %input : tensor<32xf32>
  return %output : tensor<32xf32>
}
```

A verifier test uses `-verify-diagnostics` and `expected-error` on the next
line. See `tests/lit/Dialect/Kernel/matmul-invalid.mlir`.

A pass test names the flag and the ops that must appear or disappear. See
`tests/lit/Passes/lower-to-linalg.mlir` and
`tests/lit/Passes/lower-activations.mlir`.

## C++ unit tests

`tests/unit/KernelDialectTest.cpp` checks that the `ks` dialect loads. Add
operation-builder tests there when a behavior is awkward to express as MLIR
text. Most compiler behavior belongs in lit tests.

## C API and NumPy

`make capi` builds the C smoke programs and compares selected kernels with
NumPy. Headers under test are the public `include/kernelsmith/*.h` files.
Reference sources are `lib/kernelsmith/*.c`.

## Golden references

Golden cases are the numerical source of truth shared by the host reference
runner and the RISC-V runner. Each case is a JSON descriptor in
`tests/golden/cases/`. Generate its NumPy bundle with:

```bash
python3 tests/golden/generate.py \
  --case tests/golden/cases/relu_f32_smoke.json \
  --print-manifest
```

The generator writes `tests/golden/generated/<case>/` with `manifest.json`,
input arrays, and expected arrays. The manifest records the schema version,
seed, dtypes, comparison policy, and a SHA-256 hash of each `.npy` file.

Compare a host run:

```bash
python3 tests/verify.py \
  --case tests/golden/cases/relu_f32_smoke.json \
  --target host_reference \
  --host-runner build/tests/host_reference/host-reference-runner
```

CTest runs the smoke cases as `kernelsmith-host-golden-*`. Current cases cover
f32 relu, matmul, add, mul, RMSNorm, and softmax, plus INT8 and W4A8 dot/GEMV.

To add a case:

1. Add a descriptor in `tests/golden/cases/`.
2. Use a `kernel` and `generator.function` that `tests/golden/generate.py`
   already implements.
3. Use `mode: allclose` with `rtol` and `atol` for f32 results.
4. For integer results, set `rounding` and `saturation`, and use
   `quantized_exact` or `dequantized_allclose`.
5. Generate the bundle and extend `tests/test_golden_verify.py` if the case
   needs a new comparison rule.

`tests/functional_validator.py` supplies `compare_arrays`. `tests/verify.py`
uses it. The comparison report includes pass or fail, max absolute error, and
the mismatch count.

## RISC-V QEMU

`scripts/compile-rvv.sh` lowers a `ks.matmul` module to a RISC-V object or
assembly file. Tool names may be `mlir-translate-21` and `llc-21`:

```bash
MLIR_TRANSLATE=mlir-translate-21 LLC=llc-21 \
  ./scripts/compile-rvv.sh examples/matmul.mlir /tmp/matmul.o
```

The golden RVV command runs the same descriptors on the host runner and under
QEMU:

```bash
PYTHON=.venv/bin/python ./scripts/run-tests.sh --riscv-functional
```

That path checks relu, matmul, INT8 dot/GEMV, and W4A8 dot/GEMV at VLEN 256
and 512. Pass `--qemu-vlen 256` to select one VLEN. VLEN values below the
profile baseline are rejected.

QEMU timings are simulation data. Reports can include a benchmark section when
a case is invoked with `--benchmark`.

A single generated activation, compiled by hand, is also documented in
[AGENTS.md](../../AGENTS.md) under the Cloud instructions. Use
`-float-abi=hard` with `llc-21`.

## Before a commit

```bash
cmake --build build --parallel
ctest --test-dir build --output-on-failure
ruff check .
ruff format --check .
```

`./scripts/docker-verify.sh` repeats the build and tests in the CI container
and runs `clang-format`. Use it before push when Docker is available.
