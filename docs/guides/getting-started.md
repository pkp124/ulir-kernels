# Getting Started

This guide builds KernelSmith and runs one C kernel and one MLIR lowering.

## Prerequisites

Required:

- LLVM/MLIR 21 or newer, with the MLIR tools installed
- CMake 3.20 or newer
- Ninja
- Python 3.10 or newer
- A C++17 compiler (GCC 10+ or Clang 13+)

Optional:

- Docker, for `./scripts/docker-verify.sh`
- `qemu-riscv64` and `riscv64-linux-gnu-gcc`, for RVV tests

On Ubuntu, `./scripts/setup.sh` adds the LLVM 21 apt repository and installs
`mlir-21-tools`, `libmlir-21-dev`, and `llvm-21-dev`.

## Build

```bash
git clone https://github.com/pkp124/ulir-kernels.git
cd ulir-kernels
./scripts/setup.sh
```

Setup creates `.venv`, configures `build/`, compiles, and runs CTest. Later
changes use the shorter loop:

```bash
source .venv/bin/activate
make build
make test
```

Makefile targets: `setup`, `build`, `build-debug`, `test`, `lit`, `unit`,
`capi`, `lint`, `verify`, and `clean`.

## Use the C library

Public headers live in `include/kernelsmith/`. The reference implementation
lives in `lib/kernelsmith/`. `examples/relu.c` calls `ks_relu_f32`:

```bash
cc -std=c99 examples/relu.c \
  -I include -include target/generic.h \
  lib/kernelsmith/ks_common.c \
  lib/kernelsmith/ks_activations.c \
  -lm -o /tmp/ks_relu
/tmp/ks_relu
```

`-include target/generic.h` supplies the macros the C sources expect. The
CMake library target does this for you and writes
`build/lib/kernelsmith/libkernelsmith.a`. Link that archive, and include the
headers, when you embed KernelSmith in another program. Choose a different
profile by configuring CMake with `-DKS_TARGET_PROFILE=riscv_rvv_256`.

Functions return `KS_OK` (`0`) on success. `KS_ERR_INVALID_ARG` means a null
pointer or an illegal dimension. See `include/kernelsmith/ks_common.h`.

## Lower an MLIR example

Examples are in `examples/`. `ks-opt` prints the module when you omit `-o`.

Activations:

```bash
build/bin/ks-opt examples/relu.mlir --ks-lower-activations
```

`ks.relu` becomes `linalg.generic` with `arith.maximumf`. `ks.gelu` uses
`math.erf`. `ks.silu` uses `math.exp`.

Matmul, one stage at a time:

```bash
build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg
build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg --ks-tile
```

The first command produces `linalg.fill` and `linalg.matmul`. The second wraps
that matmul in `scf.for` tile loops.

The RVV object path for matmul is `scripts/compile-rvv.sh`. It runs
lower-to-linalg, two tile passes, pack, vectorize, and lower-to-rvv, then
`mlir-translate` and `llc`. LLVM tools on some machines are named
`mlir-translate-21` and `llc-21`; point the script at them with
`MLIR_TRANSLATE` and `LLC`.

```bash
export MLIR_TRANSLATE=mlir-translate-21
export LLC=llc-21
./scripts/compile-rvv.sh examples/matmul.mlir /tmp/matmul.s
```

Use `-float-abi=hard` with `llc-21`. The script already does. The matmul
script is specific to `ks.matmul`. Activations use `--ks-lower-activations`
before any RVV lowering.

## Add a kernel or a pass

Scaffolding scripts write a spec, a lit test, and a starting file. They are
not Makefile targets.

```bash
./scripts/new-kernel.sh my_kernel
./scripts/new-pass.sh optimize-something
./scripts/new-design.sh 017 "Short Title"
```

Read [Adding Kernels](adding-kernels.md) before filling those files in. The
dialect prefix is `ks.`, and operation definitions live in
`include/KernelSmith/Dialect/Kernel/KernelOps.td`.

## Scope of the samples

The supported user API is the C library. Conv2d, attention, batch matmul,
layer norm, quantize/dequantize, and the reductions parse and verify. Their
lowering passes are still future work, tracked from M10 in the roadmap.

## Next

- [Architecture overview](../architecture/overview.md)
- [Testing guide](testing-guide.md)
- [Kernel specs](../../specs/kernels/)
- [Roadmap](../../ROADMAP.md)
