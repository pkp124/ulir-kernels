# KernelSmith

Optimized ML inference kernels for edge, embedded, and physical AI.

KernelSmith is a RISC-V-first kernel library. You link `libkernelsmith.a` and
call a stable C API. The MLIR compiler (`ks-opt`) is the build-time tool that
lowers `ks.` operations toward that library. Model runtimes such as llama.cpp
are downstream consumers.

- **Primary target:** RISC-V RVV
- **Secondary target:** ARM NEON, after the RVV path is proven
- **Quantization:** INT8 and W4A8 dot/GEMV are part of the public API

The current product goal is a quantized transformer on RISC-V. Active work is
a small llama.cpp host integration. See [ROADMAP.md](ROADMAP.md) and
[tasks/MILESTONES.md](tasks/MILESTONES.md).

## What you can use today

The C library includes handwritten f32 reference kernels and quantized
dot/GEMV kernels:

| Header | Functions |
|---|---|
| `include/kernelsmith/ks_matmul.h` | `ks_matmul_f32` |
| `include/kernelsmith/ks_activations.h` | `ks_relu_f32`, `ks_gelu_f32`, `ks_silu_f32` |
| `include/kernelsmith/ks_elementwise.h` | `ks_add_f32`, `ks_mul_f32` |
| `include/kernelsmith/ks_normalization.h` | `ks_rms_norm_f32`, `ks_softmax_f32` |
| `include/kernelsmith/ks_quantized.h` | `ks_dot_i8`, `ks_matvec_i8`, `ks_dot_w4a8`, `ks_matvec_w4a8` |

INT8 RVV objects can replace the INT8 reference kernels when the library is
built with the `riscv_rvv_256` profile. W4A8 still uses the reference
implementation. Matmul and activation objects are still handwritten; their
MLIR lowering exists and is tested, and swapping those objects in is open
work from milestones M2 and M3.

## Quick start

Requirements: LLVM/MLIR 21 or newer, CMake 3.20 or newer, Ninja, Python 3.10
or newer, and a C++17 compiler.

```bash
./scripts/setup.sh
make build
make test
```

`./scripts/setup.sh` installs the LLVM 21 packages it needs, creates `.venv`,
configures CMake, builds, and runs CTest. After that, rebuild with
`make build` and retest with `make test`.

Call ReLU from C:

```c
#include "kernelsmith/ks_activations.h"
#include "kernelsmith/ks_common.h"

int main(void) {
  float in[4] = {-1.0f, 0.0f, 0.5f, 2.0f};
  float out[4];
  if (ks_relu_f32(in, out, 4) != KS_OK)
    return 1;
  return 0;
}
```

```bash
cc -std=c99 examples/relu.c -I include -include target/generic.h \
  lib/kernelsmith/ks_common.c lib/kernelsmith/ks_activations.c \
  -lm -o /tmp/ks_relu
```

A full CMake build produces `build/lib/kernelsmith/libkernelsmith.a`. The
profile header is injected by the library target, so installed users include
only the public headers.

Lower an MLIR example with the compiler:

```bash
build/bin/ks-opt examples/relu.mlir --ks-lower-activations
build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg --ks-tile
```

Guides: [getting started](docs/guides/getting-started.md),
[adding kernels](docs/guides/adding-kernels.md),
[testing](docs/guides/testing-guide.md),
[architecture](docs/architecture/overview.md).

## Supported kernels

Parse means `ks-opt` can round-trip the operation. Verify means a verifier
rejects invalid IR. Lower means a pass rewrites the operation into standard
MLIR. C API means a public function exists in `libkernelsmith`.

| Kernel | Parse | Verify | Lower | C API |
|---|:---:|:---:|:---:|:---:|
| `ks.matmul` | Yes | Yes | linalg, tile, pack, vector, RVV | `ks_matmul_f32` |
| `ks.batch_matmul` | Yes | Yes | — | — |
| `ks.conv2d` | Yes | Yes | — | — |
| `ks.attention` | Yes | Yes | — | — |
| `ks.add` | Yes | Yes | linalg | `ks_add_f32` |
| `ks.mul` | Yes | Yes | linalg | `ks_mul_f32` |
| `ks.relu` | Yes | — | activations | `ks_relu_f32` |
| `ks.gelu` | Yes | — | activations | `ks_gelu_f32` |
| `ks.silu` | Yes | — | activations | `ks_silu_f32` |
| `ks.softmax` | Yes | Yes | linalg | `ks_softmax_f32` |
| `ks.rms_norm` | Yes | Yes | linalg | `ks_rms_norm_f32` |
| `ks.layer_norm` | Yes | Yes | — | — |
| `ks.reduce_sum` | Yes | Yes | — | — |
| `ks.reduce_max` | Yes | Yes | — | — |
| `ks.quantize` | Yes | Yes | — | — |
| `ks.dequantize` | Yes | Yes | — | — |
| `ks.dot_i8` | Yes | Yes | linalg | `ks_dot_i8` |
| `ks.matvec_i8` | Yes | Yes | linalg | `ks_matvec_i8` |
| `ks.dot_w4a8` | Yes | Yes | linalg | `ks_dot_w4a8` |
| `ks.matvec_w4a8` | Yes | Yes | linalg | `ks_matvec_w4a8` |

`ks.relu`, `ks.gelu`, and `ks.silu` do not have verifiers yet. Activation
lowering stops at `linalg.generic`. Matmul continues through tiling, B
packing, vectorization, and `--ks-lower-to-rvv`.

## Repository map

```text
include/kernelsmith/          Public C headers
include/KernelSmith/          MLIR dialect and pass headers
lib/kernelsmith/              C reference kernels
lib/Dialect/Kernel/           Dialect implementation
lib/Passes/                   Lowering, tiling, packing, vectorization, RVV
tools/ks-opt/                 Compiler driver
target/                       generic, riscv_rvv_256, and x86_avx2 profiles
tests/lit/                    MLIR FileCheck tests
tests/golden/                 NumPy golden cases
specs/                        Kernel and target specifications
docs/design/                  Design decisions (DES-XXX)
docs/guides/                  User and contributor guides
tasks/                        Milestone dashboard and task packets
examples/                     Small MLIR and C samples
```

## Development

1. Read the spec in `specs/`.
2. Check `docs/design/` before changing architecture, passes, or the C API.
3. Write a lit test, then implement the change.
4. Run `make test` before committing.

Details are in [CONTRIBUTING.md](CONTRIBUTING.md). Agents should start at
[AGENTS.md](AGENTS.md).

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
