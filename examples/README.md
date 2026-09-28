# Examples

Small programs that match the current compiler and C library. They are not
built by CMake. The commands below assume `make build` has produced
`build/bin/ks-opt`.

## C API

`relu.c` calls `ks_relu_f32` and prints `0`, `0`, `0.5`, `2`.

```bash
cc -std=c99 examples/relu.c \
  -I include -include target/generic.h \
  lib/kernelsmith/ks_common.c \
  lib/kernelsmith/ks_activations.c \
  -lm -o /tmp/ks_relu
/tmp/ks_relu
```

## MLIR

`relu.mlir` lowers with the activation pass. The result contains
`linalg.generic` and `arith.maximumf`, and it no longer contains `ks.relu`.

```bash
build/bin/ks-opt examples/relu.mlir --ks-lower-activations
```

`matmul.mlir` lowers to `linalg.matmul`. Adding `--ks-tile` wraps that matmul
in `scf.for` loops.

```bash
build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg
build/bin/ks-opt examples/matmul.mlir --ks-lower-to-linalg --ks-tile
```

RISC-V assembly for this matmul:

```bash
MLIR_TRANSLATE=mlir-translate-21 LLC=llc-21 \
  ./scripts/compile-rvv.sh examples/matmul.mlir /tmp/matmul.s
```

Set `MLIR_TRANSLATE` and `LLC` only when the unversioned tool names are absent.
The script looks for `mlir-translate` and `llc` by default.
