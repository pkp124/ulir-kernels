# KernelSmith - Claude Code Project Guide

## Project Identity

KernelSmith is an MLIR-based compiler framework that generates optimized ML inference kernels for **edge, embedded, and physical AI** devices. Primary target: RISC-V RVV. Secondary target: ARM NEON. Quantization is a core feature.

- **Language**: C++17 (core), Python (tooling/tests)
- **Compiler infra**: MLIR / LLVM 21+
- **Namespace**: `kernelsmith`, dialect prefix `ks`, CLI tool `ks-opt`
- **Domain**: Quantized inference on resource-constrained hardware (RISC-V edge SoCs, ARM phones/SBCs)
- **License**: AGPL-3.0

## Quick Reference Commands

```bash
# Full setup (installs LLVM/MLIR 21, cmake, ninja, Python deps, builds, tests)
./scripts/setup.sh

# Rebuild after changes
source .venv/bin/activate
cmake --build build --parallel
ctest --test-dir build --output-on-failure

# Container verify — run this before every push to confirm lint + build + tests
# all pass inside the container (matches CI exactly). Cached after first run;
# add --no-cache for a clean build.
./scripts/docker-verify.sh

# Quality
ruff check .                    # Python lint
ruff format --check .           # Python format check
clang-format -i lib/**/*.cpp    # C++ format

# Development helpers
./scripts/new-kernel.sh <name>              # Scaffold a new kernel op
./scripts/new-pass.sh <name>                # Scaffold a new pass
./scripts/new-design.sh <id> "<title>"      # Create design doc
```

## Architecture & Lowering Pipeline

```
ks.matmul / ks.conv2d / ...                [IMPLEMENTED - parse/verify]
    | --ks-lower-to-linalg                 [IMPLEMENTED - M3]
linalg.matmul (standard MLIR linalg)
    | --ks-tile (L2)                       [IMPLEMENTED - M3]
scf.for (L2-tiled loops)
    | --ks-pack                            [IMPLEMENTED - M4]
packed operands (B in [N/NR, K, NR] layout)
    | --ks-tile (register)                 [IMPLEMENTED - M3/M4]
scf.for (register-tiled)
    | --ks-vectorize                       [IMPLEMENTED - M4]
vector.transfer_read / vector.contract / vector.transfer_write
    | --ks-lower-to-rvv (RVV)             [IMPLEMENTED - M4]
    | or LLVM autovectorize (ARM NEON)
LLVM dialect -> mlir-translate -> llc -march=riscv64 -mattr=+v
RISC-V RVV assembly (vle32.v / vfmacc.vv / vse32.v)
```

### Target Priority
1. **RISC-V RVV** (primary) — custom `--ks-lower-to-rvv` pass, QEMU testing
2. **ARM NEON** (secondary) — LLVM autovectorization, profile-driven tile sizes
3. Generic C (reference/fallback)

## Directory Layout

| Path | Purpose |
|------|---------|
| `include/KernelSmith/Dialect/Kernel/` | TableGen (.td) and headers for the ks dialect |
| `lib/Dialect/Kernel/` | Dialect, ops, types implementation (.cpp) |
| `lib/Passes/` | Pass implementations for lowering, tiling, packing, vectorization, and target conversion |
| `tools/ks-opt/` | CLI optimizer entry point |
| `tests/lit/` | MLIR FileCheck tests (.mlir) |
| `tests/unit/` | C++ unit tests (Google Test) |
| `target/` | Target profile headers (one per hardware target) |
| `specs/` | Feature specifications (read before implementing) |
| `docs/design/` | Design decision docs (DES-XXX format) |
| `tasks/` | Task tracking files (TASK-XXX format) |

## Code Conventions

### C++ (LLVM Style)

- 2-space indent, 80 col limit, `.clang-format` at repo root
- Dialect: `KS_Dialect` in `kernelsmith::ks` namespace
- Operations: `KS_VerbNounOp` (e.g., `KS_MatmulOp`)
- Types: `KS_TypeNameType` (e.g., `KS_TileType`)
- Passes: `KS{Action}Pass` class, `--ks-action` CLI flag
- Include order: project > MLIR > LLVM > system

### TableGen Operations Template

```tablegen
def KS_ExampleOp : KS_Op<"example", [Pure]> {
  let summary = "One-line description";
  let description = [{
    Detailed description with MLIR example.
  }];
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
  let hasVerifier = 1;
}
```

### Python

- Formatter/linter: `ruff` (config in `pyproject.toml`)
- Line length: 100, target Python 3.10+
- Test framework: `pytest`

## Development Workflow

1. **Read the spec** in `specs/` before implementing any kernel or pass
2. **Check design docs** in `docs/design/` for existing decisions
3. **TDD**: Write a failing lit/unit test first, then implement
4. **Small commits**: Each commit should pass `ctest --test-dir build`
5. **Container verify before push**: Run `./scripts/docker-verify.sh` before every `git push` to confirm lint (ruff + clang-format) and the container build + tests all pass in the same environment CI uses
6. **Commit format**: `<type>(<scope>): <subject>` (e.g., `feat(dialect): add ks.relu operation`)

### Commit Types

`feat` | `fix` | `docs` | `refactor` | `test` | `chore`

### Review Checklist

Before merging any change, verify:
- Verifiers reject all invalid inputs (negative lit tests exist)
- Lowering path is complete for any new op (ks -> linalg -> target)
- Each op has lit tests for: parse/print round-trip, verifier errors, transformations
- Dialect layering is correct (no circular dependencies between passes)

## Testing Patterns

### Lit test (parse/print round-trip)

```mlir
// RUN: ks-opt %s | FileCheck %s
// CHECK-LABEL: func @test_op
func.func @test_op(%arg0: tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: ks.relu
  %0 = ks.relu %arg0 : tensor<32xf32>
  return %0 : tensor<32xf32>
}
```

### Lit test (verifier / invalid input)

```mlir
// RUN: ks-opt %s -split-input-file -verify-diagnostics
func.func @test_error(%arg0: tensor<64xf32>, %arg1: tensor<128x256xf32>) {
  // expected-error @+1 {{left operand must be a 2D tensor}}
  %0 = ks.matmul %arg0, %arg1 : tensor<64xf32>, tensor<128x256xf32> -> tensor<64x256xf32>
}
```

### Lit test (pass transformation)

```mlir
// RUN: ks-opt %s --ks-lower-to-linalg | FileCheck %s
// CHECK-LABEL: func @test_lower
// CHECK-NOT: ks.matmul
// CHECK: linalg.matmul
```

## RISC-V RVV Specifics

- Vector Length Agnostic (VLA): VLEN is runtime-determined (128-16384 bits)
- Key params: VLEN, SEW (element width), LMUL (register grouping), VLMAX
- Lowering: `vector.load` -> `vle{SEW}.v`, `vector.fma` -> `vfmacc.vv`, etc.
- Test with multiple VLENs via QEMU: `tests/qemu_runner.py`
- Minimize `vsetvl` instructions; use LMUL > 1 for compute-bound kernels

## Current Status

### Implemented
- KS dialect with 13 operations (matmul, batch_matmul, conv2d, attention, relu, gelu, silu, softmax, layer_norm, rms_norm, reduce_sum, reduce_max)
- Verifiers for 9 ops (matmul, batch_matmul, conv2d, attention, softmax, layer_norm, rms_norm, reduce_sum, reduce_max)
- TileType custom type
- `ks-opt` CLI tool with all M1–M4 passes registered
- `--ks-lower-activations` pass: lowers relu/gelu/silu to linalg.generic + arith/math ops
- C kernel library (`lib/kernelsmith/`): `ks_matmul_f32`, `ks_relu_f32`, `ks_gelu_f32`, `ks_silu_f32`
- Lit tests: parse/print round-trip, verifier negative tests (31 error cases), pass transformation
- C API tests: matmul and activation smoke tests with NumPy validation
- C++ unit test: dialect loading
- CI: lint (ruff) + native build + container build (Docker) + GHCR publish
- Python test infrastructure (test_data_generator, functional_validator, qemu_runner)
- LLVM/MLIR 21 (bumped from 20)
- `--ks-lower-to-linalg` pass: lowers ks.matmul → linalg.fill + linalg.matmul (static + dynamic shapes)
- `--ks-tile` pass: tiles linalg.matmul → nested scf.for loops (profile-driven tile sizes)
- Design doc DES-008 (M3 matmul lowering to linalg, generic target)
- **M4 (RISC-V RVV Target)**:
  - `target/riscv_rvv_256.h` — RVV target profile (VLEN=256, LMUL=4, f32)
  - `--ks-pack` pass: packs B operand into `[N/NR, K, NR]` column-panel layout
  - `--ks-vectorize` pass: linalg → vector dialect (`vector.contract` + transfer ops)
  - `--ks-lower-to-rvv` pass: full pipeline to LLVM dialect (bufferize→scf→cf→vector→LLVM)
  - `scripts/compile-rvv.sh` — driver for ks-opt + mlir-translate + llc
  - Lit tests for pack, vectorize, lower-to-rvv passes
  - QEMU runner updated for multi-VLEN correctness and benchmark testing
  - Design doc DES-009 (M4 RVV lowering pipeline)

### Not Yet Implemented
- Canonicalization patterns (MatmulOp stub exists but is empty)
- Verifiers for 3 activation ops (relu, gelu, silu — hasVerifier=0 in TableGen)
- Strengthened verifiers for layer_norm and rms_norm (currently minimal)
- Quantization ops (quantize, dequantize) and INT8/INT4 support
- Edge-critical ops (depthwise_conv2d, element-wise add/mul, pooling)
- ARM NEON target profile
- QEMU correctness tests (require cross-compiler + qemu-riscv64 in CI)

See `ROADMAP.md` for milestones (edge-first: RVV primary, ARM NEON secondary, quantization early).

## Key Specifications

Before implementing kernels, always read the relevant spec:
- `specs/kernels/matmul.md` - Matrix multiplication
- `specs/kernels/conv2d.md` - 2D convolution
- `specs/kernels/attention.md` - Attention mechanism
- `specs/targets/riscv-rvv.md` - RVV target details (primary target)
- `specs/targets/system-description.md` - Target profile format and validation

## Common Pitfalls

- Do NOT add ops without corresponding lit tests (parse + verify + transform)
- Do NOT describe unimplemented features as existing in docs or CLAUDE.md
- The `.clang-format` project include regex may reference `AIKernels/` (legacy name); all code uses `KernelSmith/`
- Always run `ctest --test-dir build` before committing
- Design docs are required for non-trivial changes; use `./scripts/new-design.sh`
- QEMU tests need specific setup; see `docs/guides/testing-guide.md`
