# KernelSmith - Claude Code Project Guide

## Project Identity

KernelSmith is an MLIR-based compiler framework that generates optimized AI accelerator kernels targeting RISC-V Vector Extension (RVV), with future support for ARM SVE and x86 AVX-512.

- **Language**: C++17 (core), Python (tooling/tests)
- **Compiler infra**: MLIR / LLVM 18+
- **Namespace**: `kernelsmith`, dialect prefix `ks`, CLI tool `ks-opt`
- **License**: AGPL-3.0

## Quick Reference Commands

```bash
# Full setup (installs LLVM/MLIR 18, cmake, ninja, Python deps, builds, tests)
./scripts/setup.sh

# Rebuild after changes
source .venv/bin/activate
cmake --build build --parallel
ctest --test-dir build --output-on-failure

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
ks.matmul (high-level KernelSmith ops)     [IMPLEMENTED - parse/verify]
    | --ks-lower-to-linalg                 [IMPLEMENTED - M3]
linalg.matmul (standard MLIR linalg)
    | --ks-tile                            [IMPLEMENTED - M3]
scf.for (tiled loops)
    | --ks-vectorize                       [NOT IMPLEMENTED]
vector.load / vector.fma / vector.store
    | --ks-lower-to-rvv                    [NOT IMPLEMENTED]
LLVM IR with RVV intrinsics
```

## Directory Layout

| Path | Purpose |
|------|---------|
| `include/KernelSmith/Dialect/Kernel/` | TableGen (.td) and headers for the ks dialect |
| `lib/Dialect/Kernel/` | Dialect, ops, types implementation (.cpp) |
| `lib/Passes/` | Pass implementations (stub — no passes yet) |
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
5. **Commit format**: `<type>(<scope>): <subject>` (e.g., `feat(dialect): add ks.relu operation`)

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
- Verifiers for 5 ops (matmul, batch_matmul, conv2d, attention, layer_norm)
- TileType custom type
- `ks-opt` CLI tool with pass pipeline support
- `KSLowerActivationsPass` (`--ks-lower-activations`): relu/gelu/silu -> linalg.generic (M2)
- `KSLowerToLinalgPass` (`--ks-lower-to-linalg`): ks.matmul -> linalg.matmul (M3)
- `KSTilePass` (`--ks-tile`): profile-driven single-level SCF tiling (M3)
- `KSAllocCheckPass` (`--ks-alloc-check`): reject stray memref.alloc (M3)
- Mixed-precision accumulator type attribute on ks.matmul (`acc_type`)
- C kernel library (`libkernelsmith.a`) with matmul and activation reference implementations
- Lit tests: parse/print round-trip, verifier negative tests, pass transformations
- C++ unit test: dialect loading
- Python test infrastructure (test_data_generator, functional_validator, qemu_runner)

### Not Yet Implemented
- Vectorization pass (ks -> vector)
- RVV lowering pass (vector -> RVV intrinsics)
- Full pipeline to .o (bufferize -> LLVM IR -> object code)
- Canonicalization patterns (MatmulOp stub exists but is empty)
- Multi-level tiling, packing, SIMD vectorization (M4)

See `ROADMAP.md` for next milestones.

## Key Specifications

Before implementing kernels, always read the relevant spec:
- `specs/kernels/matmul.md` - Matrix multiplication
- `specs/kernels/conv2d.md` - 2D convolution
- `specs/kernels/attention.md` - Attention mechanism
- `specs/targets/riscv-rvv.md` - RVV target details

## Common Pitfalls

- Do NOT add ops without corresponding lit tests (parse + verify + transform)
- Do NOT describe unimplemented features as existing in docs or CLAUDE.md
- The `.clang-format` project include regex references `AIKernels/` (legacy name); for new includes use `KernelSmith/`
- Always run `ctest --test-dir build` before committing
- Design docs are required for non-trivial changes; use `./scripts/new-design.sh`
- QEMU tests need specific setup; see `docs/guides/testing-guide.md`
