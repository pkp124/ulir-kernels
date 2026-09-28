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

1. **Check task context** in `tasks/MILESTONES.md` and the active
   `tasks/TASK-XXX.md`; use `.cursor/skills/manage-kernelsmith-tasks/SKILL.md`
   for task grooming or status updates
2. **Read the spec** in `specs/` before implementing any kernel or pass
3. **Check design docs** in `docs/design/` for existing decisions
4. **TDD**: Write a failing lit/unit test first, then implement
5. **Small commits**: Each commit should pass `ctest --test-dir build`
6. **Container verify before push**: Run `./scripts/docker-verify.sh` before every `git push` to confirm lint (ruff + clang-format) and the container build + tests all pass in the same environment CI uses
7. **Commit format**: `<type>(<scope>): <subject>` (e.g., `feat(dialect): add ks.relu operation`)

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

Read `tasks/MILESTONES.md` before choosing work. The user-facing kernel table
is in `README.md`. As of 2026-09-28:

### Active
- **M7 — known-runtime transformer smoke.** `DES-016` is under review
  (`TASK-025`). `TASK-026` (pin llama.cpp and a CI-sized model) has not started.
- M2 and M3 passes exist. Handwritten matmul and activation objects in
  `libkernelsmith.a` are still the linked implementations.

### Implemented
- 20 `ks.` operations: structured compute, activations, normalization,
  reductions, elementwise arithmetic, and quantization (dot/GEMV, not GEMM)
- Verifiers for those operations except `ks.relu`, `ks.gelu`, and `ks.silu`
- C kernel library with f32 matmul, activations, add/mul, RMSNorm, softmax,
  and INT8/W4A8 dot/GEMV
- Lowering: activations; matmul through tile, pack, vectorize, and RVV;
  add/mul; RMSNorm; softmax; INT8 and W4A8 dot/GEMV to linalg
- Generated INT8 RVV objects can be linked behind `ks_dot_i8` / `ks_matvec_i8`
- Golden host checks and QEMU user-mode checks at VLEN 256 and 512
- Passes: `--ks-lower-activations`, `--ks-lower-to-linalg`, `--ks-tile`,
  `--ks-pack`, `--ks-materialize-pack-workspace`, `--ks-vectorize`,
  `--ks-lower-to-rvv`, `--ks-alloc-check`

### Not Yet Implemented
- Canonicalization patterns (`MatmulOp::getCanonicalizationPatterns` is empty)
- Verifiers for relu, gelu, and silu (`hasVerifier` is unset)
- Quantized GEMM, and W4A8 generated-object integration (`TASK-030`)
- llama.cpp integration (`TASK-026` onward)
- `ks.depthwise_conv2d`, pooling, conv2d lowering, attention lowering
- ARM NEON target profile

See `ROADMAP.md` for the milestone sequence.

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
