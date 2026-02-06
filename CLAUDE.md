# KernelSmith - Claude Code Project Guide

## Project Identity

KernelSmith is an MLIR-based compiler framework that generates optimized AI accelerator kernels targeting RISC-V Vector Extension (RVV), with future support for ARM SVE and x86 AVX-512.

- **Language**: C++17 (core), Python (tooling/tests)
- **Compiler infra**: MLIR / LLVM 18+
- **Namespace**: `kernelsmith`, dialect prefix `ks`, CLI tool `ks-opt`
- **License**: AGPL-3.0

## Quick Reference Commands

```bash
# Build
make configure       # CMake configure (generates compile_commands.json)
make build           # Build everything
make build-debug     # Debug build

# Test
make test            # All tests via CTest
make test-lit        # MLIR lit/FileCheck tests only
make test-unit       # C++ unit tests only
make test-validate   # Python functional validation

# Quality
make lint            # Run ruff (Python) + clang-tidy (C++)
make format          # Auto-format all code
make verify          # lint + test combined

# Development helpers
make new-kernel NAME=<name>    # Scaffold a new kernel op
make new-pass NAME=<name>      # Scaffold a new pass
make new-design ID=<n> TITLE="<title>"  # Create design doc
```

## Architecture & Lowering Pipeline

```
ks.matmul (high-level KernelSmith ops)
    | --ks-lower-to-linalg
linalg.matmul (standard MLIR linalg)
    | --ks-tile
scf.for (tiled loops)
    | --ks-vectorize
vector.load / vector.fma / vector.store
    | --ks-lower-to-rvv
LLVM IR with RVV intrinsics
```

## Directory Layout

| Path | Purpose |
|------|---------|
| `include/KernelSmith/Dialect/Kernel/` | TableGen (.td) and headers for the ks dialect |
| `include/KernelSmith/Passes/` | Pass declarations |
| `lib/Dialect/Kernel/` | Dialect, ops, types implementation (.cpp) |
| `lib/Passes/` | Pass implementations |
| `tools/ks-opt/` | CLI optimizer entry point |
| `tests/lit/` | MLIR FileCheck tests (.mlir) |
| `tests/unit/` | C++ unit tests (Google Test) + Python pytest |
| `tests/integration/` | Full-pipeline integration tests |
| `specs/` | Feature specifications (read before implementing) |
| `docs/design/` | Design decision docs (DES-XXX format) |
| `tasks/` | Task tracking files (TASK-XXX format) |
| `.agents/` | Agent skill/workflow definitions |

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
4. **Small commits**: Each commit should pass `make verify`
5. **Commit format**: `<type>(<scope>): <subject>` (e.g., `feat(dialect): add ks.relu operation`)

### Commit Types

`feat` | `fix` | `docs` | `refactor` | `test` | `chore`

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

## Current Development Phase

The project is executing an 8-phase plan (see `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md`):
- Phase 1 (test infrastructure): Complete
- Phase 2 (vector ops lowering): In progress
- Phase 3-5 (matmul, conv2d, attention kernels): Planned
- Phase 6-8 (optimization, packaging, IREE integration): Future

See `DEVELOPMENT_SUMMARY.md` for detailed progress.

## Key Specifications

Before implementing kernels, always read the relevant spec:
- `specs/kernels/matmul.md` - Matrix multiplication
- `specs/kernels/conv2d.md` - 2D convolution
- `specs/kernels/attention.md` - Attention mechanism
- `specs/targets/riscv-rvv.md` - RVV target details

## Common Pitfalls

- Do NOT add ops without corresponding lit tests (parse + verify + transform)
- The `.clang-format` project include regex references `AIKernels/` (legacy name); for new includes use `KernelSmith/`
- Always run `make verify` before committing
- Design docs are required for non-trivial changes; use `make new-design`
- QEMU tests need specific setup; see `docs/guides/testing-guide.md`
