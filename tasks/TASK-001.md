# TASK-001: KernelSmith Dialect Infrastructure

## Status
[x] Complete

## Priority
P0 (Critical)

## Milestone
M0 — Dialect Infrastructure

## Owner Agent
General

## Description

Set up the foundational infrastructure for KernelSmith:
- CMake build configuration with CTest
- Kernel dialect definition (TableGen)
- Basic operations (relu, softmax, matmul)
- ks-opt tool driver
- Lit test infrastructure

## Acceptance Criteria

- [x] CMake configured for MLIR/LLVM
- [x] TableGen generates headers
- [x] Kernel dialect compiles
- [x] ks-opt tool builds
- [x] Lit tests configured with CTest
- [x] Agent workflow documented

## Implementation Notes

Created comprehensive project structure with:
- `.agents/` - Agent skills and workflows
- `docs/design/` - Design document templates
- Cursor rules for AI-assisted development
- TDD workflow documentation

## Verification

```bash
cmake -B build -DMLIR_DIR=/path/to/mlir
cmake --build build
ctest --test-dir build
```
