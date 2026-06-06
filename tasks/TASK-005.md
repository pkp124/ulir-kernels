# TASK-005: Implement RISC-V RVV Lowering Pipeline

## Status
[x] Complete

## Priority
P1 (High)

## Milestone
M4 — RISC-V RVV Target

## Owner Agent
`.cursor/agents/rvv-validation-agent.md`

## Description

Implement the checked-in lowering path from KernelSmith/vector-level IR to LLVM
dialect suitable for RISC-V RVV code generation:

1. **KSLowerToRVVPass**: run the bufferize/loop/control-flow/vector-to-LLVM
   pipeline.
2. **Driver integration**: use `scripts/compile-rvv.sh` to invoke `ks-opt`,
   `mlir-translate`, and `llc` with RISC-V V extension flags.

## Acceptance Criteria

- [x] `KSLowerToRVVPass` defined in Passes.td as `--ks-lower-to-rvv`
- [x] Vector/tensor/linalg/scf IR lowers to LLVM dialect
- [x] Driver script exists for `ks-opt` + `mlir-translate` + `llc`
- [x] Lit coverage exists for `--ks-lower-to-rvv`
- [x] Pack/vectorize/lower-to-rvv path is represented in tests
- [x] QEMU runtime validation follow-up moved to `TASK-006`

## Implementation Notes

### Target Architecture

See: `specs/targets/riscv-rvv.md`

Key considerations:
- Scalable vectors (VLEN not fixed)
- LMUL configuration
- Mask handling
- vsetvl placement optimization

### Pass Structure

```cpp
struct KSLowerToRVVPass : public PassWrapper<...> {
  void runOnOperation() override {
    // 1. one-shot-bufferize
    // 2. convert-linalg-to-loops
    // 3. lower-affine / convert-scf-to-cf
    // 4. convert-vector-to-llvm
    // 5. finalize-memref/arith/func-to-llvm
  }
};
```

### LLVM Integration

The LLVM RISC-V backend emits RVV instructions from vector/LLVM IR when `llc`
is invoked with RVV target features:
```llvm
llc -march=riscv64 -mattr=+v,+zve64d,+zvl256b ...
```

## Test Cases

1. **Lower-to-RVV lit**: Pipeline reaches LLVM dialect
2. **Pack/vectorize lit**: Vector dialect produced before RVV lowering
3. **QEMU correctness**: Tracked by `TASK-006`
4. **Benchmark**: Tracked by `TASK-006`

## Dependencies

- TASK-004: Tiling and vectorization
- TASK-006: RVV validation and benchmark follow-up

## Specification

See: `specs/targets/riscv-rvv.md`

## Verification

```bash
ctest --test-dir build -R kernelsmith-lit --output-on-failure
ctest --test-dir build --output-on-failure
```

## Notes

- This task tracks the compiler pipeline implementation.
- Runtime QEMU execution, speed comparison, and non-divisible shape hardening are
  intentionally split into `TASK-006`.

## Log

### 2026-06-06
- Updated stale status from not started to complete based on the checked-in
  `--ks-lower-to-rvv` pass, driver script, and lit coverage.
- Moved RVV runtime validation and benchmark work into `TASK-006`.
