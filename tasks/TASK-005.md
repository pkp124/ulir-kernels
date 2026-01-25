# TASK-005: Implement RISC-V RVV Lowering

## Status
[ ] Not Started

## Priority
P1 (High)

## Description

Implement lowering passes from vector dialect to RISC-V RVV:

1. **LowerVectorToRVVPass**: Map vector ops to RVV intrinsics
2. **RVVOptimizationPass**: Target-specific optimizations

## Acceptance Criteria

- [ ] Vector load/store → vle/vse
- [ ] Vector arithmetic → RVV ops (vfadd, vfmul, etc.)
- [ ] Vector FMA → vfmacc
- [ ] Reductions → vfredsum, vfredmax
- [ ] Proper vsetvl insertion
- [ ] Tail handling works correctly
- [ ] LLVM IR generated correctly
- [ ] Assembly runs on QEMU with RVV

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
struct LowerVectorToRVVPass : public PassWrapper<...> {
  void runOnOperation() override {
    // 1. Insert vsetvl operations
    // 2. Convert vector ops to RVV intrinsics
    // 3. Handle masking
  }
};
```

### Key Patterns

```cpp
// vector.load → rvv.vle
struct VectorLoadToRVV : public OpRewritePattern<vector::LoadOp> {
  LogicalResult matchAndRewrite(...) {
    // Insert vsetvl if needed
    // Create rvv.vle operation
  }
};

// vector.fma → rvv.vfmacc
struct VectorFMAToRVV : public OpRewritePattern<vector::FMAOp> {
  ...
};
```

### LLVM Integration

RVV operations lower to LLVM intrinsics:
```llvm
%vl = call i64 @llvm.riscv.vsetvli.i64(i64 %n, i64 2, i64 0)
%v = call <vscale x 4 x float> @llvm.riscv.vle.nxv4f32.i64(...)
```

## Test Cases

1. **Simple vector add**: Load, add, store
2. **FMA loop**: Fused multiply-accumulate
3. **Reduction**: Sum reduction
4. **Tail handling**: Non-multiple-of-VL sizes
5. **End-to-end**: Matmul from kernel op to assembly

## Dependencies

- TASK-004: Tiling and vectorization

## Specification

See: `specs/targets/riscv-rvv.md`

## Verification

```bash
# Transformation tests
make test-lit TESTS=tests/lit/Targets/RISCV/

# End-to-end test on QEMU
./scripts/test-rvv.sh tests/integration/matmul_rvv.mlir
```

## Log

### [Date TBD]
- Task created
