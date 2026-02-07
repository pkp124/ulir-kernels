# KernelSmith Roadmap

## Milestone 1: Reference Kernels in CI (Current)

**Goal**: At least one lowering pass running in CI with lit tests that verify the transformation.

**Approach**: Start with activation ops (relu, gelu, silu) because they are element-wise, need no tiling, and have `SameOperandsAndResultType`. This validates the full pass infrastructure before tackling matmul.

**Tasks**:
1. Implement `KSLowerActivationsPass` (`--ks-lower-activations`)
   - `ks.relu` -> `arith.maxf(input, zero)`
   - `ks.gelu` -> `math.erf` + arith (approximate)
   - `ks.silu` -> `math.exp` + arith (x * sigmoid(x))
2. Create `include/KernelSmith/Passes/Passes.h` and `Passes.td` for pass registration
3. Register pass in `lib/Passes/PassRegistration.cpp`
4. Write lit tests: `tests/lit/Passes/lower-activations.mlir`
5. Verify in CI: `ctest` runs, pass transforms, FileCheck validates

## Milestone 2: MatMul Lowering to Linalg

**Goal**: `ks.matmul` -> `linalg.matmul` + `tensor.empty`

**Tasks**:
1. Implement `KSLowerToLinalgPass` (`--ks-lower-to-linalg`) for matmul
2. Lit test: verify `ks.matmul` is replaced with `linalg.matmul`
3. Extend to `ks.batch_matmul` -> `linalg.batch_matmul`
4. Extend to `ks.conv2d` -> `linalg.conv_2d_nhwc_hwio`

## Milestone 3: Tiling Pass

**Goal**: `linalg.matmul` with configurable tiling via `scf.for` loops.

**Tasks**:
1. Implement `KSTilePass` (`--ks-tile`) using MLIR's tiling infrastructure
2. Support tile sizes from op attributes or pass options
3. Lit tests with various tile configurations

## Milestone 4: Vectorization

**Goal**: Tiled loops -> MLIR vector operations.

**Tasks**:
1. Implement `KSVectorizePass` (`--ks-vectorize`)
2. Generate `vector.load`, `vector.fma`, `vector.store`
3. Lit tests verifying vector op generation

## Milestone 5: RVV Lowering

**Goal**: Vector ops -> LLVM IR with RISC-V RVV intrinsics.

**Tasks**:
1. Implement `KSLowerToRVVPass` (`--ks-lower-to-rvv`)
2. Map vector ops to RVV: `vle{SEW}.v`, `vfmacc.vv`, etc.
3. Minimize `vsetvl` instructions
4. Handle tail elements with masking
5. Lit tests verifying RVV intrinsic generation

## Milestone 6: End-to-End Validation

**Goal**: Full pipeline from `ks.matmul` to RISC-V assembly, validated against numpy reference via QEMU.

**Tasks**:
1. Integration test: `ks-opt` full pipeline -> LLVM IR -> `llc` -> assembly
2. QEMU execution with `tests/qemu_runner.py`
3. Numerical validation against `tests/functional_validator.py`
4. Test across VLEN configurations (128, 256, 512, 1024)

---

See `RISC-V_RVV_KERNEL_LIBRARY_PLAN.md` for detailed technical reference (RVV specifics, test pyramid, QEMU setup).
