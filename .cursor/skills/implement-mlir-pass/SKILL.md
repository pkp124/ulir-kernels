# Implement a KernelSmith MLIR Pass

Use this skill for new or modified `--ks-*` compiler passes.

## TDD workflow

1. Read the relevant spec in `specs/` and design docs in `docs/design/`.
2. Write or update lit tests in `tests/lit/Passes/` first:
   - positive before/after FileCheck case,
   - `CHECK-NOT` for removed ops,
   - invalid/diagnostic case when applicable.
3. Define or update the pass in `include/KernelSmith/Passes/Passes.td`.
4. Implement the pass in `lib/Passes/<Name>Pass.cpp`.
5. Add the source file to `lib/Passes/CMakeLists.txt`.
6. Keep dialect dependencies explicit in TableGen.
7. Build and run tests:
   ```bash
   cmake --build build --parallel
   ctest --test-dir build --output-on-failure
   ```

## Local conventions

- C++ class: `KS<Action>Pass`.
- CLI flag: `--ks-action`.
- Namespace: `kernelsmith` / `kernelsmith::ks` as appropriate.
- Use MLIR/LLVM APIs and dialect interfaces; avoid ad hoc IR string matching.
- Preserve the `ks -> linalg -> vector -> target` layering.
