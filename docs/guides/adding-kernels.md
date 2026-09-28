# Adding a Kernel

Add a kernel in this order: specification, TableGen operation, verifier, lit
tests, lowering, then the C API if the kernel is part of the public library.

Look at a finished operation before copying a template. `ks.relu` is a small
activation. `ks.matmul` is the structured path through RVV. `ks.dot_i8` is the
quantized pattern.

## 1. Write the spec

```bash
./scripts/new-kernel.sh my_kernel
```

That creates `specs/kernels/my_kernel.md` and a parse test at
`tests/lit/Dialect/Kernel/my_kernel.mlir`. Edit the spec so it states the
math, the tensor ranks, the element types, and the error cases. Read
`specs/kernels/matmul.md` for the expected shape.

Significant passes, public C APIs, and target changes also need a design
document:

```bash
./scripts/new-design.sh 017 "My Kernel Lowering"
```

## 2. Define the operation

Add the operation to `include/KernelSmith/Dialect/Kernel/KernelOps.td`. The
dialect name is `ks` and the C++ namespace is `kernelsmith::ks`.

```tablegen
def KS_MyKernelOp : KS_Op<"my_kernel", [Pure]> {
  let summary = "One-line description";
  let description = [{
    What the operation computes.

    Example:
    ```mlir
    %out = ks.my_kernel %input : tensor<32xf32> -> tensor<32xf32>
    ```
  }];

  let arguments = (ins KS_FloatTensor:$input);
  let results = (outs KS_FloatTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input) `->` type($output)";
  let hasVerifier = 1;
}
```

Use `KS_FloatTensor` or another constraint from that file when the operation
has a closed set of element types. Rebuild after TableGen changes:

```bash
cmake --build build --parallel
```

## 3. Verify the operation

Implement `LogicalResult MyKernelOp::verify()` in
`lib/Dialect/Kernel/KernelOps.cpp`. Follow the style already in that file:
`dyn_cast<RankedTensorType>`, `emitOpError`, and two-space indent.

Check rank, element type, and shape relationships. Reject unranked tensors
when the operation requires ranks. Put each invalid case in
`tests/lit/Dialect/Kernel/my_kernel-invalid.mlir`:

```mlir
// RUN: ks-opt %s -split-input-file -verify-diagnostics

func.func @my_kernel_unranked(%input: tensor<*xf32>) -> tensor<*xf32> {
  // expected-error @+1 {{input must be a ranked tensor}}
  %0 = ks.my_kernel %input : tensor<*xf32> -> tensor<*xf32>
  return %0 : tensor<*xf32>
}
```

`ks.relu`, `ks.gelu`, and `ks.silu` are the exceptions that still have no
verifier. New operations should have one.

## 4. Lower it

Keep the pipeline `ks` to linalg, then tiled loops, vector, and the target.
Add a rewrite pattern to the pass that already owns that stage:

| Stage | Pass flag | File |
|---|---|---|
| Activations | `--ks-lower-activations` | `lib/Passes/LowerActivationsPass.cpp` |
| Structured and quantized ops | `--ks-lower-to-linalg` | `lib/Passes/LowerToLinalgPass.cpp` |
| Matmul tiling | `--ks-tile` | `lib/Passes/TilePass.cpp` |
| B packing | `--ks-pack` | `lib/Passes/PackPass.cpp` |
| Vector form | `--ks-vectorize` | `lib/Passes/VectorizePass.cpp` |
| RISC-V RVV | `--ks-lower-to-rvv` | `lib/Passes/LowerToRVVPass.cpp` |

A new pass is warranted when the rewrite is a separate pipeline stage. Scaffold
it with `./scripts/new-pass.sh`, declare it in
`include/KernelSmith/Passes/Passes.td`, and add the `.cpp` file to
`lib/Passes/CMakeLists.txt`. The pass flag must start with `ks-`.

Put the FileCheck test in `tests/lit/Passes/`, next to the existing lowering
tests:

```mlir
// RUN: ks-opt %s --ks-lower-to-linalg | FileCheck %s

// CHECK-LABEL: func @test_my_kernel
// CHECK-NOT: ks.my_kernel
// CHECK: linalg.generic
func.func @test_my_kernel(%input: tensor<32xf32>) -> tensor<32xf32> {
  %0 = ks.my_kernel %input : tensor<32xf32> -> tensor<32xf32>
  return %0 : tensor<32xf32>
}
```

## 5. Expose a C API when the kernel is public

Public headers go in `include/kernelsmith/`. Reference C goes in
`lib/kernelsmith/`. Keep the header C99, return `KS_OK` or a `KS_ERR_*` code,
and avoid allocating inside the kernel. Add a smoke test under `tests/capi/`
and a golden case under `tests/golden/cases/` when the numerical result needs
a NumPy reference.

Generated RVV objects replace a reference kernel behind the same symbol. The
INT8 hook is `KS_INT8_RVV_OBJECTS` in `lib/kernelsmith/CMakeLists.txt`.

## 6. Update the docs that name kernels

- The kernel table in `README.md`
- The relevant spec, if the implementation changed a contract
- `tasks/MILESTONES.md`, when the work closes a milestone item

## Checklist

- Spec describes the math, types, and error cases
- TableGen operation uses the `ks.` prefix
- Verifier tests fail the invalid cases
- Parse/print lit test round-trips
- Lowering test shows the expected downstream ops
- `ctest --test-dir build --output-on-failure` passes
- README kernel table matches the new operation
