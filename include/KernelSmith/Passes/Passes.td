//===----------------------------------------------------------------------===//
// KernelSmith Pass Definitions (TableGen)
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_PASSES_TD
#define KERNELSMITH_PASSES_TD

include "mlir/Pass/PassBase.td"

def KSLowerActivationsPass
    : Pass<"ks-lower-activations", "mlir::func::FuncOp"> {
  let summary = "Lower KS activation ops to standard MLIR dialects";
  let description = [{
    Lowers ks.relu, ks.gelu, and ks.silu operations to linalg.generic
    with arith and math operations.

    Lowering rules:
      ks.relu  -> arith.maximumf(x, 0)
      ks.gelu  -> 0.5 * x * (1 + math.erf(x / sqrt(2)))
      ks.silu  -> x / (1 + math.exp(-x))
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KSLowerToLinalgPass
    : Pass<"ks-lower-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower KS structured ops to the linalg dialect";
  let description = [{
    Lowers KS dialect structured operations to their linalg named-op equivalents.
    This is the first step in the lowering pipeline for compute-intensive ops.

    Lowering rules:
      ks.matmul %A, %B -> linalg.fill(0) + linalg.matmul ins(%A, %B)
      ks.dot_i8 %x, %w -> linalg.generic with i8->i32 extension,
                           zero-point subtraction, and i32 accumulation
      ks.matvec_i8 %x, %W -> linalg.generic with parallel rows, K reduction,
                              zero-point subtraction, and i32 accumulation

    The output tensor is zero-initialized via linalg.fill so that the
    linalg.matmul and linalg.generic accumulation semantics produce replacement
    results.

    Dynamic shapes are handled by inserting tensor.dim queries for any
    dimension of the result tensor that is not statically known.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KSTilePass
    : Pass<"ks-tile", "mlir::func::FuncOp"> {
  let summary = "Tile linalg structured ops for cache efficiency";
  let description = [{
    Tiles linalg.matmul (and future structured ops) using the MLIR
    TilingInterface / scf::tileUsingSCF infrastructure. Generates nested
    scf.for loops with tile-sized tensor.extract_slice / tensor.insert_slice
    around the inner tiled op.

    Tile sizes are configurable via pass options. A tile size of 0 on any
    dimension means "do not tile that dimension" — no loop is emitted and
    the full extent is processed in one shot.

    Default tile sizes match the generic target profile (conservative,
    single-level, no SIMD). For RISC-V RVV targets (M4), this pass will be
    invoked with profile-driven values read from target/riscv_rvv_256.h.

    Example:
      ks-opt input.mlir --ks-lower-to-linalg
             "--ks-tile=tile-size-m=64 tile-size-n=64 tile-size-k=256"
  }];
  let options = [
    Option<"tileSizeM", "tile-size-m", "int64_t", "32",
           "Tile size for the M (rows of A / rows of C) dimension. "
           "0 means do not tile this dimension.">,
    Option<"tileSizeN", "tile-size-n", "int64_t", "32",
           "Tile size for the N (cols of B / cols of C) dimension. "
           "0 means do not tile this dimension.">,
    Option<"tileSizeK", "tile-size-k", "int64_t", "64",
           "Tile size for the K (reduction) dimension. "
           "0 means do not tile this dimension.">
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::scf::SCFDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::affine::AffineDialect"
  ];
}

def KSPackPass : Pass<"ks-pack", "mlir::func::FuncOp"> {
  let summary = "Pack B operand of linalg.matmul for cache-efficient access";
  let description = [{
    Packs the B (right-hand) operand of linalg.matmul into a column-panel
    layout that enables stride-free vector loads in the inner loop.

    The packing inserts linalg.pack for the B operand and updates the
    linalg.matmul to a linalg.generic that consumes the packed tensor while
    writing C in its original logical layout. The pack factor is configurable
    (default 32 — matches VLMAX for RVV VLEN=256/LMUL=4/f32).

    This pass should run after --ks-tile (L2 tiling) and before the second
    --ks-tile (register tiling) and --ks-vectorize.

    Example (pack-factor=32):
      linalg.matmul ins(%A : tensor<128x256xf32>, %B : tensor<256x128xf32>)
      ──►
      %Bp = linalg.pack %B inner_dims_pos=[1] inner_tiles=[32]
              : tensor<256x128xf32> -> tensor<4x256x32xf32>
      linalg.generic ins(%A, %Bp) outs(%C) ...
  }];
  let options = [
    Option<"packFactor", "pack-factor", "int64_t", "32",
           "Panel width for B packing (number of columns per panel). "
           "Should equal NR (register tile N dim). "
           "0 disables packing (no-op).">
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::affine::AffineDialect"
  ];
}

def KSMaterializePackWorkspacePass
    : Pass<"ks-materialize-pack-workspace", "mlir::func::FuncOp"> {
  let summary = "Materialize packed tensors into explicit workspace buffers";
  let description = [{
    Rewrites supported tensor-level linalg.pack operations into explicit
    memref linalg.generic copies backed by a caller-provided memref<?xi8>
    workspace argument. The packed memref is bridged back to tensor IR with
    bufferization.to_tensor so the remaining tensor pipeline can continue until
    one-shot bufferization.

    This pass is intentionally conservative: it supports the KernelSmith matmul
    B-panel layout [K, N] -> [N/NR, K, NR] with static shapes, inner_dims_pos=[1],
    outer_dims_perm=[1, 0], and a single positive static inner tile.
  }];
  let options = [
    Option<"workspaceAlignment", "workspace-alignment", "int64_t", "64",
           "Byte alignment used when carving packed buffers from workspace.">
  ];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::bufferization::BufferizationDialect",
    "mlir::memref::MemRefDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KSAllocCheckPass : Pass<"ks-alloc-check", "mlir::func::FuncOp"> {
  let summary = "Reject internal memref allocations in generated kernels";
  let description = [{
    Enforces KernelSmith's caller-provided workspace model by failing the
    pipeline if any memref.alloc survives lowering inside a function.

    This pass is intended to run after bufferization and before final LLVM
    conversion. Generated kernels must use function arguments or explicit
    workspace buffers instead of heap-like internal memref.alloc operations.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

def KSVectorizePass : Pass<"ks-vectorize", "mlir::func::FuncOp"> {
  let summary = "Vectorize inner linalg ops to the vector dialect";
  let description = [{
    Converts the innermost tiled linalg.matmul (and linalg.generic from
    activation lowering) to vector dialect operations using MLIR's
    linalg vectorization infrastructure.

    Must run after --ks-tile and --ks-pack. Produces vector.transfer_read,
    vector.transfer_write, vector.contract, and arith/math vector ops that
    are then lowered to RVV intrinsics by --ks-lower-to-rvv.

    The vector size is inferred from the tile sizes set by --ks-tile. For
    best results the inner tile N dimension should equal VLMAX of the target.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::vector::VectorDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect",
    "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def KSLowerToRVVPass : Pass<"ks-lower-to-rvv", "mlir::ModuleOp"> {
  let summary = "Lower vector dialect to LLVM IR targeting RISC-V RVV";
  let description = [{
    Applies the complete lowering pipeline from the vector/linalg/scf/tensor
    dialect level down to the LLVM dialect with RISC-V V extension semantics.

    Pipeline (in order):
      1. one-shot-bufferize           — tensor -> memref (function-boundary only)
      2. convert-linalg-to-loops      — any remaining linalg.generic -> scf.for
      3. lower-affine                  — affine.apply -> arith
      4. convert-scf-to-cf            — scf.for -> cf.br
      5. convert-vector-to-llvm       — vector.* -> llvm.*
      6. finalize-memref-to-llvm      — memref.* -> llvm.*
      7. convert-func-to-llvm         — func.func -> llvm.func
      8. reconcile-unrealized-casts   — clean up cast chains

    After this pass the IR is in LLVM dialect and can be translated to LLVM IR
    via mlir-translate and compiled to RVV assembly by llc:
      mlir-translate --mlir-to-llvmir out.mlir | \
      llc -march=riscv64 -mattr=+v,+zve64d -o out.s

    The target triple and features must be set externally (e.g. via llc flags
    or the compilation driver script scripts/compile-rvv.sh).
  }];
  let options = [
    Option<"indexBitwidth", "index-bitwidth", "unsigned", "64",
           "Bitwidth for index types in LLVM IR (32 for ILP32, 64 for LP64).">
  ];
  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "mlir::vector::VectorDialect",
    "mlir::memref::MemRefDialect",
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::func::FuncDialect"
  ];
}

#endif // KERNELSMITH_PASSES_TD
