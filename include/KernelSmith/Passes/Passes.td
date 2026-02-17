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

    The output tensor is zero-initialized via linalg.fill so that the
    linalg.matmul accumulation semantics (C += A*B) produce C = A*B.

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

#endif // KERNELSMITH_PASSES_TD
