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

//===----------------------------------------------------------------------===//
// Milestone 3: MatMul Lowering Passes
//===----------------------------------------------------------------------===//

def KSLowerToLinalgPass
    : Pass<"ks-lower-to-linalg", "mlir::func::FuncOp"> {
  let summary = "Lower KS matmul ops to linalg.matmul";
  let description = [{
    Lowers ks.matmul operations to linalg.matmul with tensor.empty
    and linalg.fill for zero-initialization.

    Supports mixed-precision via the optional acc_type attribute:
    when acc_type is specified, the matmul accumulates in the wider
    type and truncates the result back to the output element type.
  }];
  let dependentDialects = [
    "mlir::linalg::LinalgDialect",
    "mlir::arith::ArithDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KSTilePass
    : Pass<"ks-tile", "mlir::func::FuncOp"> {
  let summary = "Tile linalg operations using SCF for loops";
  let description = [{
    Applies single-level tiling to linalg.matmul (and other TilingInterface
    ops) using scf.for loops. Tile sizes are profile-driven and passed as
    options.

    A tile size of 0 means "do not tile that dimension".
  }];
  let options = [
    Option<"tileM", "tile-m", "int64_t", "64",
           "Tile size for M dimension (0 = no tiling)">,
    Option<"tileN", "tile-n", "int64_t", "64",
           "Tile size for N dimension (0 = no tiling)">,
    Option<"tileK", "tile-k", "int64_t", "32",
           "Tile size for K dimension (0 = no tiling)">
  ];
  let dependentDialects = [
    "mlir::scf::SCFDialect",
    "mlir::linalg::LinalgDialect",
    "mlir::tensor::TensorDialect"
  ];
}

def KSAllocCheckPass
    : Pass<"ks-alloc-check", "mlir::func::FuncOp"> {
  let summary = "Reject stray memref.alloc operations";
  let description = [{
    Verifies that no memref.alloc operations survive after bufferization.
    All buffers must be function arguments — zero internal malloc.
    Emits an error for each memref.alloc found.
  }];
  let dependentDialects = [
    "mlir::memref::MemRefDialect"
  ];
}

#endif // KERNELSMITH_PASSES_TD
