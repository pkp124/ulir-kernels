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

#endif // KERNELSMITH_PASSES_TD
