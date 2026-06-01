//===----------------------------------------------------------------------===//
// ks-opt - KernelSmith Optimizer Driver
//===----------------------------------------------------------------------===//

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Vector/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

namespace kernelsmith {
void registerAllPasses();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;

  // Register standard MLIR dialects
  mlir::registerAllDialects(registry);

  // Register KernelSmith dialect.
  registry.insert<kernelsmith::ks::KSDialect>();

  // One-shot bufferization relies on external models for upstream dialects.
  mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
  mlir::vector::registerBufferizableOpInterfaceExternalModels(registry);

  // Register all passes
  mlir::registerAllPasses();
  kernelsmith::registerAllPasses();

  return mlir::asMainReturnCode(mlir::MlirOptMain(
      argc, argv, "KernelSmith optimizer driver\n", registry));
}
