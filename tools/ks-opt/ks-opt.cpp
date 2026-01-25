//===----------------------------------------------------------------------===//
// ks-opt - KernelSmith Optimizer Driver
//===----------------------------------------------------------------------===//

#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

namespace kernelsmith {
void registerAllPasses();
}

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  
  // Register standard MLIR dialects
  mlir::registerAllDialects(registry);
  
  // Register KernelSmith dialect
  registry.insert<kernelsmith::ks::KSDialect>();
  
  // Register all passes
  mlir::registerAllPasses();
  kernelsmith::registerAllPasses();

  return mlir::asMainReturnCode(
      mlir::MlirOptMain(argc, argv, "KernelSmith optimizer driver\n", registry));
}
