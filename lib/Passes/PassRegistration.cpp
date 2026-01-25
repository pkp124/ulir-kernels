//===----------------------------------------------------------------------===//
// KernelSmith Pass Registration
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

namespace kernelsmith {

void registerAllPasses() {
  // TODO: Register passes as implemented
  // mlir::registerPass([]() { return createKSLowerToLinalgPass(); });
}

} // namespace kernelsmith
