//===----------------------------------------------------------------------===//
// KernelSmith Pass Declarations
//===----------------------------------------------------------------------===//

#ifndef KERNELSMITH_PASSES_H
#define KERNELSMITH_PASSES_H

#include "mlir/Pass/Pass.h"

#include <memory>

namespace kernelsmith {

// Generate pass declarations from TableGen.
#define GEN_PASS_DECL
#include "KernelSmith/Passes/Passes.h.inc"

// Generate pass registration.
#define GEN_PASS_REGISTRATION
#include "KernelSmith/Passes/Passes.h.inc"

} // namespace kernelsmith

#endif // KERNELSMITH_PASSES_H
