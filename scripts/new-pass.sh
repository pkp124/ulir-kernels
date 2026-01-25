#!/bin/bash
# ==============================================================================
# Create a new MLIR pass from template
# ==============================================================================

set -e

NAME=$1

if [ -z "$NAME" ]; then
    echo "Usage: $0 <pass_name>"
    exit 1
fi

# Convert to appropriate cases
UPPER_NAME=$(echo "$NAME" | tr '[:lower:]' '[:upper:]' | tr '-' '_')
LOWER_NAME=$(echo "$NAME" | tr '[:upper:]' '[:lower:]' | tr '_' '-')
CAMEL_NAME=$(echo "$NAME" | sed -r 's/(^|[-_])([a-z])/\U\2/g')

echo "Creating pass: ${CAMEL_NAME}Pass"

# Create pass implementation file
PASS_DIR="src/passes"
mkdir -p "$PASS_DIR"

PASS_FILE="${PASS_DIR}/${CAMEL_NAME}.cpp"
if [ ! -f "$PASS_FILE" ]; then
    cat > "$PASS_FILE" << EOF
//===----------------------------------------------------------------------===//
// ${CAMEL_NAME} Pass Implementation
//===----------------------------------------------------------------------===//

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "AIKernels/Dialect/Kernel/KernelDialect.h"

namespace aikernel {

#define GEN_PASS_DEF_${UPPER_NAME}
#include "AIKernels/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

// TODO: Add rewrite patterns here

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct ${CAMEL_NAME}Pass 
    : public impl::${CAMEL_NAME}Base<${CAMEL_NAME}Pass> {
  
  void runOnOperation() override {
    auto *context = &getContext();
    RewritePatternSet patterns(context);
    
    // TODO: Add patterns
    // patterns.add<MyPattern>(context);
    
    if (failed(applyPatternsAndFoldGreedily(getOperation(), 
                                            std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> create${CAMEL_NAME}Pass() {
  return std::make_unique<${CAMEL_NAME}Pass>();
}

} // namespace aikernel
EOF
    echo "Created: $PASS_FILE"
fi

# Create test file
TEST_FILE="tests/lit/Transforms/${LOWER_NAME}.mlir"
mkdir -p "$(dirname "$TEST_FILE")"
if [ ! -f "$TEST_FILE" ]; then
    cat > "$TEST_FILE" << EOF
// RUN: aikernel-opt %s --${LOWER_NAME} | FileCheck %s

// CHECK-LABEL: func @test_${LOWER_NAME//-/_}
func.func @test_${LOWER_NAME//-/_}(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // TODO: Add test case
  // CHECK: expected.output
  return %arg0 : tensor<32x32xf32>
}
EOF
    echo "Created: $TEST_FILE"
fi

echo ""
echo "Pass scaffolding created!"
echo ""
echo "Next steps:"
echo "  1. Add pass declaration to src/passes/Passes.td"
echo "  2. Implement patterns in $PASS_FILE"
echo "  3. Register pass in PassRegistration.cpp"
echo "  4. Run tests: make test-lit"
