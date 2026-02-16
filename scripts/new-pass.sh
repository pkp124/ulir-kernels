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

echo "Creating pass: KS${CAMEL_NAME}Pass"

# Create pass implementation file
PASS_DIR="lib/Passes"
mkdir -p "$PASS_DIR"

PASS_FILE="${PASS_DIR}/${CAMEL_NAME}Pass.cpp"
if [ ! -f "$PASS_FILE" ]; then
    cat > "$PASS_FILE" << EOF
//===----------------------------------------------------------------------===//
// KS${CAMEL_NAME}Pass Implementation
//===----------------------------------------------------------------------===//

#include "KernelSmith/Passes/Passes.h"

#include "KernelSmith/Dialect/Kernel/KernelDialect.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace kernelsmith::ks {

#define GEN_PASS_DEF_KS${UPPER_NAME}
#include "KernelSmith/Passes/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// Rewrite Patterns
//===----------------------------------------------------------------------===//

// TODO: Add rewrite patterns here

//===----------------------------------------------------------------------===//
// Pass Implementation
//===----------------------------------------------------------------------===//

struct KS${CAMEL_NAME}Pass
    : public impl::KS${CAMEL_NAME}Base<KS${CAMEL_NAME}Pass> {

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

} // namespace kernelsmith::ks
EOF
    echo "Created: $PASS_FILE"
fi

# Create test file
TEST_FILE="tests/lit/${LOWER_NAME}.mlir"
mkdir -p "$(dirname "$TEST_FILE")"
if [ ! -f "$TEST_FILE" ]; then
    cat > "$TEST_FILE" << EOF
// RUN: ks-opt %s --ks-${LOWER_NAME} | FileCheck %s

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
echo "  1. Add pass declaration to include/KernelSmith/Passes/Passes.td"
echo "  2. Implement patterns in $PASS_FILE"
echo "  3. Add $PASS_FILE to lib/Passes/CMakeLists.txt"
echo "  4. Run tests: ctest --test-dir build --output-on-failure"
