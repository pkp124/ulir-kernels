#!/bin/bash
# ==============================================================================
# Create a new kernel from template
# ==============================================================================

set -e

NAME=$1

if [ -z "$NAME" ]; then
    echo "Usage: $0 <kernel_name>"
    exit 1
fi

# Convert to appropriate cases
UPPER_NAME=$(echo "$NAME" | tr '[:lower:]' '[:upper:]')
LOWER_NAME=$(echo "$NAME" | tr '[:upper:]' '[:lower:]')
CAMEL_NAME=$(echo "$NAME" | sed -r 's/(^|_)([a-z])/\U\2/g')

echo "Creating kernel: $CAMEL_NAME"

# Create specification file
SPEC_FILE="specs/kernels/${LOWER_NAME}.md"
if [ ! -f "$SPEC_FILE" ]; then
    cat > "$SPEC_FILE" << EOF
# ${CAMEL_NAME} Kernel Specification

## Overview

Brief description of the ${CAMEL_NAME} kernel.

## Mathematical Definition

\`\`\`
output = ${LOWER_NAME}(inputs...)
\`\`\`

## Input/Output Specification

### Inputs
- \`input\`: Description (tensor type, shape constraints)

### Outputs
- \`output\`: Description (tensor type, shape constraints)

### Attributes
- \`attr1\`: Description (type, default value)

## Tiling Strategy

Describe how this kernel should be tiled for efficient execution.

## Lowering Path

1. kernel.${LOWER_NAME} → linalg operations
2. linalg → vector operations
3. vector → target-specific (RVV, etc.)

## Test Cases

1. Basic functionality
2. Edge cases (empty inputs, single elements)
3. Numerical accuracy
4. Performance benchmarks

## References

- Paper/documentation links
EOF
    echo "Created: $SPEC_FILE"
fi

# Create test file
TEST_FILE="tests/lit/Dialect/Kernel/${LOWER_NAME}.mlir"
mkdir -p "$(dirname "$TEST_FILE")"
if [ ! -f "$TEST_FILE" ]; then
    cat > "$TEST_FILE" << EOF
// RUN: aikernel-opt %s | FileCheck %s

// CHECK-LABEL: func @test_${LOWER_NAME}_basic
func.func @test_${LOWER_NAME}_basic(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: kernel.${LOWER_NAME}
  %0 = kernel.${LOWER_NAME} %arg0 : tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
EOF
    echo "Created: $TEST_FILE"
fi

# Create task for implementation
TASK_FILE="tasks/KERNEL-${UPPER_NAME}.md"
if [ ! -f "$TASK_FILE" ]; then
    cat > "$TASK_FILE" << EOF
# KERNEL-${UPPER_NAME}: Implement ${CAMEL_NAME} Kernel

## Status
[ ] Not Started

## Priority
P2 (Medium)

## Description

Implement the ${CAMEL_NAME} kernel operation including:
- TableGen operation definition
- Verifier
- Lowering to linalg/vector
- Target-specific lowering (RVV)

## Acceptance Criteria

- [ ] Operation defined in KernelOps.td
- [ ] Verifier implemented and tested
- [ ] Parsing/printing works correctly
- [ ] Lowering pass to linalg implemented
- [ ] Unit tests passing
- [ ] Lit tests passing
- [ ] Documentation updated

## Specification

See: specs/kernels/${LOWER_NAME}.md

## Dependencies

- TASK-001: Kernel dialect infrastructure

## Verification

\`\`\`bash
make test-lit TESTS=tests/lit/Dialect/Kernel/${LOWER_NAME}.mlir
\`\`\`
EOF
    echo "Created: $TASK_FILE"
fi

echo ""
echo "Kernel scaffolding created!"
echo ""
echo "Next steps:"
echo "  1. Review and update spec: $SPEC_FILE"
echo "  2. Add operation to src/dialects/kernel/KernelOps.td"
echo "  3. Implement verifier and lowering"
echo "  4. Run tests: make test-lit"
