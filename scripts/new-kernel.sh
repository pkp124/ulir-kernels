#!/bin/bash
# ==============================================================================
# Create a new kernel operation from template
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

echo "Creating kernel: ks.${LOWER_NAME}"

# Create specification file
SPEC_FILE="specs/kernels/${LOWER_NAME}.md"
if [ ! -f "$SPEC_FILE" ]; then
    mkdir -p "$(dirname "$SPEC_FILE")"
    cat > "$SPEC_FILE" << EOF
# ${CAMEL_NAME} Kernel Specification

## Overview

Brief description of the ks.${LOWER_NAME} kernel.

## Operation Definition

\`\`\`mlir
%output = ks.${LOWER_NAME} %input : tensor<...> -> tensor<...>
\`\`\`

## Mathematical Definition

\`\`\`
output = ${LOWER_NAME}(input)
\`\`\`

## Input/Output Specification

### Inputs
| Name | Type | Description |
|------|------|-------------|
| input | tensor | Input tensor |

### Outputs
| Name | Type | Description |
|------|------|-------------|
| output | tensor | Output tensor |

### Attributes
| Name | Type | Default | Description |
|------|------|---------|-------------|

## Verification Rules

1. Input must be a ranked tensor
2. Output shape matches input shape (if applicable)

## Lowering Strategy

1. ks.${LOWER_NAME} → linalg/arith operations
2. Vectorization
3. Target-specific lowering

## Test Cases

1. Basic functionality
2. Different shapes
3. Different element types
4. Edge cases
EOF
    echo "Created: $SPEC_FILE"
fi

# Create test file
TEST_FILE="tests/lit/Dialect/Kernel/${LOWER_NAME}.mlir"
mkdir -p "$(dirname "$TEST_FILE")"
if [ ! -f "$TEST_FILE" ]; then
    cat > "$TEST_FILE" << EOF
// RUN: ks-opt %s | FileCheck %s

// CHECK-LABEL: func @test_${LOWER_NAME}_basic
func.func @test_${LOWER_NAME}_basic(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: ks.${LOWER_NAME}
  %0 = ks.${LOWER_NAME} %arg0 : tensor<32x32xf32> -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}
EOF
    echo "Created: $TEST_FILE"
fi

# Create task
TASK_FILE="tasks/KERNEL-${UPPER_NAME}.md"
if [ ! -f "$TASK_FILE" ]; then
    cat > "$TASK_FILE" << EOF
# KERNEL-${UPPER_NAME}: Implement ks.${LOWER_NAME} Operation

## Status
[ ] Not Started

## Priority
P2 (Medium)

## Description

Implement the ks.${LOWER_NAME} kernel operation.

## Acceptance Criteria

- [ ] Operation defined in KernelOps.td
- [ ] Verifier implemented
- [ ] Parse/print round-trip works
- [ ] Lit tests passing
- [ ] Lowering to linalg implemented
- [ ] Documentation updated

## Specification

See: specs/kernels/${LOWER_NAME}.md

## Design Document

Create: docs/design/DES-XXX-${LOWER_NAME}.md

## Verification

\`\`\`bash
ctest --test-dir build -R "${LOWER_NAME}"
\`\`\`
EOF
    echo "Created: $TASK_FILE"
fi

echo ""
echo "Kernel scaffolding created for ks.${LOWER_NAME}"
echo ""
echo "Next steps:"
echo "  1. Review and complete spec: $SPEC_FILE"
echo "  2. Create design doc: make new-design ID=XXX TITLE=\"${CAMEL_NAME} Operation\""
echo "  3. Get design approved"
echo "  4. Implement using TDD"
