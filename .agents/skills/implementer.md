# Implementer Agent Skills

## Role

The Implementer Agent writes code following Test-Driven Development (TDD) practices, creating well-tested, maintainable implementations that satisfy approved designs.

## Core Competencies

### 1. Test-Driven Development
- Write failing tests before implementation
- Implement minimal code to pass tests
- Refactor while maintaining green tests

### 2. MLIR Development
- TableGen operation definitions
- C++ dialect implementation
- Pass development
- Pattern rewriting

### 3. Clean Code
- Follow LLVM coding standards
- Write self-documenting code
- Keep functions small and focused

### 4. Incremental Development
- Make small, focused commits
- Each commit passes all tests
- Clear commit messages

## TDD Cycle

```
┌─────────────────┐
│   RED           │ Write a failing test
│   (Test First)  │ that defines expected behavior
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   GREEN         │ Write minimal code
│   (Make Pass)   │ to make the test pass
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   REFACTOR      │ Improve code quality
│   (Clean Up)    │ while keeping tests green
└────────┬────────┘
         │
         └──────────→ (Repeat)
```

## Implementation Workflow

### 1. Start with Tests

```mlir
// tests/lit/Dialect/Kernel/matmul.mlir
// RUN: ks-opt %s | FileCheck %s

// CHECK-LABEL: func @test_matmul
func.func @test_matmul(%A: tensor<64x128xf32>, %B: tensor<128x256xf32>) 
    -> tensor<64x256xf32> {
  // CHECK: ks.matmul
  %C = ks.matmul %A, %B : tensor<64x128xf32>, tensor<128x256xf32> 
                          -> tensor<64x256xf32>
  return %C : tensor<64x256xf32>
}
```

### 2. Minimal Implementation

```cpp
// Just enough to make the test pass
LogicalResult MatmulOp::verify() {
  // Basic verification only
  return success();
}
```

### 3. Add More Tests, Extend Implementation

```mlir
// tests/lit/Dialect/Kernel/matmul-invalid.mlir
// RUN: ks-opt %s -split-input-file -verify-diagnostics

func.func @test_matmul_rank_error(%A: tensor<64xf32>, %B: tensor<128x256xf32>) {
  // expected-error @+1 {{left operand must be a 2D tensor}}
  %C = ks.matmul %A, %B : tensor<64xf32>, tensor<128x256xf32> 
                          -> tensor<64x256xf32>
}
```

### 4. Refactor

- Extract common patterns
- Improve naming
- Add documentation
- Keep tests passing

## Commit Guidelines

### Atomic Commits

Each commit should:
- Address one logical change
- Pass all tests
- Be revertible independently

### Commit Message Format

```
<type>(<scope>): <subject>

<body explaining what and why>

Tests: describe test coverage
Design: link to design doc if applicable
```

### Example

```
feat(dialect): add ks.matmul operation

Implements matrix multiplication operation with:
- 2D tensor inputs (M×K, K×N)
- Shape verification
- Element type checking

Tests: tests/lit/Dialect/Kernel/matmul.mlir
Design: docs/design/DES-001-matmul.md
```

## Checklist Before Commit

- [ ] All tests pass: `ctest --test-dir build`
- [ ] New code has tests
- [ ] No compiler warnings
- [ ] Code formatted: `make format`
- [ ] Commit message is descriptive
