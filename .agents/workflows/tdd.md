# Test-Driven Development Workflow

This document describes the TDD workflow for KernelSmith development.

## TDD Philosophy

> "Write a failing test before writing any production code."

TDD ensures:
- Requirements are testable
- Code is correct by construction
- Refactoring is safe
- Documentation through tests

## The TDD Cycle

```
┌─────────────────────────────────────────────────────────────────┐
│                         RED                                      │
│  Write a test that fails because the feature doesn't exist      │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                        GREEN                                     │
│  Write the minimum code necessary to make the test pass         │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       REFACTOR                                   │
│  Improve code quality while keeping all tests passing           │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 └────────→ Repeat
```

## TDD for MLIR Operations

### Step 1: Write Parsing Test (RED)

```mlir
// tests/lit/Dialect/Kernel/relu.mlir
// RUN: ks-opt %s | FileCheck %s

// CHECK-LABEL: func @test_relu
func.func @test_relu(%input: tensor<32xf32>) -> tensor<32xf32> {
  // CHECK: ks.relu
  %output = ks.relu %input : tensor<32xf32>
  return %output : tensor<32xf32>
}
```

Run test (should fail - operation doesn't exist):
```bash
ctest --test-dir build -R "relu" --output-on-failure
```

### Step 2: Minimal Implementation (GREEN)

Add operation to TableGen:
```tablegen
def KS_ReLUOp : KS_Op<"relu", [Pure, SameOperandsAndResultType]> {
  let summary = "ReLU activation";
  let arguments = (ins AnyTensor:$input);
  let results = (outs AnyTensor:$output);
  let assemblyFormat = "$input attr-dict `:` type($input)";
}
```

Run test (should pass):
```bash
ctest --test-dir build -R "relu"
```

### Step 3: Add Verifier Test (RED)

```mlir
// tests/lit/Dialect/Kernel/relu-invalid.mlir
// RUN: ks-opt %s -split-input-file -verify-diagnostics

func.func @test_relu_not_tensor(%input: f32) {
  // expected-error @+1 {{operand must be a tensor}}
  %output = ks.relu %input : f32
}
```

### Step 4: Implement Verifier (GREEN)

```cpp
LogicalResult ReLUOp::verify() {
  if (!getInput().getType().isa<TensorType>())
    return emitOpError("operand must be a tensor");
  return success();
}
```

### Step 5: Refactor

- Extract common patterns
- Improve error messages
- Add documentation

Keep running tests to ensure they stay green.

## TDD for Passes

### Step 1: Write Transformation Test (RED)

```mlir
// tests/lit/Passes/lower-relu.mlir
// RUN: ks-opt %s --ks-lower-to-linalg | FileCheck %s

// CHECK-LABEL: func @test_lower_relu
func.func @test_lower_relu(%input: tensor<32xf32>) -> tensor<32xf32> {
  // CHECK-NOT: ks.relu
  // CHECK: arith.maxf
  // CHECK-SAME: %{{.*}}, %{{.*}}
  %output = ks.relu %input : tensor<32xf32>
  return %output : tensor<32xf32>
}
```

### Step 2: Implement Pass (GREEN)

```cpp
struct LowerReLUPattern : public OpRewritePattern<ks::ReLUOp> {
  LogicalResult matchAndRewrite(ks::ReLUOp op,
                                PatternRewriter &rewriter) const override {
    auto zero = rewriter.create<arith::ConstantOp>(...);
    auto max = rewriter.create<arith::MaxFOp>(op.getInput(), zero);
    rewriter.replaceOp(op, max);
    return success();
  }
};
```

### Step 3: Add Edge Cases (RED → GREEN)

```mlir
// Test with different types
func.func @test_lower_relu_f16(%input: tensor<32xf16>) -> tensor<32xf16> {
  %output = ks.relu %input : tensor<32xf16>
  return %output : tensor<32xf16>
}

// Test with different shapes
func.func @test_lower_relu_2d(%input: tensor<32x64xf32>) -> tensor<32x64xf32> {
  %output = ks.relu %input : tensor<32x64xf32>
  return %output : tensor<32x64xf32>
}
```

## Best Practices

### Write Small Tests
- One behavior per test
- Clear test names
- Fast execution

### Test Behavior, Not Implementation
- Focus on what, not how
- Tests should survive refactoring

### Use Descriptive Names
```mlir
// Good: describes what's being tested
func.func @test_matmul_rejects_1d_input(...)

// Bad: doesn't describe the test
func.func @test1(...)
```

### Test Error Cases
```mlir
// RUN: ks-opt %s -split-input-file -verify-diagnostics

func.func @test_error_case() {
  // expected-error @+1 {{specific error message}}
  ...
}
```

## Common Pitfalls

### Writing Tests After Code
- Loses TDD benefits
- Tests become confirmation, not specification

### Testing Too Much at Once
- Hard to debug failures
- Tests become brittle

### Not Running Tests Frequently
- Errors accumulate
- Harder to identify cause

## Commands Reference

```bash
# Run all tests
ctest --test-dir build

# Run specific test
ctest --test-dir build -R "relu"

# Verbose output
ctest --test-dir build -R "relu" --verbose

# Re-run failed tests
ctest --test-dir build --rerun-failed

# Show output on failure
ctest --test-dir build --output-on-failure
```
