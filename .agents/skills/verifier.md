# Verifier Agent Skills

## Role

The Verifier Agent ensures implementation correctness through comprehensive testing, specification compliance checking, and quality verification.

## Core Competencies

### 1. Test Execution
- Run full test suite
- Analyze test failures
- Ensure coverage

### 2. Specification Compliance
- Verify implementation matches spec
- Check all requirements addressed
- Validate edge cases

### 3. Quality Assessment
- Check code quality metrics
- Verify documentation
- Assess maintainability

### 4. Regression Prevention
- Ensure no existing tests broken
- Add regression tests for bugs
- Monitor test stability

## Verification Workflow

### 1. Pre-Verification Checks

```bash
# Ensure clean build
make clean
make build

# Check for uncommitted changes
git status
```

### 2. Run Full Test Suite

```bash
# All tests via CTest
ctest --test-dir build --output-on-failure

# Verbose output for failures
ctest --test-dir build --rerun-failed --verbose
```

### 3. Test Categories

```bash
# Lit tests (MLIR FileCheck)
ctest --test-dir build -R "lit"

# Unit tests
ctest --test-dir build -R "unit"

# Target-specific tests
ctest --test-dir build -R "RISCV"
```

### 4. Coverage Analysis

Check that tests cover:
- [ ] All operations defined
- [ ] Valid inputs accepted
- [ ] Invalid inputs rejected with clear errors
- [ ] Edge cases (empty, single element, large)
- [ ] All code paths in passes

## Specification Compliance Checklist

For each requirement in the spec:

```markdown
## Verification: [Spec Name]

### REQ-1: [Requirement]
- [ ] Implemented
- [ ] Test exists: `tests/lit/...`
- [ ] Verified manually

### REQ-2: [Requirement]
...
```

## Quality Checks

### Code Quality
- [ ] No compiler warnings
- [ ] No linter errors: `make lint`
- [ ] Code formatted: `make format`

### Documentation
- [ ] Public APIs documented
- [ ] Design doc updated if needed
- [ ] README reflects current state

### Test Quality
- [ ] Tests are meaningful (not just coverage)
- [ ] Error messages tested
- [ ] Edge cases covered

## Verification Report Format

```markdown
# Verification Report: [Feature/PR]

## Summary
✅ All checks passed / ⚠️ Issues found

## Test Results
- Total: XX tests
- Passed: XX
- Failed: XX
- Skipped: XX

## Specification Compliance
| Requirement | Status | Test |
|-------------|--------|------|
| REQ-1 | ✅ | test_xxx.mlir |
| REQ-2 | ✅ | test_yyy.mlir |

## Quality Metrics
- Lint: ✅ Clean
- Format: ✅ Clean
- Warnings: ✅ None

## Issues Found
1. [Issue description]

## Recommendation
Approve / Request Changes
```

## Common Verification Scenarios

### New Operation
1. Parse/print round-trip works
2. Verifier catches all invalid cases
3. Lowering produces correct output
4. End-to-end pipeline works

### New Pass
1. Transformation correct on valid input
2. No crash on edge cases
3. Preserves semantics
4. Performance acceptable

### Bug Fix
1. Regression test added
2. Original issue resolved
3. No new failures introduced
