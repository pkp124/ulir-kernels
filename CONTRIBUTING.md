# Contributing to KernelSmith

Thank you for your interest in contributing! This document provides guidelines and workflows for contributing to KernelSmith.

## Development Philosophy

KernelSmith follows a rigorous software engineering process:

1. **Specification-Driven**: Define behavior before implementing
2. **Design Reviews**: All significant changes require design review
3. **Test-Driven Development**: Write tests before implementation
4. **Documentation**: Document decisions and rationale

## Development Workflow

### 1. Specification Phase

Before implementing any feature:

1. Check `specs/` for existing specification
2. If none exists, create one following the template
3. Get specification reviewed and approved

### 2. Design Phase

For non-trivial changes:

1. Create a design document in `docs/design/`
2. Document alternatives considered
3. Request design review
4. Address feedback before implementing

### 3. Test-Driven Development

1. Write failing tests first
2. Implement minimal code to pass tests
3. Refactor while keeping tests green
4. Add edge case tests

### 4. Implementation

1. Make small, focused commits
2. Each commit should pass all tests
3. Use descriptive commit messages

### 5. Review and Merge

1. Ensure all tests pass: `ctest --test-dir build`
2. Update documentation if needed
3. Create PR with clear description

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Build/tooling changes

### Example

```
feat(dialect): add ks.matmul operation

Implements the matmul operation with:
- Shape verification for 2D tensors
- Inner dimension compatibility check
- Canonicalization patterns

Design: docs/design/DES-001-matmul.md
Spec: specs/kernels/matmul.md
```

## Code Style

### C++ (LLVM Style)
- Follow LLVM Coding Standards
- Use `clang-format` with provided config
- Run `cmake --build build --target format`

### TableGen
- One operation per logical group
- Include summary and description
- Use consistent naming: `Dialect_VerbNounOp`

## Testing Requirements

| Change Type | Required Tests |
|-------------|----------------|
| New operation | Lit tests (parse, verify), unit tests |
| New pass | Lit tests (transformation), edge cases |
| Bug fix | Regression test |

### Running Tests

```bash
ctest --test-dir build                    # All tests
ctest --test-dir build -R "Dialect"       # Dialect tests
ctest --test-dir build --output-on-failure
```

## Design Review Process

1. Create design document from template
2. Tag with `[DESIGN-REVIEW]` in PR
3. Reviewers check:
   - Requirements coverage
   - Alternative analysis
   - Risk assessment
   - Test strategy
4. Address all feedback
5. Get explicit approval before implementing

## Getting Help

- Check documentation in `docs/`
- Look at similar implementations
- Create an issue for questions
