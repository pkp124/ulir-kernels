# Contributing to MLIR Kernel Generation

Thank you for your interest in contributing! This document provides guidelines and workflows for contributing to the project.

## Development Setup

1. **Fork and clone** the repository
2. **Run setup**: `make setup`
3. **Create a branch**: `git checkout -b feature/your-feature`

## Development Workflow

### 1. Specification First

Before implementing a feature:

1. Check if a specification exists in `specs/`
2. If not, create one using the templates
3. Get feedback on the spec before implementing

### 2. Test-Driven Development

1. Write tests first (or use `make new-kernel` to scaffold)
2. Implement until tests pass
3. Add edge case tests

### 3. Incremental Commits

- Make small, focused commits
- Each commit should pass `make verify`
- Use descriptive commit messages

## Commit Message Format

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Types
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `refactor`: Code change that neither fixes a bug nor adds a feature
- `test`: Adding missing tests
- `chore`: Changes to build process or auxiliary tools

### Examples

```
feat(kernel): add scaled dot-product attention operation

Implements kernel.scaled_dot_product_attention with support for
optional attention mask. Includes verifier and basic lowering
to linalg operations.

Closes #123
```

```
fix(rvv): correct vector length calculation in matmul

The previous implementation didn't account for LMUL when
calculating the effective vector length, causing incorrect
results with LMUL > 1.
```

## Code Style

### C++ (LLVM Style)

- Follow [LLVM Coding Standards](https://llvm.org/docs/CodingStandards.html)
- Use `clang-format` with the provided `.clang-format`
- Run `make format-cpp` before committing

### Python (PEP 8 + Type Hints)

- Use type hints for all function signatures
- Use `ruff` for linting and formatting
- Run `make format-python` before committing

### TableGen

- One operation per logical group
- Include summary and description for all operations
- Use consistent naming: `Dialect_VerbNounOp`

## Adding New Kernels

Use the scaffolding script:

```bash
make new-kernel NAME=my_kernel
```

This creates:
- `specs/kernels/my_kernel.md` - Specification
- `tests/lit/Dialect/Kernel/my_kernel.mlir` - Test file
- `tasks/KERNEL-MY_KERNEL.md` - Task tracking

Then:
1. Complete the specification
2. Add operation to `src/dialects/kernel/KernelOps.td`
3. Implement verifier
4. Implement lowering passes
5. Run tests: `make test-lit`

## Adding New Passes

Use the scaffolding script:

```bash
make new-pass NAME=tile-convolution
```

This creates:
- `src/passes/TileConvolution.cpp` - Pass implementation
- `tests/lit/Transforms/tile-convolution.mlir` - Test file

Then:
1. Add pass declaration to `src/passes/Passes.td`
2. Implement patterns in the `.cpp` file
3. Register the pass
4. Run tests

## Testing Requirements

All contributions must include appropriate tests:

| Change Type | Required Tests |
|-------------|----------------|
| New kernel operation | Lit test (parsing, verifier), unit test |
| New pass | Lit test (transformation), edge cases |
| Bug fix | Regression test that would have caught the bug |
| Performance change | Benchmark comparison |

### Running Tests

```bash
make test           # All tests
make test-unit      # Unit tests only
make test-lit       # MLIR lit tests only
make verify         # Full verification
```

## Pull Request Process

1. **Ensure all tests pass**: `make verify`
2. **Update documentation** if needed
3. **Create PR** with clear description
4. **Link related issues** in the PR description
5. **Respond to feedback** promptly

### PR Checklist

- [ ] Tests pass (`make verify`)
- [ ] Code follows style guidelines
- [ ] Documentation updated
- [ ] Commit messages follow format
- [ ] Specification updated (if applicable)

## Task Tracking

For larger features, create a task file:

```bash
make new-task ID=001 TITLE="Implement RVV matmul lowering"
```

Update the task file as you progress to help with handoffs between development sessions.

## Getting Help

- Check existing documentation in `docs/`
- Look at similar implementations in the codebase
- Create an issue for questions or discussions

## Code of Conduct

Be respectful and constructive. Focus on the code, not the person.
