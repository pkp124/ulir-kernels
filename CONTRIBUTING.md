# Contributing to KernelSmith

## Prerequisites

- LLVM/MLIR 21+
- CMake 3.20+
- Ninja
- Python 3.10+
- C++17 compiler
- Optional but recommended: Docker for container parity checks

## First-time setup

```bash
./scripts/setup.sh
```

This installs/checks system dependencies, creates `.venv`, configures CMake,
builds the project, and runs CTest.

## Development loop

```bash
source .venv/bin/activate
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Or use the Makefile shims:

```bash
make build
make test
make verify
```

## Required process

1. Read the relevant spec under `specs/`.
2. Check `docs/design/` for prior decisions.
3. Add or update a design doc for significant architecture/pass/API changes.
4. Write focused tests first, especially lit FileCheck tests for MLIR behavior.
5. Implement the smallest correct change.
6. Run tests before committing.

## Verification commands

| Purpose | Command |
|---|---|
| Full native setup/build/test | `./scripts/setup.sh` |
| Rebuild | `cmake --build build --parallel` |
| All CTest tests | `ctest --test-dir build --output-on-failure` |
| Lit tests | `cmake --build build --target check-kernelsmith-lit` |
| Python lint/format | `ruff check . && ruff format --check .` |
| Container parity | `./scripts/docker-verify.sh` |
| RVV simulator setup | `./scripts/setup-rvv-sim.sh` |

## Commit style

Use conventional project commit prefixes:

```text
feat(scope): subject
fix(scope): subject
docs(scope): subject
refactor(scope): subject
test(scope): subject
chore(scope): subject
```

Keep commits focused. Include tests and docs in the same logical change when they
are required to understand or validate the behavior.
