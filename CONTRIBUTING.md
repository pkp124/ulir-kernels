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

## Where to look first

| Question | Document |
|---|---|
| What can I call or lower today? | [README.md](README.md) |
| How do I build and run an example? | [docs/guides/getting-started.md](docs/guides/getting-started.md) |
| How do I add an operation? | [docs/guides/adding-kernels.md](docs/guides/adding-kernels.md) |
| How do I run tests? | [docs/guides/testing-guide.md](docs/guides/testing-guide.md) |
| What milestone is active? | [tasks/MILESTONES.md](tasks/MILESTONES.md) |
| What is the product sequence? | [ROADMAP.md](ROADMAP.md) |

Active work is M7, the llama.cpp host smoke. M2 and M3 passes exist; replacing
the handwritten matmul and activation objects is still open and is not the
current task.

## Development loop

```bash
source .venv/bin/activate
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Makefile targets: `make build`, `make test`, `make lit`, `make unit`,
`make capi`, `make lint`, and `make verify`. New kernels, passes, and design
docs are scripts, not make targets: `./scripts/new-kernel.sh`,
`./scripts/new-pass.sh`, and `./scripts/new-design.sh`.

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
