# KernelSmith Agent Guide

This is the canonical entry point for AI agents and contributors working in this
repository. Keep this file aligned with `CLAUDE.md`, `.cursorrules`, and
`.cursor/rules/*` when infrastructure or workflow changes.

## Project snapshot

KernelSmith is an MLIR/LLVM 21+ compiler framework and C kernel library for
optimized ML inference kernels on edge targets. The primary backend is RISC-V
RVV; ARM NEON and quantized kernels are planned/ongoing.

Current high-level pipeline:

```text
ks ops
  -> --ks-lower-to-linalg
linalg ops
  -> --ks-tile
scf.for tiled loops
  -> --ks-pack / --ks-materialize-pack-workspace where applicable
packed/tiled linalg
  -> --ks-vectorize
vector dialect
  -> --ks-lower-to-rvv
LLVM dialect / LLVM IR / target object
```

## Non-negotiable workflow

1. Read the relevant spec in `specs/` before implementing behavior.
2. Check `docs/design/` before changing architecture. Add a design doc with
   `./scripts/new-design.sh` for significant new passes, target decisions, or
   public interfaces.
3. Use TDD for compiler behavior: write lit/unit tests before or alongside the
   implementation.
4. Preserve the layered lowering pipeline. Do not bypass `ks -> linalg -> vector
   -> target` without a design document.
5. Run at least `ctest --test-dir build --output-on-failure` before committing.
6. Prefer `./scripts/docker-verify.sh` before push when Docker is available; it
   checks container build/test plus C++ formatting.

## Development commands

Bootstrap a fresh machine:

```bash
./scripts/setup.sh
```

Rebuild/test after source changes:

```bash
source .venv/bin/activate
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

Container parity check:

```bash
./scripts/docker-verify.sh          # cached
./scripts/docker-verify.sh --no-cache
```

Useful aliases are also available through `make`:

```bash
make setup
make build
make test
make lit
make unit
make lint
make verify
```

## Infrastructure map

| Area | Files | Notes |
|---|---|---|
| Native setup | `scripts/setup.sh` | Installs LLVM/MLIR 21, creates `.venv`, builds, runs CTest. |
| Local rebuild | `cmake --build build --parallel` | Reuses configured build directory. |
| Tests | `tests/`, `tests/CMakeLists.txt` | Lit, C++ unit, C API, NumPy validation via CTest. |
| Container parity | `Dockerfile`, `docker-compose.yml`, `scripts/docker-verify.sh` | Docker lint includes `clang-format`; native CI lint currently only runs ruff. |
| CI | `.github/workflows/ci.yml` | Native lint/build/test on PRs; container test/publish on push. |
| RVV simulation | `.github/workflows/ci-rvv-sim.yml`, `scripts/setup-rvv-sim.sh`, `scripts/compile-rvv.sh` | Path-filtered QEMU matrix; Spike only on manual dispatch. |
| Cursor rules | `.cursor/rules/*.mdc` | Contextual guidance for project, MLIR, tests, CI, RVV. |
| Cursor skills | `.cursor/skills/*/SKILL.md` | Repeatable workflows agents should follow. |
| Agent prompts | `.cursor/agents/*.md` | Role templates for focused subagents/handoffs. |

## Testing expectations by change type

| Change | Minimum validation |
|---|---|
| TableGen op/verifier | Parse/print lit + invalid diagnostics + `ctest` |
| New pass/rewrite | Before/after FileCheck lit + edge/negative tests + `ctest` |
| RVV lowering | Lit + `ctest`; run RVV/QEMU workflow locally or note why unavailable |
| Python tooling | `ruff check .`, `ruff format --check .`, relevant pytest/CTest |
| CI/Docker/scripts | Syntax/readability check + targeted script dry run where safe |
| C API | C smoke tests + NumPy validation through CTest |

## Known gotchas

- LLVM/MLIR version is 21+. Do not reintroduce LLVM 18 guidance.
- `clang-format` enforcement is container-based unless the local formatter is
  installed. PR native lint runs ruff only.
- `ci-rvv-sim.yml` is path-filtered and separate from main CTest.
- QEMU/Spike tooling may not exist on all developer machines; use
  `scripts/setup-rvv-sim.sh` when needed.
- Keep generated or build artifacts out of commits.
