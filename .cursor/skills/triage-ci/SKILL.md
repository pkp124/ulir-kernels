# Triage KernelSmith CI

Use this skill when CI or local verification fails.

## Source of truth

- Main PR checks: `.github/workflows/ci.yml`
- RVV/QEMU checks: `.github/workflows/ci-rvv-sim.yml`
- Local container parity: `./scripts/docker-verify.sh`
- Native test command: `ctest --test-dir build --output-on-failure`

## Triage steps

1. Identify the failing job: native lint, native build/test, container test, RVV simulation, or publish.
2. Reproduce the closest local command:
   - Python lint: `ruff check . && ruff format --check .`
   - Build/test: `cmake --build build --parallel && ctest --test-dir build --output-on-failure`
   - Container: `./scripts/docker-verify.sh`
   - RVV: `./scripts/setup-rvv-sim.sh` then `scripts/compile-rvv.sh ...`
3. Fix the smallest root cause.
4. Rerun the failed command and then the full native CTest suite.
5. Commit and push the fix with a `fix(...)` or `chore(...)` message.

## Known CI split

PR CI does not run the Docker `container-test` job. Docker is still the best way
to catch clang-format and container dependency issues before push.
