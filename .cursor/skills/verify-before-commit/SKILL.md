# Verify KernelSmith Before Commit

Use this skill before committing or summarizing implementation work.

## Steps

1. Check the worktree:
   ```bash
   git status --short --branch
   ```
2. Rebuild changed C++/TableGen code:
   ```bash
   cmake --build build --parallel
   ```
   If `build/` does not exist, run `./scripts/setup.sh`.
3. Run the full native test suite:
   ```bash
   ctest --test-dir build --output-on-failure
   ```
4. For Python changes, run:
   ```bash
   ruff check .
   ruff format --check .
   ```
5. For C++ formatting or pre-push parity, prefer:
   ```bash
   ./scripts/docker-verify.sh
   ```
   If Docker is unavailable, say so and report the native checks you did run.
6. Include exact commands and results in the final summary or PR body.

## Notes

- `make verify` is the native shorthand for lint/build/test.
- Docker lint includes clang-format; native PR lint currently does not.
- Always run `ctest --test-dir build --output-on-failure` before claiming a change is ready.
