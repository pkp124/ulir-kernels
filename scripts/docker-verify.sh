#!/usr/bin/env bash
# ==============================================================================
# KernelSmith — container lint + build + test verification
#
# Replicates exactly what CI does, so you can catch failures locally before
# pushing to the branch:
#   1. Lint  (ruff + clang-format) — mirrors the CI 'lint' job
#   2. Build + test (ctest)        — mirrors the CI 'container-test' job
#
# Usage:
#   ./scripts/docker-verify.sh             # cached (fast after first run)
#   ./scripts/docker-verify.sh --no-cache  # full clean build
#
# Requires: docker (BuildKit — shipped with Docker Desktop and Engine 20+)
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_ARGS=()

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${GREEN}✓${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
fail() { echo -e "${RED}✗ $*${NC}"; exit 1; }

if [[ "${1:-}" == "--no-cache" ]]; then
  BUILD_ARGS+=(--no-cache)
  warn "Cache disabled — full build (slow)"
fi

command -v docker &>/dev/null || fail "docker not found. Install Docker Desktop or Docker Engine."

echo "============================================"
echo "KernelSmith — container verify (lint + test)"
echo "============================================"
echo ""

cd "$REPO_ROOT"

# ---------------------------------------------------------------------------
# Stage 1: lint (ruff + clang-format) — fast, no C++ compilation
# ---------------------------------------------------------------------------
echo "--- Building 'lint' stage ---"
DOCKER_BUILDKIT=1 docker build \
  "${BUILD_ARGS[@]}" \
  --target lint \
  --tag kernelsmith:verify-lint \
  .

echo ""
echo "--- Running lint checks in container ---"
docker run --rm kernelsmith:verify-lint

info "Lint passed."
echo ""

# ---------------------------------------------------------------------------
# Stage 2: build + test (ctest) — mirrors CI container-test job exactly
# ---------------------------------------------------------------------------
echo "--- Building 'test' stage ---"
DOCKER_BUILDKIT=1 docker build \
  "${BUILD_ARGS[@]}" \
  --target test \
  --tag kernelsmith:verify-test \
  .

echo ""
echo "--- Running tests in container ---"
docker run --rm kernelsmith:verify-test

echo ""
info "Container build and tests passed."
echo ""
echo "All checks match CI — safe to push."
