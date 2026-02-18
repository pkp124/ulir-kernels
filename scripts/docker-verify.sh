#!/usr/bin/env bash
# ==============================================================================
# KernelSmith — container build + test verification
#
# Replicates exactly what the CI container-test job does, so you can catch
# failures locally before pushing to the branch.
#
# Usage:
#   ./scripts/docker-verify.sh             # cached (fast after first run)
#   ./scripts/docker-verify.sh --no-cache  # full clean build
#
# Requires: docker (BuildKit — shipped with Docker Desktop and Engine 20+)
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE="kernelsmith:verify"
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
echo "KernelSmith — container build + test"
echo "============================================"
echo ""

cd "$REPO_ROOT"

# Stage 1: build the 'test' image (mirrors CI container-test exactly)
echo "--- Building 'test' stage ---"
DOCKER_BUILDKIT=1 docker build \
  "${BUILD_ARGS[@]}" \
  --target test \
  --tag "$IMAGE" \
  .

echo ""

# Stage 2: run the tests inside the container
echo "--- Running tests in container ---"
docker run --rm "$IMAGE"

echo ""
info "Container build and tests passed."
echo ""
echo "The image matches what CI will build — safe to push."
