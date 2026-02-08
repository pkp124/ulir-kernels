#!/usr/bin/env bash
# ==============================================================================
# KernelSmith — development environment setup
# Run from repository root: ./scripts/setup.sh
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
PYTHON="${PYTHON:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "============================================"
echo "KernelSmith — dev environment setup"
echo "============================================"
echo ""

cd "$REPO_ROOT"

if [ ! -f "pyproject.toml" ]; then
  echo -e "${RED}Error: pyproject.toml not found. Run this script from the repo root.${NC}"
  exit 1
fi

if ! command -v "$PYTHON" &>/dev/null; then
  echo -e "${RED}Error: $PYTHON not found. Install Python 3.10+ or set PYTHON.${NC}"
  exit 1
fi

echo "Python: $($PYTHON --version)"
echo ""

# Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
  echo -e "${GREEN}✓${NC} Created venv"
else
  echo -e "${GREEN}✓${NC} Using existing venv at $VENV_DIR"
fi

# Activate and install
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

echo "Upgrading pip ..."
pip install --quiet --upgrade pip

echo "Installing project with dev dependencies (ruff, pytest, lit, filecheck) ..."
pip install --quiet -e ".[dev]"

echo -e "${GREEN}✓${NC} Python dev environment ready"
echo ""

# Optional: compile_commands.json for IDE
if [ -f "build/compile_commands.json" ]; then
  ln -sf build/compile_commands.json compile_commands.json 2>/dev/null || true
  echo -e "${GREEN}✓${NC} Linked compile_commands.json"
fi

echo ""
echo "============================================"
echo "Setup complete"
echo "============================================"
echo ""
echo "Activate the environment:"
echo "  source .venv/bin/activate"
echo ""
echo "Build and test:"
echo "  cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMLIR_DIR=/path/to/mlir/lib/cmake/mlir"
echo "  cmake --build build --parallel"
echo "  ctest --test-dir build --output-on-failure"
echo ""
