#!/usr/bin/env bash
# ==============================================================================
# KernelSmith — development environment setup
# Installs all dependencies and builds the project.
# Run from repository root: ./scripts/setup.sh
#
# Supports: macOS (Homebrew) and Ubuntu/Debian (apt)
# ==============================================================================

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${REPO_ROOT}/.venv"
BUILD_DIR="${REPO_ROOT}/build"
PYTHON="${PYTHON:-python3}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

info()  { echo -e "${GREEN}✓${NC} $*"; }
warn()  { echo -e "${YELLOW}!${NC} $*"; }
fail()  { echo -e "${RED}✗ $*${NC}"; exit 1; }

echo "============================================"
echo "KernelSmith — dev environment setup"
echo "============================================"
echo ""

cd "$REPO_ROOT"

[ -f "pyproject.toml" ] || fail "pyproject.toml not found. Run from repo root."

# ==============================================================================
# 1. System dependencies (LLVM/MLIR 20, cmake, ninja)
# ==============================================================================

echo "--- System dependencies ---"

OS="$(uname -s)"

install_macos() {
  if ! command -v brew &>/dev/null; then
    fail "Homebrew not found. Install from https://brew.sh"
  fi

  local pkgs=()

  # cmake
  if ! command -v cmake &>/dev/null; then
    pkgs+=(cmake)
  else
    info "cmake $(cmake --version | head -1 | awk '{print $3}')"
  fi

  # ninja
  if ! command -v ninja &>/dev/null; then
    pkgs+=(ninja)
  else
    info "ninja $(ninja --version)"
  fi

  # llvm@20 (includes MLIR)
  if ! brew ls --versions llvm@20 &>/dev/null; then
    pkgs+=(llvm@20)
  else
    info "llvm@20 (Homebrew)"
  fi

  if [ ${#pkgs[@]} -gt 0 ]; then
    echo "Installing: ${pkgs[*]} ..."
    brew install "${pkgs[@]}"
    info "Installed ${pkgs[*]}"
  fi

  # Resolve LLVM paths (Homebrew keg-only)
  LLVM_PREFIX="$(brew --prefix llvm@20)"
  LLVM_DIR="${LLVM_PREFIX}/lib/cmake/llvm"
  MLIR_DIR="${LLVM_PREFIX}/lib/cmake/mlir"
}

install_linux() {
  local pkgs=()

  # cmake
  if ! command -v cmake &>/dev/null; then
    pkgs+=(cmake)
  else
    info "cmake $(cmake --version | head -1 | awk '{print $3}')"
  fi

  # ninja
  if ! command -v ninja &>/dev/null; then
    pkgs+=(ninja-build)
  else
    info "ninja $(ninja --version)"
  fi

  # LLVM/MLIR 20
  if ! dpkg -s mlir-20-tools &>/dev/null 2>&1; then
    echo "Adding LLVM 20 apt repository ..."
    wget -qO- https://apt.llvm.org/llvm.sh | sudo bash -s -- 20
    pkgs+=(mlir-20-tools libmlir-20-dev llvm-20-dev libgtest-dev)
  else
    info "mlir-20-tools (apt)"
  fi

  if [ ${#pkgs[@]} -gt 0 ]; then
    echo "Installing: ${pkgs[*]} ..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq "${pkgs[@]}"
    info "Installed ${pkgs[*]}"
  fi

  LLVM_DIR="/usr/lib/llvm-20/lib/cmake/llvm"
  MLIR_DIR="/usr/lib/llvm-20/lib/cmake/mlir"
}

case "$OS" in
  Darwin) install_macos ;;
  Linux)  install_linux ;;
  *)      fail "Unsupported OS: $OS. Only macOS and Linux are supported." ;;
esac

# Verify MLIR is findable
if [ ! -d "$MLIR_DIR" ]; then
  fail "MLIR cmake config not found at $MLIR_DIR"
fi
info "MLIR: $MLIR_DIR"
echo ""

# ==============================================================================
# 2. Python virtual environment
# ==============================================================================

echo "--- Python environment ---"

if ! command -v "$PYTHON" &>/dev/null; then
  fail "$PYTHON not found. Install Python 3.10+ or set PYTHON env var."
fi

echo "Python: $($PYTHON --version)"

if [ ! -d "$VENV_DIR" ]; then
  echo "Creating virtual environment at $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
  info "Created venv"
else
  info "Using existing venv at $VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

pip install --quiet --upgrade pip
pip install --quiet -e ".[dev]"

info "Python dev environment ready"
echo ""

# ==============================================================================
# 3. CMake configure & build
# ==============================================================================

echo "--- Build ---"

cmake -S "$REPO_ROOT" -B "$BUILD_DIR" \
  -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_DIR="$LLVM_DIR" \
  -DMLIR_DIR="$MLIR_DIR"

cmake --build "$BUILD_DIR" --parallel

info "Build complete"
echo ""

# Link compile_commands.json for IDE support
if [ -f "$BUILD_DIR/compile_commands.json" ]; then
  ln -sf "$BUILD_DIR/compile_commands.json" "$REPO_ROOT/compile_commands.json" 2>/dev/null || true
  info "Linked compile_commands.json"
fi

# ==============================================================================
# 4. Run tests
# ==============================================================================

echo "--- Tests ---"

ctest --test-dir "$BUILD_DIR" --output-on-failure

info "All tests passed"
echo ""

# ==============================================================================
# Done
# ==============================================================================

echo "============================================"
echo "Setup complete — build and tests passed"
echo "============================================"
echo ""
echo "To rebuild after changes:"
echo "  source .venv/bin/activate"
echo "  cmake --build build --parallel"
echo "  ctest --test-dir build --output-on-failure"
echo ""
