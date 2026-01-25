#!/bin/bash
# ==============================================================================
# KernelSmith Project Setup Script
# ==============================================================================

set -e

echo "============================================"
echo "KernelSmith Project Setup"
echo "============================================"
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

check_command() {
    if command -v "$1" &> /dev/null; then
        echo -e "${GREEN}✓${NC} $1 found"
        return 0
    else
        echo -e "${RED}✗${NC} $1 not found"
        return 1
    fi
}

echo "Checking dependencies..."
echo ""

# Check required tools
MISSING=0

check_command cmake || MISSING=1
check_command ninja || check_command make
check_command python3 || MISSING=1
check_command clang++ || check_command g++ || MISSING=1

echo ""

# Check LLVM/MLIR
echo "Checking LLVM/MLIR installation..."

if command -v llvm-config &> /dev/null; then
    LLVM_VERSION=$(llvm-config --version)
    echo -e "${GREEN}✓${NC} LLVM found: version $LLVM_VERSION"
    
    LLVM_PREFIX=$(llvm-config --prefix)
    if [ -d "$LLVM_PREFIX/lib/cmake/mlir" ]; then
        echo -e "${GREEN}✓${NC} MLIR found at $LLVM_PREFIX"
        export MLIR_DIR="$LLVM_PREFIX/lib/cmake/mlir"
    else
        echo -e "${YELLOW}!${NC} MLIR not found in LLVM installation"
    fi
else
    echo -e "${YELLOW}!${NC} LLVM not found in PATH"
    echo "  Install LLVM 18+ with MLIR, or set MLIR_DIR"
fi

echo ""

# Python environment
echo "Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Created Python virtual environment"
fi

source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt 2>/dev/null || {
    pip install --quiet pytest ruff lit filecheck
}

echo -e "${GREEN}✓${NC} Python dependencies installed"

echo ""

# Create compile_commands.json symlink
if [ -f "build/compile_commands.json" ]; then
    ln -sf build/compile_commands.json compile_commands.json
    echo -e "${GREEN}✓${NC} Linked compile_commands.json"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate Python environment: source venv/bin/activate"
echo "  2. Configure build: make configure"
echo "  3. Build: make build"
echo "  4. Test: make test"
echo ""
