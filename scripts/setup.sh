#!/bin/bash
# ==============================================================================
# Project Setup Script
# ==============================================================================

set -e

echo "============================================"
echo "MLIR Kernel Generation Project Setup"
echo "============================================"
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

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

# Check for LLVM/MLIR
echo "Checking LLVM/MLIR installation..."

if command -v llvm-config &> /dev/null; then
    LLVM_VERSION=$(llvm-config --version)
    echo -e "${GREEN}✓${NC} LLVM found: version $LLVM_VERSION"
    
    LLVM_PREFIX=$(llvm-config --prefix)
    if [ -d "$LLVM_PREFIX/lib/cmake/mlir" ]; then
        echo -e "${GREEN}✓${NC} MLIR found at $LLVM_PREFIX"
    else
        echo -e "${YELLOW}!${NC} MLIR not found in LLVM installation"
        echo "  You may need to build LLVM with MLIR enabled"
    fi
else
    echo -e "${YELLOW}!${NC} LLVM not found in PATH"
    echo "  Install LLVM 18+ with MLIR, or set LLVM_DIR"
fi

echo ""

# Setup Python environment
echo "Setting up Python environment..."

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓${NC} Created Python virtual environment"
fi

source venv/bin/activate

pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt 2>/dev/null || {
    echo "Creating requirements.txt..."
    cat > requirements.txt << 'EOF'
# Development dependencies
pytest>=7.0
ruff>=0.1.0
cmake-format>=0.6

# Documentation
mkdocs>=1.5
mkdocs-material>=9.0

# Python bindings (optional)
pybind11>=2.11
numpy>=1.24
EOF
    pip install --quiet -r requirements.txt
}

echo -e "${GREEN}✓${NC} Python dependencies installed"

echo ""

# Create compile_commands.json symlink for IDE support
if [ -f "build/compile_commands.json" ]; then
    ln -sf build/compile_commands.json compile_commands.json
    echo -e "${GREEN}✓${NC} Linked compile_commands.json for IDE support"
fi

echo ""
echo "============================================"
echo "Setup complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. Activate Python environment: source venv/bin/activate"
echo "  2. Build the project: make build"
echo "  3. Run tests: make test"
echo "  4. See all commands: make help"
echo ""

if [ $MISSING -eq 1 ]; then
    echo -e "${YELLOW}Warning:${NC} Some dependencies are missing."
    echo "Install them before building."
fi
