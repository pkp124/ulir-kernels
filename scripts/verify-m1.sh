#!/bin/bash
# ==============================================================================
# M1 Milestone Verification Script
# ==============================================================================
# 
# Verifies all M1 criteria:
# - ks.matmul parses and verifies
# - Full lowering pipeline works
# - Generated RVV code runs on QEMU
# - Output matches reference for 3 sizes
# - Works with multiple VLEN values
#
# Usage: ./scripts/verify-m1.sh
# ==============================================================================

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

PASS=0
FAIL=0

check() {
    local name=$1
    local cmd=$2
    
    echo -n "Checking: $name... "
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "${GREEN}PASS${NC}"
        ((PASS++))
    else
        echo -e "${RED}FAIL${NC}"
        ((FAIL++))
    fi
}

echo "============================================"
echo "M1 Milestone Verification"
echo "============================================"
echo ""

# Check prerequisites
echo "Prerequisites:"
check "ks-opt binary exists" "[ -f build/bin/ks-opt ]"
check "QEMU with RVV available" "command -v qemu-riscv64"
check "RISC-V GCC available" "command -v riscv64-linux-gnu-gcc || command -v riscv64-unknown-elf-gcc"
echo ""

# Check dialect functionality
echo "Dialect Tests:"
check "ks.matmul parses" "build/bin/ks-opt tests/lit/Dialect/Kernel/matmul.mlir"
check "Lit tests pass" "ctest --test-dir build -R 'lit' --output-on-failure"
echo ""

# Check lowering pipeline
echo "Lowering Pipeline:"
check "--ks-lower-to-linalg works" "build/bin/ks-opt tests/e2e/matmul_64x64.mlir --ks-lower-to-linalg"
check "--ks-tile works" "build/bin/ks-opt tests/e2e/matmul_64x64.mlir --ks-lower-to-linalg --ks-tile"
check "--ks-vectorize works" "build/bin/ks-opt tests/e2e/matmul_64x64.mlir --ks-lower-to-linalg --ks-tile --ks-vectorize"
check "--ks-lower-to-rvv works" "build/bin/ks-opt tests/e2e/matmul_64x64.mlir --ks-lower-to-linalg --ks-tile --ks-vectorize --ks-lower-to-rvv"
echo ""

# Check end-to-end execution
echo "End-to-End Execution:"
check "64x64 matmul correct" "./scripts/run-matmul-test.sh 64"
check "128x128 matmul correct" "./scripts/run-matmul-test.sh 128"
check "256x256 matmul correct" "./scripts/run-matmul-test.sh 256"
echo ""

# Check multi-VLEN
echo "Multi-VLEN Compatibility:"
check "VLEN=128 works" "./scripts/run-matmul-test.sh 64 128"
check "VLEN=256 works" "./scripts/run-matmul-test.sh 64 256"
check "VLEN=512 works" "./scripts/run-matmul-test.sh 64 512"
echo ""

# Summary
echo "============================================"
echo "M1 Verification Summary"
echo "============================================"
echo -e "Passed: ${GREEN}$PASS${NC}"
echo -e "Failed: ${RED}$FAIL${NC}"
echo ""

TOTAL=$((PASS + FAIL))
if [ $FAIL -eq 0 ]; then
    echo -e "${GREEN}M1 MILESTONE COMPLETE!${NC}"
    echo "All $TOTAL criteria passed."
    exit 0
else
    echo -e "${RED}M1 NOT COMPLETE${NC}"
    echo "$FAIL of $TOTAL criteria failed."
    echo ""
    echo "Fix failing criteria before claiming M1 complete."
    exit 1
fi
