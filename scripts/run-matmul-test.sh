#!/bin/bash
# ==============================================================================
# Run a matmul test on QEMU with RVV
# ==============================================================================
#
# Usage: ./scripts/run-matmul-test.sh <size> [vlen]
#   size: Matrix size (e.g., 64, 128, 256)
#   vlen: Vector length in bits (default: 256)
#
# Example: ./scripts/run-matmul-test.sh 64 256
# ==============================================================================

set -e

SIZE=${1:-64}
VLEN=${2:-256}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="$PROJECT_DIR/build"
TMP_DIR="/tmp/kernelsmith-test-$$"

mkdir -p "$TMP_DIR"
trap "rm -rf $TMP_DIR" EXIT

echo "Testing ${SIZE}x${SIZE} matmul with VLEN=$VLEN"

# 1. Compile kernel to RVV
echo "  Compiling kernel..."
"$BUILD_DIR/bin/ks-opt" "$PROJECT_DIR/tests/e2e/matmul_${SIZE}x${SIZE}.mlir" \
    --ks-lower-to-linalg \
    --ks-tile \
    --ks-vectorize \
    --ks-lower-to-rvv \
    --convert-to-llvm \
    -o "$TMP_DIR/matmul.llvm.mlir"

# 2. Generate LLVM IR
echo "  Generating LLVM IR..."
mlir-translate --mlir-to-llvmir "$TMP_DIR/matmul.llvm.mlir" -o "$TMP_DIR/matmul.ll"

# 3. Compile to assembly
echo "  Generating RISC-V assembly..."
llc -march=riscv64 -mattr=+v "$TMP_DIR/matmul.ll" -o "$TMP_DIR/matmul.s"

# 4. Build executable with test harness
echo "  Building executable..."
RISCV_GCC=$(command -v riscv64-linux-gnu-gcc || command -v riscv64-unknown-elf-gcc)
$RISCV_GCC "$TMP_DIR/matmul.s" "$PROJECT_DIR/tests/e2e/matmul_harness.c" \
    -DMATRIX_SIZE=$SIZE \
    -o "$TMP_DIR/matmul_test" \
    -static

# 5. Run on QEMU
echo "  Running on QEMU (VLEN=$VLEN)..."
qemu-riscv64 -cpu rv64,v=true,vlen=$VLEN "$TMP_DIR/matmul_test" > "$TMP_DIR/output.txt"

# 6. Verify output
echo "  Verifying output..."
EXPECTED="$PROJECT_DIR/tests/e2e/matmul_${SIZE}x${SIZE}_expected.txt"
if [ -f "$EXPECTED" ]; then
    if diff -q "$TMP_DIR/output.txt" "$EXPECTED" > /dev/null; then
        echo "  Result: PASS"
        exit 0
    else
        echo "  Result: FAIL (output mismatch)"
        diff "$TMP_DIR/output.txt" "$EXPECTED"
        exit 1
    fi
else
    echo "  Warning: No expected output file, checking for errors only"
    if grep -q "ERROR" "$TMP_DIR/output.txt"; then
        echo "  Result: FAIL (errors in output)"
        cat "$TMP_DIR/output.txt"
        exit 1
    else
        echo "  Result: PASS (no errors)"
        exit 0
    fi
fi
