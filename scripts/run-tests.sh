#!/bin/bash
# KernelSmith Test Runner
# Runs tests in different categories with support for multi-VLEN testing

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_ROOT}/build"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test categories
TEST_LIT=false
TEST_UNIT=false
TEST_INTEGRATION=false
TEST_RISCV_FUNCTIONAL=false
TEST_ALL=false
VERBOSE=false
QEMU_VLEN=""

print_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Test Categories:
  --lit              Run MLIR lit tests (parsing, lowering verification)
  --unit             Run C++ unit tests
  --integration      Run integration tests
  --riscv-functional Run golden-backed RISC-V QEMU functional tests
  --all              Run all tests (default if no category specified)

Options:
  --verbose, -v      Verbose output
  --qemu-vlen BITS   Run with QEMU using specific VLEN (128, 256, 512)
  --help, -h         Show this help message

Examples:
  $0 --lit                           # Run lit tests
  $0 --all --verbose                 # Run all tests verbosely
  $0 --integration --qemu-vlen 256   # Run integration tests on QEMU with VLEN=256
  $0 --riscv-functional              # Run RVV golden checks at VLEN=256/512
EOF
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --lit)
            TEST_LIT=true
            shift
            ;;
        --unit)
            TEST_UNIT=true
            shift
            ;;
        --integration)
            TEST_INTEGRATION=true
            shift
            ;;
        --riscv-functional)
            TEST_RISCV_FUNCTIONAL=true
            shift
            ;;
        --all)
            TEST_ALL=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --qemu-vlen)
            QEMU_VLEN="$2"
            shift 2
            ;;
        --help|-h)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Default to --all if no category specified
if [ "$TEST_LIT" = false ] && [ "$TEST_UNIT" = false ] && [ "$TEST_INTEGRATION" = false ] && [ "$TEST_RISCV_FUNCTIONAL" = false ] && [ "$TEST_ALL" = false ]; then
    TEST_ALL=true
fi

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo -e "${RED}Error: Build directory not found at $BUILD_DIR${NC}"
    echo "Please run: make build"
    exit 1
fi

cd "$BUILD_DIR"

# Function to run tests
run_lit_tests() {
    echo -e "${YELLOW}=== Running LIT Tests ===${NC}"
    if command -v lit &> /dev/null; then
        if [ "$VERBOSE" = true ]; then
            cmake --build . --target check-kernelsmith-lit -- -v
        else
            cmake --build . --target check-kernelsmith-lit
        fi
    else
        echo -e "${RED}Error: lit not found. Install with: pip install lit${NC}"
        return 1
    fi
}

run_unit_tests() {
    echo -e "${YELLOW}=== Running Unit Tests ===${NC}"
    if [ "$VERBOSE" = true ]; then
        ctest --output-on-failure -V -R "kernelsmith-unit"
    else
        ctest --output-on-failure -R "kernelsmith-unit"
    fi
}

run_integration_tests() {
    echo -e "${YELLOW}=== Running Integration Tests ===${NC}"

    if [ -z "$QEMU_VLEN" ]; then
        # Host tests
        if [ "$VERBOSE" = true ]; then
            ctest --output-on-failure -V -R "kernelsmith-integration"
        else
            ctest --output-on-failure -R "kernelsmith-integration"
        fi
    else
        # QEMU tests with specific VLEN
        if ! command -v qemu-riscv64 &> /dev/null; then
            echo -e "${RED}Error: qemu-riscv64 not found. Install with: apt install qemu-user${NC}"
            return 1
        fi

        echo "Running on QEMU with VLEN=${QEMU_VLEN}"
        # Integration tests will set QEMU_VLEN environment variable
        QEMU_VLEN="$QEMU_VLEN" ctest --output-on-failure -R "kernelsmith-integration"
    fi
}

run_riscv_functional_tests() {
    echo -e "${YELLOW}=== Running RISC-V Golden Functional Tests ===${NC}"

    if ! command -v qemu-riscv64 &> /dev/null; then
        echo -e "${RED}Error: qemu-riscv64 not found. Run: ./scripts/setup-rvv-sim.sh${NC}"
        return 1
    fi
    if ! command -v riscv64-linux-gnu-gcc &> /dev/null; then
        echo -e "${RED}Error: riscv64-linux-gnu-gcc not found. Run: ./scripts/setup-rvv-sim.sh${NC}"
        return 1
    fi

    local python_bin="${PYTHON:-python3}"
    if [[ "${python_bin}" != /* && -x "${PROJECT_ROOT}/${python_bin}" ]]; then
        python_bin="${PROJECT_ROOT}/${python_bin}"
    fi
    local runner_dir="${BUILD_DIR}/rvv-functional/bin"
    local int8_object_dir="${BUILD_DIR}/rvv-functional/int8-objects"
    local output_dir="${BUILD_DIR}/rvv-functional/actual"
    local vlen_args=()
    local int8_rvv_objects=(
        "${int8_object_dir}/ks_dot_i8_riscv_rvv_256.o"
        "${int8_object_dir}/ks_matvec_i8_riscv_rvv_256.o"
    )
    mkdir -p "${runner_dir}" "${output_dir}"

    cmake --build . --target host-reference-runner
    cp "${BUILD_DIR}/tests/host_reference/host-reference-runner" \
       "${runner_dir}/host-reference-runner"

    "${python_bin}" "${PROJECT_ROOT}/scripts/generate-int8-rvv-objects.py" \
        --cc riscv64-linux-gnu-gcc \
        --source-root "${PROJECT_ROOT}" \
        --output-dir "${int8_object_dir}"

    riscv64-linux-gnu-gcc \
        -std=c99 \
        -O2 \
        -static \
        -march=rv64gcv \
        -mabi=lp64d \
        -DKS_QUANTIZED_INT8_EXTERNAL=1 \
        -I "${PROJECT_ROOT}/include" \
        -include "${PROJECT_ROOT}/target/riscv_rvv_256.h" \
        "${PROJECT_ROOT}/tests/host_reference/host_reference_runner.c" \
        "${PROJECT_ROOT}/lib/kernelsmith/ks_common.c" \
        "${PROJECT_ROOT}/lib/kernelsmith/ks_matmul.c" \
        "${PROJECT_ROOT}/lib/kernelsmith/ks_activations.c" \
        "${PROJECT_ROOT}/lib/kernelsmith/ks_quantized.c" \
        "${int8_rvv_objects[@]}" \
        -o "${runner_dir}/riscv-golden-runner" \
        -lm

    if [ -n "$QEMU_VLEN" ]; then
        vlen_args=("$QEMU_VLEN")
    else
        vlen_args=(256 512)
    fi

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/relu_f32_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/relu_riscv_report.json"

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/matmul_f32_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/matmul_riscv_report.json" \
        --benchmark \
        --benchmark-runs 3 \
        --benchmark-warmup 1

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/dot_i8_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/dot_i8_riscv_report.json"

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/matvec_i8_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/matvec_i8_riscv_report.json"

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/dot_w4a8_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/dot_w4a8_riscv_report.json"

    "${python_bin}" "${PROJECT_ROOT}/tests/verify.py" \
        --case "${PROJECT_ROOT}/tests/golden/cases/matvec_w4a8_smoke.json" \
        --target riscv_rvv_256 \
        --riscv-runner "${runner_dir}/riscv-golden-runner" \
        --host-runner "${runner_dir}/host-reference-runner" \
        --qemu qemu-riscv64 \
        --vlens "${vlen_args[@]}" \
        --output-dir "${output_dir}" \
        --report "${output_dir}/matvec_w4a8_riscv_report.json" \
        --benchmark \
        --benchmark-runs 3 \
        --benchmark-warmup 1
}

# Run tests based on flags
FAILED=0

if [ "$TEST_ALL" = true ]; then
    run_lit_tests || FAILED=$((FAILED + 1))
    run_unit_tests || FAILED=$((FAILED + 1))
    run_integration_tests || FAILED=$((FAILED + 1))
else
    if [ "$TEST_LIT" = true ]; then
        run_lit_tests || FAILED=$((FAILED + 1))
    fi
    if [ "$TEST_UNIT" = true ]; then
        run_unit_tests || FAILED=$((FAILED + 1))
    fi
    if [ "$TEST_INTEGRATION" = true ]; then
        run_integration_tests || FAILED=$((FAILED + 1))
    fi
    if [ "$TEST_RISCV_FUNCTIONAL" = true ]; then
        run_riscv_functional_tests || FAILED=$((FAILED + 1))
    fi
fi

# Summary
echo ""
if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    exit 1
fi
