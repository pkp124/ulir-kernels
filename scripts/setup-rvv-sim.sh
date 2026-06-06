#!/usr/bin/env bash
# =============================================================================
# KernelSmith — RISC-V RVV Simulation Environment Setup
#
# Installs the tools needed to compile and simulate RVV kernels:
#   - qemu-user (qemu-riscv64)          mandatory — user-mode Linux ELF execution
#   - gcc-riscv64-linux-gnu             mandatory — cross-link C test harnesses
#   - spike (riscv-isa-sim)             optional  — ISA simulator, better RVV 1.0
#
# Usage:
#   ./scripts/setup-rvv-sim.sh                # install QEMU + cross-compiler
#   INSTALL_SPIKE=1 ./scripts/setup-rvv-sim.sh  # also install Spike
#
# Note: LLVM/MLIR 21 (ks-opt, mlir-translate, llc) must already be installed.
#       Run ./scripts/setup.sh first if starting from scratch.
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Config ------------------------------------------------------------------
INSTALL_SPIKE="${INSTALL_SPIKE:-0}"
SPIKE_INSTALL_DIR="${SPIKE_INSTALL_DIR:-/opt/riscv}"

# --- Colors ------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'
NC='\033[0m'
info()    { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()    { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error()   { echo -e "${RED}[ERROR]${NC} $*"; }
section() { echo ""; echo -e "${BLUE}====== $* ======${NC}"; }

# =============================================================================
# 1. QEMU user-mode (qemu-riscv64)
# =============================================================================
install_qemu_user() {
    section "QEMU user-mode (qemu-riscv64)"

    if command -v qemu-riscv64 &>/dev/null; then
        info "qemu-riscv64 already installed: $(qemu-riscv64 --version | head -1)"
        return 0
    fi

    info "Installing qemu-user via apt..."
    sudo apt-get update -qq
    # qemu-user provides qemu-riscv64 for user-mode Linux ELF execution.
    # qemu-user-static provides statically linked variants (for Docker).
    sudo apt-get install -y qemu-user qemu-user-static binfmt-support

    if command -v qemu-riscv64 &>/dev/null; then
        info "qemu-riscv64 installed: $(qemu-riscv64 --version | head -1)"
    else
        error "qemu-riscv64 not found after installation"
        exit 1
    fi
}

# =============================================================================
# 2. RISC-V Linux cross-compiler (riscv64-linux-gnu-gcc)
# =============================================================================
install_cross_compiler() {
    section "RISC-V cross-compiler (riscv64-linux-gnu-gcc)"

    if command -v riscv64-linux-gnu-gcc &>/dev/null; then
        info "Cross-compiler already installed: $(riscv64-linux-gnu-gcc --version | head -1)"
        return 0
    fi

    info "Installing riscv64-linux-gnu toolchain via apt..."
    sudo apt-get install -y \
        gcc-riscv64-linux-gnu \
        binutils-riscv64-linux-gnu \
        libc6-dev-riscv64-cross

    if command -v riscv64-linux-gnu-gcc &>/dev/null; then
        info "Cross-compiler installed: $(riscv64-linux-gnu-gcc --version | head -1)"
    else
        error "riscv64-linux-gnu-gcc not found after installation"
        exit 1
    fi
}

# =============================================================================
# 3. Spike — RISC-V ISA Simulator (optional)
#    Spike has better RVV 1.0 spec compliance than QEMU for edge cases.
#    Required: device-tree-compiler, libboost-{regex,system}-dev
# =============================================================================
install_spike() {
    section "Spike (riscv-isa-sim) — optional"

    if command -v spike &>/dev/null || [[ -f "${SPIKE_INSTALL_DIR}/bin/spike" ]]; then
        info "Spike already installed"
        return 0
    fi

    if [[ "${INSTALL_SPIKE}" != "1" ]]; then
        warn "Spike not installed. Set INSTALL_SPIKE=1 to build from source."
        warn "Spike is useful for edge-case RVV 1.0 validation."
        return 0
    fi

    info "Installing Spike build dependencies..."
    sudo apt-get install -y \
        device-tree-compiler \
        libboost-regex-dev \
        libboost-system-dev \
        build-essential git

    local tmpdir
    tmpdir="$(mktemp -d)"
    info "Cloning riscv-isa-sim..."
    git clone --depth 1 https://github.com/riscv-software-src/riscv-isa-sim.git "${tmpdir}"

    info "Building Spike..."
    pushd "${tmpdir}" >/dev/null
    mkdir -p build && cd build
    ../configure --prefix="${SPIKE_INSTALL_DIR}"
    make -j"$(nproc)"
    sudo make install
    popd >/dev/null
    rm -rf "${tmpdir}"

    if [[ -f "${SPIKE_INSTALL_DIR}/bin/spike" ]]; then
        info "Spike installed to ${SPIKE_INSTALL_DIR}/bin/spike"
        info "Add to PATH: export PATH=\"${SPIKE_INSTALL_DIR}/bin:\$PATH\""
    else
        error "Spike build failed"
        exit 1
    fi
}

# =============================================================================
# 4. Verify LLVM has RISC-V target (required for llc -march=riscv64)
# =============================================================================
verify_llvm_riscv() {
    section "Verify LLVM RISC-V target"

    if ! command -v llc &>/dev/null; then
        # Try llc-21
        if command -v llc-21 &>/dev/null; then
            info "Found llc-21; consider: sudo update-alternatives --install /usr/bin/llc llc /usr/bin/llc-21 21"
        else
            warn "llc not found. Install LLVM 21 first: ./scripts/setup.sh"
            return 0
        fi
    fi

    local llc_bin
    llc_bin="$(command -v llc-21 2>/dev/null || command -v llc)"
    if "${llc_bin}" --version 2>&1 | grep -qi "RISCV"; then
        info "LLVM RISC-V target available in $(basename "${llc_bin}")"
    else
        warn "LLVM RISC-V target not found in $(basename "${llc_bin}")"
        warn "You may need to build LLVM with -DLLVM_TARGETS_TO_BUILD=RISCV"
    fi
}

# =============================================================================
# 5. Summary
# =============================================================================
print_summary() {
    section "Summary"

    echo ""
    echo "  Tool                         Status"
    echo "  ──────────────────────────── ────────────────────────────────"

    if command -v qemu-riscv64 &>/dev/null; then
        printf "  %-28s %s\n" "qemu-riscv64" "$(qemu-riscv64 --version | head -1)"
    else
        printf "  %-28s %s\n" "qemu-riscv64" "NOT INSTALLED"
    fi

    if command -v riscv64-linux-gnu-gcc &>/dev/null; then
        printf "  %-28s %s\n" "riscv64-linux-gnu-gcc" \
               "$(riscv64-linux-gnu-gcc --version | head -1)"
    else
        printf "  %-28s %s\n" "riscv64-linux-gnu-gcc" "NOT INSTALLED"
    fi

    if command -v spike &>/dev/null || [[ -f "${SPIKE_INSTALL_DIR}/bin/spike" ]]; then
        printf "  %-28s %s\n" "spike" "installed"
    else
        printf "  %-28s %s\n" "spike" "not installed (optional)"
    fi

    local llc_bin
    llc_bin="$(command -v llc-21 2>/dev/null || command -v llc 2>/dev/null || echo '')"
    if [[ -n "${llc_bin}" ]]; then
        printf "  %-28s %s\n" "llc (LLVM)" "$(${llc_bin} --version 2>&1 | head -1)"
    else
        printf "  %-28s %s\n" "llc (LLVM)" "NOT FOUND"
    fi

    echo ""
    info "RVV simulation environment ready."
    echo ""
    info "Quick test:"
    echo "  PATH=\"/usr/lib/llvm-21/bin:\$PATH\" python3 tests/riscv_runner.py --all"
}

# =============================================================================
# Main
# =============================================================================
main() {
    info "KernelSmith RVV Simulation Environment Setup"

    install_qemu_user
    install_cross_compiler
    install_spike
    verify_llvm_riscv
    print_summary
}

main "$@"
