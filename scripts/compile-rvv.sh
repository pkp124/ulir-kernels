#!/usr/bin/env bash
# compile-rvv.sh — Compile a KernelSmith MLIR module to RISC-V RVV object code.
#
# Usage:
#   scripts/compile-rvv.sh input.mlir output.o
#   scripts/compile-rvv.sh input.mlir output.s   # stop at assembly
#
# Requirements:
#   - ks-opt in PATH or BUILD_DIR/bin/ks-opt
#   - mlir-translate in PATH (from LLVM 21 install)
#   - llc in PATH (from LLVM 21 install, with RISC-V target enabled)
#
# The input MLIR must already be at the ks dialect level (ks.matmul, etc.).
# This script applies the full M4 lowering pipeline:
#   ks.matmul → linalg → tile (L2) → pack → tile (reg) → vectorize → llvm → .o
#
# Target: riscv64 with RVV V1.0, ZVE64D, ZVL256B (VLEN=256 baseline).
# Generated .o files link against libkernelsmith.a (see docs/guides/cross-compile.md).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# --- Configuration -----------------------------------------------------------
BUILD_DIR="${BUILD_DIR:-${REPO_ROOT}/build}"
KS_OPT="${KS_OPT:-${BUILD_DIR}/bin/ks-opt}"
MLIR_TRANSLATE="${MLIR_TRANSLATE:-mlir-translate}"
LLC="${LLC:-llc}"

# RVV target parameters (matches target/riscv_rvv_256.h)
RVV_TRIPLE="riscv64-unknown-linux-gnu"
RVV_MARCH="rv64gcv"
RVV_MATTR="+v,+zve64d,+zvl256b"
RVV_FLOAT_ABI="hard"

# M4 tile sizes (from target/riscv_rvv_256.h)
L2_M=128; L2_N=128; L2_K=256
REG_M=16;  REG_N=32;  REG_K=0
PACK_FACTOR=32

# --- Argument parsing --------------------------------------------------------
if [[ $# -lt 2 ]]; then
    echo "Usage: $0 <input.mlir> <output.o|output.s>" >&2
    exit 1
fi

INPUT_MLIR="$1"
OUTPUT="$2"

if [[ ! -f "${INPUT_MLIR}" ]]; then
    echo "ERROR: Input file not found: ${INPUT_MLIR}" >&2
    exit 1
fi

EXT="${OUTPUT##*.}"
if [[ "${EXT}" != "o" && "${EXT}" != "s" && "${EXT}" != "ll" ]]; then
    echo "ERROR: Output must be .o (object), .s (assembly), or .ll (LLVM IR)" >&2
    exit 1
fi

# --- Tool checks -------------------------------------------------------------
for tool in "${KS_OPT}" "${MLIR_TRANSLATE}" "${LLC}"; do
    if ! command -v "${tool}" &>/dev/null && [[ ! -x "${tool}" ]]; then
        echo "ERROR: Required tool not found: ${tool}" >&2
        echo "  Set BUILD_DIR, KS_OPT, MLIR_TRANSLATE, or LLC env vars." >&2
        exit 1
    fi
done

# --- Temporary files ---------------------------------------------------------
TMP_LOWERED=$(mktemp /tmp/ks_lowered_XXXXXX.mlir)
TMP_LLVM_IR=$(mktemp /tmp/ks_llvm_XXXXXX.ll)
trap 'rm -f "${TMP_LOWERED}" "${TMP_LLVM_IR}"' EXIT

echo "[ks-compile-rvv] Input:  ${INPUT_MLIR}"
echo "[ks-compile-rvv] Output: ${OUTPUT}"

# --- Stage 1: KernelSmith → LLVM dialect (via ks-opt) -----------------------
echo "[ks-compile-rvv] Stage 1: ks-opt lowering pipeline..."
"${KS_OPT}" "${INPUT_MLIR}" \
    --ks-lower-to-linalg \
    "--ks-tile=tile-size-m=${L2_M} tile-size-n=${L2_N} tile-size-k=${L2_K}" \
    "--ks-pack=pack-factor=${PACK_FACTOR}" \
    "--ks-tile=tile-size-m=${REG_M} tile-size-n=${REG_N} tile-size-k=${REG_K}" \
    --ks-vectorize \
    --ks-lower-to-rvv \
    -o "${TMP_LOWERED}"

# --- Stage 2: MLIR LLVM dialect → LLVM IR -----------------------------------
echo "[ks-compile-rvv] Stage 2: mlir-translate → LLVM IR..."
"${MLIR_TRANSLATE}" --mlir-to-llvmir "${TMP_LOWERED}" -o "${TMP_LLVM_IR}"

if [[ "${EXT}" == "ll" ]]; then
    cp "${TMP_LLVM_IR}" "${OUTPUT}"
    echo "[ks-compile-rvv] Done (LLVM IR): ${OUTPUT}"
    exit 0
fi

# --- Stage 3: LLVM IR → assembly or object ----------------------------------
echo "[ks-compile-rvv] Stage 3: llc → ${EXT}..."
LLC_FILETYPE="obj"
if [[ "${EXT}" == "s" ]]; then
    LLC_FILETYPE="asm"
fi

"${LLC}" \
    -mtriple="${RVV_TRIPLE}" \
    -march=riscv64 \
    -mcpu=generic-rv64 \
    -mattr="${RVV_MATTR}" \
    -float-abi="${RVV_FLOAT_ABI}" \
    -filetype="${LLC_FILETYPE}" \
    "${TMP_LLVM_IR}" \
    -o "${OUTPUT}"

echo "[ks-compile-rvv] Done: ${OUTPUT}"
