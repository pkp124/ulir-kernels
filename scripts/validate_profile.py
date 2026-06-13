#!/usr/bin/env python3
"""Validate KernelSmith target profile headers."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


DEFINE_RE = re.compile(
    r"^\s*#\s*define\s+([A-Za-z_][A-Za-z0-9_]*)(?:\([^)]*\))?\s*(.*)$"
)

COMMON_REQUIRED_FIELDS = [
    "KS_TARGET_NAME",
    "KS_TARGET_ARCH",
    "KS_SIMD_WIDTH_BITS",
    "KS_SIMD_WIDTH_F32",
    "KS_SIMD_WIDTH_F16",
    "KS_SIMD_WIDTH_I8",
    "KS_L1D_SIZE_KB",
    "KS_L2_SIZE_KB",
    "KS_L3_SIZE_KB",
    "KS_CACHELINE_BYTES",
    "KS_PREFERRED_ALIGN",
    "KS_REQUIRED_ALIGN",
    "KS_MATMUL_TILE_M_L2",
    "KS_MATMUL_TILE_N_L2",
    "KS_MATMUL_TILE_K_L2",
    "KS_MATMUL_TILE_M_L1",
    "KS_MATMUL_TILE_N_L1",
    "KS_MATMUL_TILE_K_L1",
    "KS_MATMUL_MR",
    "KS_MATMUL_NR",
    "KS_MATMUL_PACK_B",
    "KS_MATMUL_PACK_A",
    "KS_HAS_RVV",
    "KS_MATMUL_DOUBLE_BUFFER",
    "KS_ELEMENTWISE_TILE",
]

RVV_QUANTIZED_REQUIRED_FIELDS = [
    "KS_RVV_SEW_I32",
    "KS_RVV_SEW_I8",
    "KS_RVV_VLMAX_I32_LMUL1",
    "KS_RVV_VLMAX_I8_LMUL4",
    "KS_QUANT_DOT_I8_TILE_K",
    "KS_QUANT_MATVEC_I8_TILE_ROWS",
    "KS_QUANT_MATVEC_I8_TILE_COLS",
    "KS_QUANT_I8_PACK_FACTOR",
    "KS_QUANT_I8_PACK_ALIGN",
    "KS_QUANT_MATMUL_I8_TILE_M_L2",
    "KS_QUANT_MATMUL_I8_TILE_N_L2",
    "KS_QUANT_MATMUL_I8_TILE_K_L2",
    "KS_QUANT_MATMUL_I8_TILE_M_L1",
    "KS_QUANT_MATMUL_I8_TILE_N_L1",
    "KS_QUANT_MATMUL_I8_TILE_K_L1",
    "KS_QUANT_DOT_W4A8_TILE_K",
    "KS_QUANT_MATVEC_W4A8_TILE_ROWS",
    "KS_QUANT_MATVEC_W4A8_TILE_COLS",
    "KS_QUANT_W4A8_GROUP_SIZE",
    "KS_QUANT_W4A8_PACK_FACTOR",
    "KS_QUANT_W4A8_PACKED_TILE_BYTES",
    "KS_QUANT_W4A8_PACK_ALIGN",
    "KS_QUANT_W4A8_SCALE_ALIGN",
    "KS_QUANT_MATMUL_W4A8_TILE_M_L2",
    "KS_QUANT_MATMUL_W4A8_TILE_N_L2",
    "KS_QUANT_MATMUL_W4A8_TILE_K_L2",
    "KS_QUANT_MATMUL_W4A8_TILE_M_L1",
    "KS_QUANT_MATMUL_W4A8_TILE_N_L1",
    "KS_QUANT_MATMUL_W4A8_TILE_K_L1",
    "KS_QUANT_DOT_I8_WORKSPACE_BYTES",
    "KS_QUANT_MATVEC_I8_WORKSPACE_BYTES",
    "KS_QUANT_DOT_W4A8_WORKSPACE_BYTES",
    "KS_QUANT_MATVEC_W4A8_WORKSPACE_BYTES",
]


def parse_defines(profile: Path) -> dict[str, str]:
    defines: dict[str, str] = {}
    for line in profile.read_text(encoding="utf-8").splitlines():
        match = DEFINE_RE.match(line)
        if match is None:
            continue
        name, value = match.groups()
        defines[name] = value.strip()
    return defines


def is_rvv_profile(defines: dict[str, str]) -> bool:
    return defines.get("KS_HAS_RVV") == "1" or defines.get("KS_TARGET_ARCH") == "KS_ARCH_RISCV64"


def missing_fields(defines: dict[str, str], required_fields: list[str]) -> list[str]:
    return [field for field in required_fields if field not in defines]


def validate_profile(profile: Path) -> list[str]:
    defines = parse_defines(profile)
    errors = [f"missing required field: {field}" for field in missing_fields(defines, COMMON_REQUIRED_FIELDS)]

    if is_rvv_profile(defines):
        errors.extend(
            f"missing required RVV quantized field: {field}"
            for field in missing_fields(defines, RVV_QUANTIZED_REQUIRED_FIELDS)
        )

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", nargs="+", type=Path)
    args = parser.parse_args()

    failed = False
    for profile in args.profiles:
        errors = validate_profile(profile)
        if not errors:
            print(f"{profile}: ok")
            continue

        failed = True
        print(f"{profile}: invalid", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
