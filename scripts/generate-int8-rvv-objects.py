#!/usr/bin/env python3
"""Generate and compile INT8 RVV objects for the public quantized C API."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

COMMON_SOURCE = r"""
#include "kernelsmith/ks_common.h"
#include "kernelsmith/ks_quantized.h"

#include <riscv_vector.h>
#include <stddef.h>
#include <stdint.h>

static int ks_i8_zero_point_valid(int32_t zero_point) {
  return zero_point >= -128 && zero_point <= 127;
}

static int32_t ks_rvv_dot_i8_impl(const int8_t *input, const int8_t *weight,
                                  size_t k, int32_t input_zero_point,
                                  int32_t weight_zero_point) {
  int32_t accumulator = 0;

  while (k != 0) {
    size_t vl = __riscv_vsetvl_e8m1(k);
    vint8m1_t input_i8 = __riscv_vle8_v_i8m1(input, vl);
    vint8m1_t weight_i8 = __riscv_vle8_v_i8m1(weight, vl);
    vint16m2_t input_i16 = __riscv_vsext_vf2_i16m2(input_i8, vl);
    vint16m2_t weight_i16 = __riscv_vsext_vf2_i16m2(weight_i8, vl);

    input_i16 = __riscv_vsub_vx_i16m2(input_i16, input_zero_point, vl);
    weight_i16 = __riscv_vsub_vx_i16m2(weight_i16, weight_zero_point, vl);

    vint32m4_t product = __riscv_vwmul_vv_i32m4(input_i16, weight_i16, vl);
    vint32m1_t zero =
        __riscv_vmv_v_x_i32m1(0, __riscv_vsetvl_e32m1(1));
    vint32m1_t partial = __riscv_vredsum_vs_i32m4_i32m1(product, zero, vl);
    accumulator += __riscv_vmv_x_s_i32m1_i32(partial);

    input += vl;
    weight += vl;
    k -= vl;
  }

  return accumulator;
}
"""


DOT_SOURCE = (
    COMMON_SOURCE
    + r"""
int ks_dot_i8(const int8_t *input, const int8_t *weight, size_t k,
              int32_t input_zero_point, int32_t weight_zero_point,
              int32_t *output, void *workspace, size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !weight || !output)
    return KS_ERR_INVALID_ARG;
  if (!ks_i8_zero_point_valid(input_zero_point) ||
      !ks_i8_zero_point_valid(weight_zero_point))
    return KS_ERR_INVALID_ARG;

  *output = ks_rvv_dot_i8_impl(input, weight, k, input_zero_point,
                               weight_zero_point);
  return KS_OK;
}
"""
)


MATVEC_SOURCE = (
    COMMON_SOURCE
    + r"""
int ks_matvec_i8(const int8_t *input, const int8_t *weights,
                 size_t weight_stride, int32_t *output, size_t rows,
                 size_t cols, int32_t input_zero_point,
                 int32_t weight_zero_point, void *workspace, size_t ws_size) {
  (void)workspace;
  (void)ws_size;

  if (!input || !weights || !output)
    return KS_ERR_INVALID_ARG;
  if (rows == 0 || cols == 0 || weight_stride < cols)
    return KS_ERR_INVALID_ARG;
  if (!ks_i8_zero_point_valid(input_zero_point) ||
      !ks_i8_zero_point_valid(weight_zero_point))
    return KS_ERR_INVALID_ARG;

  for (size_t row = 0; row < rows; ++row) {
    const int8_t *weight_row = weights + row * weight_stride;
    output[row] = ks_rvv_dot_i8_impl(input, weight_row, cols, input_zero_point,
                                     weight_zero_point);
  }

  return KS_OK;
}
"""
)


OBJECTS = (
    ("ks_dot_i8_riscv_rvv_256", DOT_SOURCE),
    ("ks_matvec_i8_riscv_rvv_256", MATVEC_SOURCE),
)


def run(command: list[str]) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, check=True)


def write_if_changed(path: Path, contents: str) -> None:
    if path.exists() and path.read_text() == contents:
        return
    path.write_text(contents)


def compile_object(
    *,
    compiler: str,
    source_root: Path,
    output_dir: Path,
    name: str,
    source: str,
    extra_cflags: list[str],
) -> Path:
    source_path = output_dir / f"{name}.c"
    object_path = output_dir / f"{name}.o"
    write_if_changed(source_path, source.lstrip())
    run(
        [
            compiler,
            "-std=c99",
            "-O2",
            "-march=rv64gcv",
            "-mabi=lp64d",
            "-I",
            str(source_root / "include"),
            "-include",
            str(source_root / "target" / "riscv_rvv_256.h"),
            *extra_cflags,
            "-c",
            str(source_path),
            "-o",
            str(object_path),
        ]
    )
    return object_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cc", default="riscv64-linux-gnu-gcc")
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--extra-cflag",
        action="append",
        default=[],
        help="Additional C compiler flag, may be repeated.",
    )
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    object_paths = [
        compile_object(
            compiler=args.cc,
            source_root=source_root,
            output_dir=output_dir,
            name=name,
            source=source,
            extra_cflags=args.extra_cflag,
        )
        for name, source in OBJECTS
    ]

    for object_path in object_paths:
        print(object_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
