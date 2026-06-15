#!/usr/bin/env python3
"""Check that an RVV profile build can link external INT8 API objects."""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
from pathlib import Path


def run(command: list[str], cwd: Path | None = None) -> None:
    print("+", " ".join(shlex.quote(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", required=True, type=Path)
    parser.add_argument("--build-dir", required=True, type=Path)
    parser.add_argument("--cmake", required=True)
    parser.add_argument("--ctest", required=True)
    parser.add_argument("--c-compiler", required=True)
    parser.add_argument("--cxx-compiler", required=True)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--mlir-dir", required=True, type=Path)
    parser.add_argument("--llvm-dir", required=True, type=Path)
    args = parser.parse_args()

    source_dir = args.source_dir.resolve()
    build_dir = args.build_dir.resolve()
    object_dir = build_dir / "objects"
    mock_source = source_dir / "tests" / "capi" / "int8_generated_override.c"
    mock_object = object_dir / "int8_generated_override.o"

    if build_dir.exists():
        shutil.rmtree(build_dir)
    object_dir.mkdir(parents=True)

    run(
        [
            args.c_compiler,
            "-std=c99",
            "-c",
            str(mock_source),
            "-I",
            str(source_dir / "include"),
            "-o",
            str(mock_object),
        ]
    )

    run(
        [
            args.cmake,
            "-S",
            str(source_dir),
            "-B",
            str(build_dir),
            "-G",
            args.generator,
            f"-DCMAKE_C_COMPILER={args.c_compiler}",
            f"-DCMAKE_CXX_COMPILER={args.cxx_compiler}",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DKS_BUILD_EXAMPLES=OFF",
            "-DKS_ENABLE_TESTS=ON",
            "-DKS_TARGET_PROFILE=riscv_rvv_256",
            f"-DKS_INT8_RVV_OBJECTS={mock_object}",
            f"-DMLIR_DIR={args.mlir_dir}",
            f"-DLLVM_DIR={args.llvm_dir}",
        ]
    )

    run(
        [
            args.cmake,
            "--build",
            str(build_dir),
            "--target",
            "test-quantized-generated-override",
            "--parallel",
            "2",
        ]
    )
    run(
        [
            args.ctest,
            "--test-dir",
            str(build_dir),
            "--output-on-failure",
            "-R",
            "^capi-quantized-generated-override$",
        ]
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
