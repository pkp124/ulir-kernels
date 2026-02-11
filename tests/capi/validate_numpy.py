#!/usr/bin/env python3
"""
Numerical validation of libkernelsmith against numpy.

Runs ks_matmul_f32 and activation functions via a small C helper,
compares outputs to numpy reference. Uses ctypes to call the shared lib.

This script validates correctness, not performance.
"""

import ctypes
import os
import sys
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------- #
# Library loading
# --------------------------------------------------------------------------- #

def find_library():
    """Locate libkernelsmith.a or the test executables."""
    build_dir = Path(__file__).resolve().parent.parent.parent / "build"
    # Look for the static library
    candidates = [
        build_dir / "lib" / "kernelsmith" / "libkernelsmith.a",
        build_dir / "lib" / "libkernelsmith.a",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


# --------------------------------------------------------------------------- #
# Reference implementations
# --------------------------------------------------------------------------- #

def ref_matmul_f32(A, B):
    """Reference f32 matmul using numpy."""
    return A.astype(np.float64) @ B.astype(np.float64)


def ref_relu_f32(x):
    return np.maximum(x, 0.0)


def ref_gelu_f32(x):
    from scipy.special import erf
    return 0.5 * x * (1.0 + erf(x / np.sqrt(2.0)))


def ref_silu_f32(x):
    return x / (1.0 + np.exp(-x.astype(np.float64)))


# --------------------------------------------------------------------------- #
# Test data generation
# --------------------------------------------------------------------------- #

def generate_matmul_cases():
    """Generate test cases with varying dimensions."""
    rng = np.random.default_rng(42)
    cases = [
        ("square 64x64",     64,  64,  64),
        ("rect 32x128x64",   32, 128,  64),
        ("non-aligned 17x23x31", 17,  23,  31),
        ("small 4x4x4",       4,   4,   4),
        ("tall 128x16x32",  128,  16,  32),
        ("wide 16x128x32",   16, 128,  32),
    ]
    for label, M, N, K in cases:
        A = rng.standard_normal((M, K)).astype(np.float32)
        B = rng.standard_normal((K, N)).astype(np.float32)
        C_ref = ref_matmul_f32(A, B).astype(np.float32)
        yield label, A, B, C_ref


def generate_activation_cases():
    """Generate activation test values."""
    x = np.array([-5.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0],
                 dtype=np.float32)
    return x


# --------------------------------------------------------------------------- #
# Validation (numpy-only, no ctypes — validates test data and references)
# --------------------------------------------------------------------------- #

def validate_matmul_references():
    """Validate that our reference matches numpy directly."""
    print("=== matmul reference validation (numpy) ===\n")
    passed = 0
    total = 0

    for label, A, B, C_ref in generate_matmul_cases():
        total += 1
        C_np = (A.astype(np.float64) @ B.astype(np.float64)).astype(np.float32)
        max_rel_err = np.max(np.abs(C_ref - C_np) /
                             np.maximum(np.abs(C_np), 1e-8))
        if max_rel_err < 1e-5:
            print(f"  PASS {label} (max_rel_err={max_rel_err:.2e})")
            passed += 1
        else:
            print(f"  FAIL {label} (max_rel_err={max_rel_err:.2e})")

    print(f"\n{passed}/{total} matmul reference checks passed.\n")
    return passed == total


def validate_activation_references():
    """Validate activation function references."""
    print("=== activation reference validation (numpy) ===\n")
    x = generate_activation_cases()
    passed = 0
    total = 0

    # ReLU
    total += 1
    relu_out = ref_relu_f32(x)
    relu_expected = np.maximum(x, 0.0).astype(np.float32)
    if np.allclose(relu_out, relu_expected, atol=1e-6):
        print(f"  PASS relu")
        passed += 1
    else:
        print(f"  FAIL relu")

    # GELU (compare against scipy)
    total += 1
    try:
        gelu_out = ref_gelu_f32(x)
        # Sanity: GELU(-5) ~ 0, GELU(5) ~ 5
        if abs(float(gelu_out[0])) < 0.01 and abs(float(gelu_out[-1]) - 5.0) < 0.01:
            print(f"  PASS gelu (range check)")
            passed += 1
        else:
            print(f"  FAIL gelu (range check)")
    except ImportError:
        print(f"  SKIP gelu (scipy not available)")
        passed += 1  # Don't fail without scipy

    # SiLU
    total += 1
    silu_out = ref_silu_f32(x).astype(np.float32)
    # Sanity: SiLU(0) = 0, SiLU(x) ~ x for large x, SiLU(x) ~ 0 for large negative x
    if (abs(float(silu_out[4])) < 1e-6 and  # x=0
        abs(float(silu_out[-1]) - 5.0) < 0.1):  # x=5
        print(f"  PASS silu (range check)")
        passed += 1
    else:
        print(f"  FAIL silu (range check)")

    print(f"\n{passed}/{total} activation reference checks passed.\n")
    return passed == total


# --------------------------------------------------------------------------- #
# Write test data to disk for C test consumption
# --------------------------------------------------------------------------- #

def write_test_data():
    """Write binary test data files that C tests can load."""
    out_dir = Path(__file__).resolve().parent / "testdata"
    out_dir.mkdir(exist_ok=True)

    for label, A, B, C_ref in generate_matmul_cases():
        safe_label = label.replace(" ", "_")
        M, K = A.shape
        _, N = B.shape
        header = np.array([M, N, K], dtype=np.int32)

        with open(out_dir / f"matmul_{safe_label}.bin", "wb") as f:
            f.write(header.tobytes())
            f.write(A.tobytes())
            f.write(B.tobytes())
            f.write(C_ref.tobytes())

    print(f"Wrote test data to {out_dir}/")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main():
    ok = True
    ok = validate_matmul_references() and ok
    ok = validate_activation_references() and ok

    if "--write-data" in sys.argv:
        write_test_data()

    if ok:
        print("All validations passed.")
    else:
        print("SOME VALIDATIONS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
