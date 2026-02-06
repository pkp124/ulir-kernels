"""
Integration test: full pipeline for MatMul f32 on RISC-V with RVV.

Pipeline stages tested:
  1. Generate test data (numpy)
  2. Cross-compile C matmul kernel for RISC-V (riscv64-linux-gnu-gcc)
  3. Execute on QEMU with multiple VLEN configurations
  4. Compare RISC-V output against numpy reference
  5. Validate numerical accuracy within tolerance

This exercises the same pipeline that KernelSmith's generated code will
follow once lowering passes are implemented:
  ks.matmul -> linalg -> tiled -> vectorized -> RVV -> binary -> QEMU
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "tests"))

from functional_validator import FunctionalValidator  # noqa: E402
from test_data_generator import TestDataGenerator  # noqa: E402

# Paths
KERNEL_SRC = PROJECT_ROOT / "tests" / "integration" / "riscv_kernels" / "matmul_f32.c"

# Tool availability
RISCV_GCC = shutil.which("riscv64-linux-gnu-gcc")
QEMU_RISCV = shutil.which("qemu-riscv64")

SKIP_NO_TOOLCHAIN = pytest.mark.skipif(
    RISCV_GCC is None or QEMU_RISCV is None,
    reason="RISC-V toolchain (riscv64-linux-gnu-gcc + qemu-riscv64) not available",
)

# Standard VLEN configurations to test (per RVV spec)
STANDARD_VLENS = [128, 256, 512]

# Tolerances by size (larger matrices accumulate more FP error)
TOLERANCE_MAP = {
    "small": {"atol": 1e-6, "rtol": 1e-5},  # <=8x8
    "medium": {"atol": 1e-4, "rtol": 1e-3},  # <=64x64
    "large": {"atol": 1e-3, "rtol": 1e-2},  # >64x64
}


def get_tolerance(M: int, N: int, K: int) -> dict:
    """Select tolerance based on problem size."""
    max_dim = max(M, N, K)
    if max_dim <= 8:
        return TOLERANCE_MAP["small"]
    elif max_dim <= 64:
        return TOLERANCE_MAP["medium"]
    return TOLERANCE_MAP["large"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def build_dir():
    """Temporary directory for compiled binaries."""
    d = tempfile.mkdtemp(prefix="ks_riscv_test_")
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="session")
def compiled_binary(build_dir) -> Path | None:
    """Cross-compile the matmul kernel for RISC-V."""
    if not RISCV_GCC:
        return None

    binary = build_dir / "matmul_f32_riscv"
    result = subprocess.run(
        [
            RISCV_GCC,
            "-static",
            "-O2",
            "-march=rv64gcv",
            "-mabi=lp64d",
            "-o",
            str(binary),
            str(KERNEL_SRC),
            "-lm",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        pytest.skip(f"RISC-V compilation failed: {result.stderr}")
    return binary


@pytest.fixture(scope="session")
def test_data_dir(build_dir):
    """Generate test data files."""
    data_dir = build_dir / "test_data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def save_f32_bin(path: Path, arr: np.ndarray):
    """Save f32 numpy array as raw binary."""
    arr.astype(np.float32).tofile(str(path))


def load_f32_bin(path: Path, shape: tuple) -> np.ndarray:
    """Load f32 raw binary into numpy array."""
    return np.fromfile(str(path), dtype=np.float32).reshape(shape)


def run_on_qemu(
    binary: Path, args: list, vlen: int = 256, timeout: int = 30
) -> subprocess.CompletedProcess:
    """Execute a RISC-V binary on QEMU with specific VLEN."""
    cmd = [
        QEMU_RISCV,
        "-cpu",
        f"rv64,v=true,vlen={vlen}",
        str(binary),
    ] + [str(a) for a in args]

    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


def run_matmul_on_qemu(
    binary: Path,
    A: np.ndarray,
    B: np.ndarray,
    data_dir: Path,
    vlen: int = 256,
) -> np.ndarray:
    """
    Run matmul kernel on QEMU and return the result matrix.

    Steps:
      1. Write A and B as binary files
      2. Run the compiled kernel on QEMU
      3. Read the output C binary
      4. Return as numpy array
    """
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"Inner dimension mismatch: {K} != {K2}"

    a_path = data_dir / f"A_{M}x{K}.bin"
    b_path = data_dir / f"B_{K}x{N}.bin"
    c_path = data_dir / f"C_{M}x{N}_vlen{vlen}.bin"

    save_f32_bin(a_path, A)
    save_f32_bin(b_path, B)

    result = run_on_qemu(
        binary,
        [str(M), str(N), str(K), str(a_path), str(b_path), str(c_path)],
        vlen=vlen,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"QEMU execution failed (VLEN={vlen}):\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    assert c_path.exists(), f"Output file not created: {c_path}"
    return load_f32_bin(c_path, (M, N))


# ---------------------------------------------------------------------------
# Test: Self-test mode (built-in validation)
# ---------------------------------------------------------------------------
@SKIP_NO_TOOLCHAIN
class TestMatmulSelfTest:
    """Run the kernel's built-in self-test on QEMU."""

    @pytest.mark.parametrize("vlen", STANDARD_VLENS)
    def test_self_test_passes(self, compiled_binary, vlen):
        """Built-in self-test should pass for all VLEN configurations."""
        result = run_on_qemu(compiled_binary, ["--self-test"], vlen=vlen)
        assert result.returncode == 0, (
            f"Self-test failed with VLEN={vlen}:\n{result.stdout}\n{result.stderr}"
        )
        assert "ALL PASSED" in result.stdout


# ---------------------------------------------------------------------------
# Test: End-to-end with numpy reference validation
# ---------------------------------------------------------------------------
@SKIP_NO_TOOLCHAIN
class TestMatmulEndToEnd:
    """Full pipeline: generate data -> run on QEMU -> validate against numpy."""

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (4, 4, 4),  # Smallest: fits in registers
            (8, 8, 8),  # Small square
            (16, 16, 16),  # Single RVV tile (VLEN=256, f32)
        ],
    )
    @pytest.mark.parametrize("vlen", STANDARD_VLENS)
    def test_square_matmul(self, compiled_binary, test_data_dir, M, K, N, vlen):
        """Square matmul at various sizes across VLEN configs."""
        rng = np.random.RandomState(M * 100 + K * 10 + N)
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)

        C_qemu = run_matmul_on_qemu(compiled_binary, A, B, test_data_dir, vlen=vlen)
        C_ref = A @ B

        tol = get_tolerance(M, N, K)
        np.testing.assert_allclose(
            C_qemu,
            C_ref,
            atol=tol["atol"],
            rtol=tol["rtol"],
            err_msg=f"Matmul {M}x{K} @ {K}x{N} failed with VLEN={vlen}",
        )

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (3, 5, 2),  # All small, non-square
            (7, 13, 11),  # Prime dimensions
            (17, 23, 31),  # Larger primes (tail handling test)
        ],
    )
    @pytest.mark.parametrize("vlen", [256])  # Test one VLEN for non-square
    def test_rectangular_matmul(self, compiled_binary, test_data_dir, M, K, N, vlen):
        """Rectangular matmul with non-power-of-2 dimensions."""
        rng = np.random.RandomState(M * 1000 + K * 100 + N)
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)

        C_qemu = run_matmul_on_qemu(compiled_binary, A, B, test_data_dir, vlen=vlen)
        C_ref = A @ B

        tol = get_tolerance(M, N, K)
        np.testing.assert_allclose(
            C_qemu,
            C_ref,
            atol=tol["atol"],
            rtol=tol["rtol"],
            err_msg=f"Rectangular matmul {M}x{K} @ {K}x{N} failed",
        )

    def test_identity_matmul(self, compiled_binary, test_data_dir):
        """A * I = A property test."""
        N = 16
        rng = np.random.RandomState(42)
        A = rng.randn(N, N).astype(np.float32)
        eye = np.eye(N, dtype=np.float32)

        C_qemu = run_matmul_on_qemu(compiled_binary, A, eye, test_data_dir, vlen=256)
        np.testing.assert_allclose(C_qemu, A, atol=1e-6, rtol=1e-5)

    def test_zero_matmul(self, compiled_binary, test_data_dir):
        """A * 0 = 0 property test."""
        A = np.random.RandomState(1).randn(8, 16).astype(np.float32)
        Z = np.zeros((16, 4), dtype=np.float32)

        C_qemu = run_matmul_on_qemu(compiled_binary, A, Z, test_data_dir, vlen=256)
        np.testing.assert_array_equal(C_qemu, np.zeros((8, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# Test: Multi-VLEN consistency
# ---------------------------------------------------------------------------
@SKIP_NO_TOOLCHAIN
class TestMatmulMultiVLEN:
    """Verify results are consistent across different VLEN configurations."""

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (4, 4, 4),
            (16, 16, 16),
            (32, 32, 32),
        ],
    )
    def test_cross_vlen_consistency(self, compiled_binary, test_data_dir, M, K, N):
        """
        Same inputs should produce identical (or nearly identical) outputs
        regardless of VLEN. This is critical for VLA correctness.
        """
        rng = np.random.RandomState(M * 100 + N)
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)

        results = {}
        for vlen in STANDARD_VLENS:
            C = run_matmul_on_qemu(compiled_binary, A, B, test_data_dir, vlen=vlen)
            results[vlen] = C

        # All VLENs should produce the same result
        baseline = results[STANDARD_VLENS[0]]
        for vlen in STANDARD_VLENS[1:]:
            np.testing.assert_allclose(
                results[vlen],
                baseline,
                atol=1e-6,
                rtol=1e-5,
                err_msg=f"VLEN={vlen} differs from VLEN={STANDARD_VLENS[0]} for {M}x{K}x{N} matmul",
            )


# ---------------------------------------------------------------------------
# Test: Validation framework integration
# ---------------------------------------------------------------------------
@SKIP_NO_TOOLCHAIN
class TestMatmulValidationFramework:
    """Use the FunctionalValidator to validate QEMU results."""

    def test_validator_integration(self, compiled_binary, test_data_dir):
        """
        End-to-end test using the project's FunctionalValidator.
        This mirrors the validation pattern for compiler-generated code.
        """
        validator = FunctionalValidator(verbose=True)

        test_cases = [
            ("4x4_square", 4, 4, 4, 42),
            ("8x8_square", 8, 8, 8, 43),
            ("16x16_square", 16, 16, 16, 44),
            ("3x5x2_rect", 3, 5, 2, 45),
            ("7x13x11_prime", 7, 13, 11, 46),
        ]

        for name, M, K, N, seed in test_cases:
            rng = np.random.RandomState(seed)
            A = rng.randn(M, K).astype(np.float32)
            B = rng.randn(K, N).astype(np.float32)

            C_qemu = run_matmul_on_qemu(compiled_binary, A, B, test_data_dir, vlen=256)

            result = validator.validate_matmul(C_qemu, A, B)
            assert result.passed, (
                f"Validation failed for {name}: "
                f"max_error={result.max_error:.2e}, "
                f"mean_error={result.mean_error:.2e}"
            )

        summary = validator.get_summary()
        assert summary["pass_rate"] == 1.0, f"Not all tests passed: {summary}"

    def test_generated_test_data_pipeline(self, compiled_binary, test_data_dir):
        """
        Use TestDataGenerator to create data, run on QEMU,
        validate with FunctionalValidator.
        """
        gen = TestDataGenerator(str(test_data_dir / "generated"))
        validator = FunctionalValidator(verbose=True)

        cases = gen.generate_matmul_tests()
        # Test the small and medium cases (skip large for speed)
        for name, A, B, _C_ref in cases[:3]:
            if A.dtype != np.float32:
                continue  # Skip f16 for this RISC-V C kernel

            C_qemu = run_matmul_on_qemu(compiled_binary, A, B, test_data_dir, vlen=256)

            result = validator.validate_matmul(C_qemu, A, B)
            assert result.passed, f"Failed for {name}: max_error={result.max_error:.2e}"


# ---------------------------------------------------------------------------
# Test: Compilation and toolchain
# ---------------------------------------------------------------------------
@SKIP_NO_TOOLCHAIN
class TestRISCVToolchain:
    """Verify the RISC-V cross-compilation toolchain works correctly."""

    def test_kernel_source_exists(self):
        """C source file exists."""
        assert KERNEL_SRC.exists(), f"Kernel source not found: {KERNEL_SRC}"

    def test_compilation_succeeds(self, compiled_binary):
        """Cross-compilation produces a valid binary."""
        assert compiled_binary is not None
        assert compiled_binary.exists()

    def test_binary_is_riscv(self, compiled_binary):
        """Compiled binary is actually RISC-V."""
        result = subprocess.run(
            ["file", str(compiled_binary)],
            capture_output=True,
            text=True,
        )
        assert "RISC-V" in result.stdout, f"Not a RISC-V binary: {result.stdout}"

    def test_binary_is_statically_linked(self, compiled_binary):
        """Binary is statically linked (required for QEMU user mode)."""
        result = subprocess.run(
            ["file", str(compiled_binary)],
            capture_output=True,
            text=True,
        )
        assert "statically linked" in result.stdout

    @pytest.mark.parametrize("opt_level", ["-O0", "-O1", "-O2", "-O3"])
    def test_compilation_at_different_opt_levels(self, build_dir, opt_level):
        """Kernel compiles at all optimization levels."""
        binary = build_dir / f"matmul_f32_{opt_level.replace('-', '')}"
        result = subprocess.run(
            [
                RISCV_GCC,
                "-static",
                opt_level,
                "-march=rv64gcv",
                "-mabi=lp64d",
                "-o",
                str(binary),
                str(KERNEL_SRC),
                "-lm",
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"Compilation failed at {opt_level}: {result.stderr}"

        # Run self-test to verify correctness at each opt level
        qemu_result = run_on_qemu(binary, ["--self-test"], vlen=256)
        assert qemu_result.returncode == 0, (
            f"Self-test failed at {opt_level}:\n{qemu_result.stdout}"
        )
        assert "ALL PASSED" in qemu_result.stdout
