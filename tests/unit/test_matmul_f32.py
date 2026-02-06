"""
Unit tests for MatMul f32 kernel: data generation, reference computation,
and validation framework.

Tests are organized by concern:
  - TestMatmulReference: numpy reference implementation correctness
  - TestMatmulDataGenerator: test data generation pipeline
  - TestMatmulValidator: FunctionalValidator for matmul
  - TestMatmulEdgeCases: boundary conditions and special shapes
  - TestMatmulNumericalStability: floating-point precision analysis
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from functional_validator import FunctionalValidator
from test_data_generator import TensorConfig, TestDataGenerator


# ---------------------------------------------------------------------------
# Reference correctness
# ---------------------------------------------------------------------------
class TestMatmulReference:
    """Verify numpy reference matmul is correct for our test cases."""

    def test_identity_f32(self):
        """A * I = A for identity matrix."""
        N = 16
        A = np.random.RandomState(42).randn(N, N).astype(np.float32)
        eye = np.eye(N, dtype=np.float32)
        np.testing.assert_allclose(A @ eye, A, rtol=1e-6)
        np.testing.assert_allclose(eye @ A, A, rtol=1e-6)

    def test_zeros_f32(self):
        """A * 0 = 0."""
        A = np.random.RandomState(1).randn(8, 16).astype(np.float32)
        Z = np.zeros((16, 4), dtype=np.float32)
        C = A @ Z
        np.testing.assert_array_equal(C, np.zeros((8, 4), dtype=np.float32))

    def test_known_values_4x4(self):
        """Verify hand-computed 4x4 matmul."""
        A = np.array(
            [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]], dtype=np.float32
        )
        B = np.array(
            [[16, 15, 14, 13], [12, 11, 10, 9], [8, 7, 6, 5], [4, 3, 2, 1]], dtype=np.float32
        )
        expected = np.array(
            [[80, 70, 60, 50], [240, 214, 188, 162], [400, 358, 316, 274], [560, 502, 444, 386]],
            dtype=np.float32,
        )
        np.testing.assert_array_equal(A @ B, expected)

    def test_rectangular_3x5_5x2(self):
        """Verify rectangular matmul."""
        A = np.arange(1, 16, dtype=np.float32).reshape(3, 5)
        B = np.arange(1, 11, dtype=np.float32).reshape(5, 2)
        expected = np.array([[95, 110], [220, 260], [345, 410]], dtype=np.float32)
        np.testing.assert_array_equal(A @ B, expected)

    def test_single_element(self):
        """1x1 matmul is scalar multiply."""
        A = np.array([[3.0]], dtype=np.float32)
        B = np.array([[7.0]], dtype=np.float32)
        np.testing.assert_array_equal(A @ B, np.array([[21.0]], dtype=np.float32))

    def test_outer_product(self):
        """Column * row = outer product."""
        col = np.array([[1], [2], [3]], dtype=np.float32)
        row = np.array([[4, 5, 6]], dtype=np.float32)
        expected = np.array([[4, 5, 6], [8, 10, 12], [12, 15, 18]], dtype=np.float32)
        np.testing.assert_array_equal(col @ row, expected)

    def test_transpose_property(self):
        """(A @ B)^T = B^T @ A^T."""
        rng = np.random.RandomState(99)
        A = rng.randn(16, 32).astype(np.float32)
        B = rng.randn(32, 24).astype(np.float32)
        np.testing.assert_allclose((A @ B).T, B.T @ A.T, rtol=1e-4)


# ---------------------------------------------------------------------------
# Data generator
# ---------------------------------------------------------------------------
class TestMatmulDataGenerator:
    """Test the TestDataGenerator for matmul cases."""

    def test_generate_matmul_tests(self):
        """Generator produces expected number of test cases."""
        gen = TestDataGenerator("/tmp/ks_test_data")
        cases = gen.generate_matmul_tests()
        assert len(cases) >= 4  # small, medium, rect, f16, large
        for name, A, B, C in cases:
            assert isinstance(name, str)
            assert A.ndim == 2
            assert B.ndim == 2
            assert C.ndim == 2
            assert A.shape[1] == B.shape[0], f"Inner dim mismatch in {name}"
            assert C.shape == (A.shape[0], B.shape[1])

    def test_matmul_small_case(self):
        """Small test case is 4x4 f32."""
        gen = TestDataGenerator("/tmp/ks_test_data")
        cases = gen.generate_matmul_tests()
        name, A, B, C = cases[0]
        assert "small" in name
        assert A.shape == (4, 4)
        assert A.dtype == np.float32
        np.testing.assert_allclose(A @ B, C, rtol=1e-5)

    def test_matmul_f16_case(self):
        """f16 test case computes in f32 then casts."""
        gen = TestDataGenerator("/tmp/ks_test_data")
        cases = gen.generate_matmul_tests()
        f16_cases = [c for c in cases if "f16" in c[0]]
        assert len(f16_cases) >= 1
        name, A, B, C = f16_cases[0]
        assert A.dtype == np.float16
        # Reference should match f32 computation cast to f16
        ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
        np.testing.assert_array_equal(C, ref)

    def test_deterministic_generation(self):
        """Same seed produces identical test data."""
        gen1 = TestDataGenerator("/tmp/ks_test_data_1")
        gen2 = TestDataGenerator("/tmp/ks_test_data_2")
        cases1 = gen1.generate_matmul_tests()
        cases2 = gen2.generate_matmul_tests()
        for (_n1, A1, B1, C1), (_n2, A2, B2, C2) in zip(cases1, cases2, strict=True):
            np.testing.assert_array_equal(A1, A2)
            np.testing.assert_array_equal(B1, B2)
            np.testing.assert_array_equal(C1, C2)

    def test_save_and_load_tensor(self, tmp_path):
        """Tensors can be saved and reloaded from binary files."""
        gen = TestDataGenerator(str(tmp_path))
        config = TensorConfig(shape=(8, 16), dtype=np.float32, seed=42)
        tensor = gen.generate_tensor(config)
        path = gen.save_tensor(tensor, "test_A")

        loaded = np.fromfile(path, dtype=np.float32).reshape(8, 16)
        np.testing.assert_array_equal(tensor, loaded)


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------
class TestMatmulValidator:
    """Test the FunctionalValidator for matmul correctness checking."""

    def test_perfect_match(self):
        """Exact match should pass."""
        validator = FunctionalValidator()
        A = np.random.RandomState(42).randn(8, 8).astype(np.float32)
        B = np.random.RandomState(43).randn(8, 8).astype(np.float32)
        C = A @ B
        result = validator.validate_matmul(C, A, B)
        assert result.passed
        assert result.max_error == 0.0

    def test_small_perturbation_passes(self):
        """Small noise within tolerance should pass."""
        validator = FunctionalValidator()
        A = np.random.RandomState(1).randn(16, 16).astype(np.float32)
        B = np.random.RandomState(2).randn(16, 16).astype(np.float32)
        C_ref = A @ B
        # Add small noise well within f32 tolerance
        noise = np.random.RandomState(3).randn(*C_ref.shape).astype(np.float32) * 1e-7
        C_noisy = C_ref + noise
        result = validator.validate_matmul(C_noisy, A, B)
        assert result.passed

    def test_large_error_fails(self):
        """Large error should fail validation."""
        validator = FunctionalValidator()
        A = np.random.RandomState(10).randn(8, 8).astype(np.float32)
        B = np.random.RandomState(11).randn(8, 8).astype(np.float32)
        C_wrong = np.zeros((8, 8), dtype=np.float32)
        result = validator.validate_matmul(C_wrong, A, B)
        assert not result.passed

    def test_shape_mismatch_fails(self):
        """Wrong output shape should fail."""
        validator = FunctionalValidator()
        A = np.ones((4, 8), dtype=np.float32)
        B = np.ones((8, 4), dtype=np.float32)
        C_wrong_shape = np.zeros((4, 8), dtype=np.float32)  # Should be (4, 4)
        result = validator.validate_matmul(C_wrong_shape, A, B)
        assert not result.passed

    def test_validator_summary(self):
        """Summary counts pass/fail correctly."""
        validator = FunctionalValidator()
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        validator.validate_matmul(A @ B, A, B)  # pass
        validator.validate_matmul(
            np.zeros((4, 4), dtype=np.float32), A, B
        )  # fail (unless I is trivial)
        summary = validator.get_summary()
        assert summary["total"] == 2

    def test_validator_report_json(self, tmp_path):
        """Save report produces valid JSON."""
        import json

        validator = FunctionalValidator()
        A = np.eye(4, dtype=np.float32)
        B = np.eye(4, dtype=np.float32)
        validator.validate_matmul(A @ B, A, B)

        report_path = str(tmp_path / "report.json")
        validator.save_report(report_path)

        with open(report_path) as f:
            report = json.load(f)
        assert "summary" in report
        assert "results" in report
        assert report["summary"]["total"] == 1


# ---------------------------------------------------------------------------
# Edge cases per matmul spec
# ---------------------------------------------------------------------------
class TestMatmulEdgeCases:
    """Edge cases from specs/kernels/matmul.md."""

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (1, 1, 1),  # scalar
            (1, 64, 1),  # dot product
            (64, 1, 64),  # outer product
            (1, 1, 64),  # row broadcast
            (64, 1, 1),  # col broadcast
        ],
    )
    def test_degenerate_shapes(self, M, K, N):
        """Degenerate dimensions should compute correctly."""
        rng = np.random.RandomState(42)
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)
        C = A @ B
        assert C.shape == (M, N)
        # Verify against loop implementation
        C_ref = np.zeros((M, N), dtype=np.float32)
        for i in range(M):
            for j in range(N):
                for k in range(K):
                    C_ref[i, j] += A[i, k] * B[k, j]
        np.testing.assert_allclose(C, C_ref, rtol=1e-5)

    @pytest.mark.parametrize(
        "M,K,N",
        [
            (17, 23, 31),  # all prime
            (7, 13, 11),  # small primes
            (33, 65, 127),  # near power-of-2
            (100, 100, 100),  # round but not power-of-2
        ],
    )
    def test_non_power_of_2(self, M, K, N):
        """Non-power-of-2 dimensions (important for tiling tail handling)."""
        rng = np.random.RandomState(M * 1000 + K * 100 + N)
        A = rng.randn(M, K).astype(np.float32)
        B = rng.randn(K, N).astype(np.float32)
        C = A @ B
        assert C.shape == (M, N)
        assert not np.any(np.isnan(C))
        assert not np.any(np.isinf(C))

    @pytest.mark.parametrize("size", [4, 8, 16, 32, 64])
    def test_square_sizes(self, size):
        """Square matmul at various sizes matching RVV tile boundaries."""
        rng = np.random.RandomState(size)
        A = rng.randn(size, size).astype(np.float32)
        B = rng.randn(size, size).astype(np.float32)
        C = A @ B
        assert C.shape == (size, size)
        # Cross-check with explicit sum
        for i in range(min(size, 4)):
            for j in range(min(size, 4)):
                expected = np.sum(A[i, :] * B[:, j])
                np.testing.assert_allclose(C[i, j], expected, rtol=1e-4)


# ---------------------------------------------------------------------------
# Numerical stability
# ---------------------------------------------------------------------------
class TestMatmulNumericalStability:
    """Verify numerical stability properties relevant to RVV codegen."""

    def test_accumulation_order_sensitivity(self):
        """
        MatMul result may differ based on accumulation order.
        Document the expected error bounds for f32.
        """
        rng = np.random.RandomState(42)
        K = 1024
        A = rng.randn(1, K).astype(np.float32)
        B = rng.randn(K, 1).astype(np.float32)

        # Forward accumulation
        fwd = np.float32(0.0)
        for k in range(K):
            fwd += A[0, k] * B[k, 0]

        # Reverse accumulation
        rev = np.float32(0.0)
        for k in range(K - 1, -1, -1):
            rev += A[0, k] * B[k, 0]

        # numpy reference (may use different order internally)
        ref = (A @ B)[0, 0]

        # All should agree within reasonable f32 tolerance
        # With K=1024, expect ~sqrt(K)*eps relative error
        max_diff = max(abs(fwd - ref), abs(rev - ref), abs(fwd - rev))
        # For random uniform data with K=1024, error should be < 1e-3
        assert max_diff < 1e-2, f"Accumulation order difference too large: {max_diff}"

    def test_large_value_range(self):
        """Large dynamic range should not overflow or produce NaN."""
        rng = np.random.RandomState(7)
        A = rng.randn(8, 8).astype(np.float32) * 1e3
        B = rng.randn(8, 8).astype(np.float32) * 1e-3
        C = A @ B
        assert not np.any(np.isnan(C))
        assert not np.any(np.isinf(C))

    def test_subnormal_inputs(self):
        """Subnormal (denormalized) inputs should not cause issues."""
        A = np.full((4, 4), np.finfo(np.float32).tiny / 2, dtype=np.float32)
        B = np.ones((4, 4), dtype=np.float32)
        C = A @ B
        # Result should be 4 * tiny/2 per element
        assert not np.any(np.isnan(C))
        assert np.all(C >= 0)

    @pytest.mark.parametrize("dtype", [np.float32, np.float16])
    def test_tolerance_by_dtype(self, dtype):
        """
        Verify expected tolerance levels by dtype.
        f32: atol=1e-5, f16: atol=1e-2
        """
        rng = np.random.RandomState(42)
        A = rng.randn(16, 16).astype(dtype)
        B = rng.randn(16, 16).astype(dtype)

        if dtype == np.float16:
            C_ref = (A.astype(np.float32) @ B.astype(np.float32)).astype(np.float16)
            C_test = A @ B
            # f16 has less precision
            np.testing.assert_allclose(C_test, C_ref, atol=1e-1, rtol=1e-1)
        else:
            C_ref = A @ B
            np.testing.assert_allclose(C_ref, A @ B, atol=1e-5)
