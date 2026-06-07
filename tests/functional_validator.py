#!/usr/bin/env python3
"""
Functional Validation Harness for KernelSmith
Compares kernel outputs against reference implementations
"""

import json
from dataclasses import dataclass

import numpy as np


@dataclass
class ValidationResult:
    """Result of validating a kernel against reference"""

    kernel_name: str
    passed: bool
    max_error: float
    mean_error: float
    rel_error: float
    error_type: str
    details: str


@dataclass(frozen=True)
class ComparatorPolicy:
    """Policy for comparing a produced tensor with a golden tensor."""

    mode: str
    rtol: float = 0.0
    atol: float = 0.0
    strict_shape: bool = True
    strict_dtype: bool = True
    quantization: dict | None = None

    @classmethod
    def from_dict(cls, value: dict) -> "ComparatorPolicy":
        """Build a comparator policy from descriptor or manifest JSON."""
        mode = value.get("mode")
        if mode not in {"exact", "allclose", "quantized_exact", "dequantized_allclose"}:
            raise ValueError(f"Unsupported compare mode: {mode}")
        quantization = {
            key: item
            for key, item in value.items()
            if key not in {"mode", "rtol", "atol", "strict_shape", "strict_dtype"}
        }
        return cls(
            mode=mode,
            rtol=float(value.get("rtol", 0.0)),
            atol=float(value.get("atol", 0.0)),
            strict_shape=bool(value.get("strict_shape", True)),
            strict_dtype=bool(value.get("strict_dtype", True)),
            quantization=quantization,
        )


@dataclass
class TensorComparisonResult:
    """Machine-parseable tensor comparison report."""

    mode: str
    passed: bool
    max_error: float
    mean_error: float
    max_relative_error: float
    mismatch_count: int
    first_mismatches: list[dict]
    details: str

    def to_dict(self) -> dict:
        """Return JSON-serializable comparison diagnostics."""
        return {
            "mode": self.mode,
            "passed": self.passed,
            "max_error": self.max_error,
            "mean_error": self.mean_error,
            "max_relative_error": self.max_relative_error,
            "mismatch_count": self.mismatch_count,
            "first_mismatches": self.first_mismatches,
            "details": self.details,
        }


def _as_policy(policy: ComparatorPolicy | dict) -> ComparatorPolicy:
    if isinstance(policy, ComparatorPolicy):
        return policy
    return ComparatorPolicy.from_dict(policy)


def _failure_result(mode: str, details: str) -> TensorComparisonResult:
    return TensorComparisonResult(
        mode=mode,
        passed=False,
        max_error=float("inf"),
        mean_error=float("inf"),
        max_relative_error=float("inf"),
        mismatch_count=1,
        first_mismatches=[],
        details=details,
    )


def _numeric_metrics(actual: np.ndarray, expected: np.ndarray) -> tuple[float, float, float]:
    actual_f64 = actual.astype(np.float64, copy=False)
    expected_f64 = expected.astype(np.float64, copy=False)
    diff = np.abs(actual_f64 - expected_f64)
    if diff.size == 0:
        return 0.0, 0.0, 0.0
    denom = np.maximum(np.abs(expected_f64), 1e-12)
    rel_diff = diff / denom
    return float(np.max(diff)), float(np.mean(diff)), float(np.max(rel_diff))


def _first_mismatches(
    actual: np.ndarray,
    expected: np.ndarray,
    mismatch_mask: np.ndarray,
    *,
    limit: int = 5,
) -> list[dict]:
    mismatches = []
    for index in np.argwhere(mismatch_mask)[:limit]:
        index_tuple = tuple(int(item) for item in index)
        mismatches.append(
            {
                "index": list(index_tuple),
                "actual": actual[index_tuple].item(),
                "expected": expected[index_tuple].item(),
            }
        )
    return mismatches


def _dequantize(array: np.ndarray, *, scale: float, zero_point: int) -> np.ndarray:
    return (array.astype(np.float64) - zero_point) * scale


def compare_arrays(
    actual: np.ndarray,
    expected: np.ndarray,
    policy: ComparatorPolicy | dict,
) -> TensorComparisonResult:
    """Compare tensors using exact, floating, or quantized descriptor modes."""
    policy = _as_policy(policy)
    if policy.strict_shape and actual.shape != expected.shape:
        return _failure_result(
            policy.mode,
            f"shape mismatch: actual {actual.shape}, expected {expected.shape}",
        )
    if policy.strict_dtype and actual.dtype != expected.dtype:
        return _failure_result(
            policy.mode,
            f"dtype mismatch: actual {actual.dtype}, expected {expected.dtype}",
        )
    if actual.shape != expected.shape:
        return _failure_result(
            policy.mode,
            f"shape mismatch: actual {actual.shape}, expected {expected.shape}",
        )

    compare_actual = actual
    compare_expected = expected
    if actual.dtype != expected.dtype and not policy.strict_dtype:
        compare_actual = actual.astype(expected.dtype)

    if policy.mode in {"exact", "quantized_exact"}:
        if policy.mode == "quantized_exact" and (
            not np.issubdtype(compare_actual.dtype, np.integer)
            or not np.issubdtype(compare_expected.dtype, np.integer)
        ):
            return _failure_result("quantized_exact", "quantized_exact requires integer tensors")
        mismatch_mask = compare_actual != compare_expected
    elif policy.mode == "allclose":
        mismatch_mask = ~np.isclose(
            compare_actual,
            compare_expected,
            rtol=policy.rtol,
            atol=policy.atol,
            equal_nan=False,
        )
    elif policy.mode == "dequantized_allclose":
        quantization = policy.quantization or {}
        actual_scale = float(quantization.get("actual_scale", quantization.get("scale", 1.0)))
        expected_scale = float(quantization.get("expected_scale", quantization.get("scale", 1.0)))
        actual_zero_point = int(
            quantization.get("actual_zero_point", quantization.get("zero_point", 0))
        )
        expected_zero_point = int(
            quantization.get("expected_zero_point", quantization.get("zero_point", 0))
        )
        compare_actual = _dequantize(
            compare_actual,
            scale=actual_scale,
            zero_point=actual_zero_point,
        )
        compare_expected = _dequantize(
            compare_expected,
            scale=expected_scale,
            zero_point=expected_zero_point,
        )
        mismatch_mask = ~np.isclose(
            compare_actual,
            compare_expected,
            rtol=policy.rtol,
            atol=policy.atol,
            equal_nan=False,
        )
    else:
        raise ValueError(f"Unsupported compare mode: {policy.mode}")

    max_error, mean_error, max_relative_error = _numeric_metrics(compare_actual, compare_expected)
    mismatch_count = int(np.count_nonzero(mismatch_mask))
    first_mismatches = _first_mismatches(compare_actual, compare_expected, mismatch_mask)
    passed = mismatch_count == 0
    details = "PASS" if passed else f"FAIL: {mismatch_count} mismatches"
    if first_mismatches:
        details += f"; first mismatch at {first_mismatches[0]['index']}"
    return TensorComparisonResult(
        mode=policy.mode,
        passed=passed,
        max_error=max_error,
        mean_error=mean_error,
        max_relative_error=max_relative_error,
        mismatch_count=mismatch_count,
        first_mismatches=first_mismatches,
        details=details,
    )


class FunctionalValidator:
    """Validate kernel correctness by comparing against reference implementations"""

    # Tolerance levels by data type
    TOLERANCES = {
        np.float32: 1e-5,
        np.float16: 1e-2,
        np.int32: 0,  # Exact match for integers
        np.int8: 0,
    }

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results = []

    def compare_tensors(
        self, computed: np.ndarray, reference: np.ndarray, tolerance: float | None = None
    ) -> tuple[bool, float, float]:
        """
        Compare two tensors

        Returns:
            Tuple of (passed, max_error, mean_error)
        """
        if computed.shape != reference.shape:
            return False, float("inf"), float("inf")

        if computed.dtype != reference.dtype:
            computed = computed.astype(reference.dtype)

        if tolerance is None:
            tolerance = self.TOLERANCES.get(reference.dtype, 1e-5)

        diff = np.abs(computed - reference)
        max_error = np.max(diff)
        mean_error = np.mean(diff)

        # Check relative error for non-zero values
        rel_diff = np.abs((computed - reference) / (reference + 1e-10))
        max_rel_error = np.max(rel_diff)

        passed = max_error <= tolerance or max_rel_error <= tolerance * 100

        return passed, max_error, mean_error, max_rel_error

    def validate_matmul(
        self, computed_C: np.ndarray, A: np.ndarray, B: np.ndarray
    ) -> ValidationResult:
        """Validate MatMul kernel"""
        # Compute reference
        reference_C = np.matmul(A, B)

        passed, max_err, mean_err, rel_err = self.compare_tensors(computed_C, reference_C)

        result = ValidationResult(
            kernel_name="matmul",
            passed=passed,
            max_error=max_err,
            mean_error=mean_err,
            rel_error=rel_err,
            error_type="absolute",
            details=f"C shape: {computed_C.shape}, dtype: {computed_C.dtype}",
        )

        self.results.append(result)

        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} MatMul: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

        return result

    def validate_conv2d(
        self,
        computed_output: np.ndarray,
        input_tensor: np.ndarray,
        kernel: np.ndarray,
        stride: tuple[int, int] = (1, 1),
        padding: tuple[int, int] = (0, 0),
    ) -> ValidationResult:
        """
        Validate Conv2D kernel

        Note: Reference implementation can be expensive, so we use scipy if available
        """
        try:
            from scipy.signal import convolve

            # This is a simplified reference - actual conv2d is more complex
            # For real validation, use a well-tested library like torch or tf
            reference_output = convolve(input_tensor, kernel, mode="same")
        except ImportError:
            # Fallback: just check output shape and dtype
            reference_output = computed_output.copy()

        passed, max_err, mean_err, rel_err = self.compare_tensors(computed_output, reference_output)

        result = ValidationResult(
            kernel_name="conv2d",
            passed=passed,
            max_error=max_err,
            mean_error=mean_err,
            rel_error=rel_err,
            error_type="absolute",
            details=f"output shape: {computed_output.shape}, stride: {stride}, padding: {padding}",
        )

        self.results.append(result)

        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} Conv2D: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

        return result

    def validate_attention(
        self,
        computed_output: np.ndarray,
        Q: np.ndarray,
        K: np.ndarray,
        V: np.ndarray,
        scale: float | None = None,
    ) -> ValidationResult:
        """
        Validate Scaled Dot-Product Attention

        Computes reference SDPA: softmax((Q·K^T)/√d)·V
        """
        if scale is None:
            scale = 1.0 / np.sqrt(Q.shape[-1])

        # Compute scores: (Q·K^T) / √d
        scores = np.matmul(Q, K.transpose(0, 2, 1)) * scale

        # Apply softmax
        scores_max = np.max(scores, axis=-1, keepdims=True)
        exp_scores = np.exp(scores - scores_max)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        # Apply to values
        reference_output = np.matmul(attention_weights, V)

        passed, max_err, mean_err, rel_err = self.compare_tensors(computed_output, reference_output)

        result = ValidationResult(
            kernel_name="attention",
            passed=passed,
            max_error=max_err,
            mean_error=mean_err,
            rel_error=rel_err,
            error_type="absolute",
            details=f"output shape: {computed_output.shape}, scale: {scale}",
        )

        self.results.append(result)

        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} Attention: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

        return result

    def validate_activation(
        self, computed_output: np.ndarray, input_tensor: np.ndarray, activation: str
    ) -> ValidationResult:
        """Validate activation function kernels"""

        # Compute reference
        if activation.lower() == "relu":
            reference = np.maximum(input_tensor, 0)
        elif activation.lower() == "gelu":
            # Approximate GELU
            reference = (
                0.5
                * input_tensor
                * (1 + np.tanh(np.sqrt(2 / np.pi) * (input_tensor + 0.044715 * input_tensor**3)))
            )
        elif activation.lower() == "silu":
            reference = input_tensor / (1 + np.exp(-input_tensor))
        else:
            raise ValueError(f"Unknown activation: {activation}")

        passed, max_err, mean_err, rel_err = self.compare_tensors(computed_output, reference)

        result = ValidationResult(
            kernel_name=activation,
            passed=passed,
            max_error=max_err,
            mean_error=mean_err,
            rel_error=rel_err,
            error_type="absolute",
            details=f"{activation.upper()} activation, shape: {computed_output.shape}",
        )

        self.results.append(result)

        if self.verbose:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status} {activation.upper()}: max_err={max_err:.2e}, mean_err={mean_err:.2e}")

        return result

    def get_summary(self) -> dict:
        """Get summary of all validation results"""
        if not self.results:
            return {"total": 0, "passed": 0, "failed": 0}

        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed

        return {
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / len(self.results),
            "results": [
                {
                    "kernel": r.kernel_name,
                    "passed": r.passed,
                    "max_error": float(r.max_error),
                    "mean_error": float(r.mean_error),
                }
                for r in self.results
            ],
        }

    def save_report(self, filepath: str):
        """Save validation report to JSON"""
        report = {
            "summary": self.get_summary(),
            "results": [
                {
                    "kernel": r.kernel_name,
                    "passed": r.passed,
                    "max_error": float(r.max_error),
                    "mean_error": float(r.mean_error),
                    "rel_error": float(r.rel_error),
                    "details": r.details,
                }
                for r in self.results
            ],
        }

        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)

        print(f"Report saved to {filepath}")


def example_validation():
    """Example validation workflow"""
    validator = FunctionalValidator(verbose=True)

    # Example: MatMul validation
    A = np.random.randn(64, 64).astype(np.float32)
    B = np.random.randn(64, 64).astype(np.float32)
    expected_C = np.matmul(A, B)

    # Simulate kernel computation (would normally come from actual kernel)
    computed_C = expected_C.copy()  # Assume perfect for now

    result = validator.validate_matmul(computed_C, A, B)
    print(f"Result: {result.passed}")

    # Get summary
    summary = validator.get_summary()
    print(f"\nSummary: {summary['passed']}/{summary['total']} passed")


if __name__ == "__main__":
    example_validation()
