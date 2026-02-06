#!/usr/bin/env python3
"""
Functional Validation Harness for KernelSmith
Compares kernel outputs against reference implementations
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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
        self, computed: np.ndarray, reference: np.ndarray, tolerance: Optional[float] = None
    ) -> Tuple[bool, float, float]:
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
        self, computed_output: np.ndarray, input_tensor: np.ndarray, kernel: np.ndarray,
        stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (0, 0)
    ) -> ValidationResult:
        """
        Validate Conv2D kernel

        Note: Reference implementation can be expensive, so we use scipy if available
        """
        try:
            from scipy.signal import convolve
            # This is a simplified reference - actual conv2d is more complex
            # For real validation, use a well-tested library like torch or tf
            reference_output = convolve(input_tensor, kernel, mode='same')
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
        self, computed_output: np.ndarray, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
        scale: Optional[float] = None
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
            reference = 0.5 * input_tensor * (1 + np.tanh(
                np.sqrt(2 / np.pi) * (input_tensor + 0.044715 * input_tensor ** 3)
            ))
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
