import json
from pathlib import Path

import numpy as np
import pytest

from tests.functional_validator import compare_arrays
from tests.golden.generate import generate_bundle, generate_case_arrays
from tests.golden.schema import (
    CaseDescriptor,
    GoldenSchemaError,
    load_case_descriptor,
    validate_manifest,
)
from tests.test_data_generator import TestDataGenerator

CASES_DIR = Path(__file__).parent / "golden" / "cases"


def test_committed_descriptors_validate() -> None:
    relu = load_case_descriptor(CASES_DIR / "relu_f32_smoke.json")
    matmul = load_case_descriptor(CASES_DIR / "matmul_f32_smoke.json")

    assert relu.name == "relu_f32_smoke"
    assert matmul.inputs[0].shape == (4, 5)
    assert matmul.outputs[0].shape == (4, 3)


def test_descriptor_validation_rejects_bad_matmul_shape() -> None:
    with (CASES_DIR / "matmul_f32_smoke.json").open() as handle:
        descriptor = json.load(handle)
    descriptor["outputs"][0]["shape"] = [4, 4]

    with pytest.raises(GoldenSchemaError, match="matmul output shape"):
        CaseDescriptor.from_json(descriptor)


def test_generator_is_deterministic_for_f32_relu_and_matmul() -> None:
    generator = TestDataGenerator()
    relu_input_a, relu_output_a = generator.generate_relu_f32_golden(seed=7)
    relu_input_b, relu_output_b = generator.generate_relu_f32_golden(seed=7)
    lhs_a, rhs_a, matmul_output_a = generator.generate_matmul_f32_golden(seed=11)
    lhs_b, rhs_b, matmul_output_b = generator.generate_matmul_f32_golden(seed=11)

    np.testing.assert_array_equal(relu_input_a, relu_input_b)
    np.testing.assert_array_equal(relu_output_a, relu_output_b)
    np.testing.assert_array_equal(lhs_a, lhs_b)
    np.testing.assert_array_equal(rhs_a, rhs_b)
    np.testing.assert_array_equal(matmul_output_a, matmul_output_b)


def test_bundle_manifest_records_and_validates_hashes(tmp_path: Path) -> None:
    descriptor = load_case_descriptor(CASES_DIR / "relu_f32_smoke.json")
    manifest_path = generate_bundle(
        descriptor, tmp_path, descriptor_path=CASES_DIR / "relu_f32_smoke.json"
    )

    manifest = validate_manifest(manifest_path)
    assert manifest["case"] == "relu_f32_smoke"
    assert {tensor["role"] for tensor in manifest["tensors"]} == {"input", "expected_output"}

    generated = generate_case_arrays(descriptor)
    np.testing.assert_array_equal(
        generated["expected_Y"],
        np.load(manifest_path.parent / "expected_Y.npy"),
    )


def test_manifest_hash_validation_reports_corruption(tmp_path: Path) -> None:
    descriptor = load_case_descriptor(CASES_DIR / "matmul_f32_smoke.json")
    manifest_path = generate_bundle(descriptor, tmp_path)
    (manifest_path.parent / "expected_C.npy").write_bytes(b"corrupted")

    with pytest.raises(GoldenSchemaError, match="hash mismatch"):
        validate_manifest(manifest_path)


def test_comparator_exact_and_diagnostics() -> None:
    actual = np.array([1, 2, 4], dtype=np.int32)
    expected = np.array([1, 2, 3], dtype=np.int32)

    result = compare_arrays(actual, expected, {"mode": "exact"})

    assert not result.passed
    assert result.mismatch_count == 1
    assert result.first_mismatches == [{"index": [2], "actual": 4, "expected": 3}]
    assert "first mismatch" in result.details


def test_comparator_allclose_pass_and_fail() -> None:
    expected = np.array([1.0, 2.0], dtype=np.float32)
    close = np.array([1.0, 2.000001], dtype=np.float32)
    far = np.array([1.0, 2.1], dtype=np.float32)

    assert compare_arrays(close, expected, {"mode": "allclose", "rtol": 1e-5, "atol": 1e-5}).passed
    assert not compare_arrays(
        far, expected, {"mode": "allclose", "rtol": 1e-5, "atol": 1e-5}
    ).passed


def test_comparator_quantized_modes() -> None:
    expected = np.array([0, 2, 4], dtype=np.int8)
    exact = np.array([0, 2, 4], dtype=np.int8)
    off_by_one = np.array([0, 3, 4], dtype=np.int8)

    assert compare_arrays(exact, expected, {"mode": "quantized_exact"}).passed
    assert not compare_arrays(off_by_one, expected, {"mode": "quantized_exact"}).passed

    dequantized = compare_arrays(
        off_by_one,
        expected,
        {
            "mode": "dequantized_allclose",
            "scale": 0.5,
            "zero_point": 0,
            "rtol": 0.0,
            "atol": 0.5,
        },
    )
    assert dequantized.passed
