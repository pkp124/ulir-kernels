import json
import stat
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
from tests.verify import VerifyError, _json_safe, benchmark_case, verify_case

CASES_DIR = Path(__file__).parent / "golden" / "cases"
TRANSFORMER_HELPER_CASE_NAMES = [
    "add_f32_smoke",
    "mul_f32_smoke",
    "rms_norm_f32_smoke",
    "softmax_f32_smoke",
]
QUANTIZED_CASE_NAMES = [
    "dot_i8_smoke",
    "matvec_i8_smoke",
    "matmul_i8_smoke",
    "dot_w4a8_smoke",
    "matvec_w4a8_smoke",
]


def test_committed_descriptors_validate() -> None:
    relu = load_case_descriptor(CASES_DIR / "relu_f32_smoke.json")
    matmul = load_case_descriptor(CASES_DIR / "matmul_f32_smoke.json")

    assert relu.name == "relu_f32_smoke"
    assert matmul.inputs[0].shape == (4, 5)
    assert matmul.outputs[0].shape == (4, 3)


def test_committed_quantized_descriptors_validate() -> None:
    descriptors = [
        load_case_descriptor(CASES_DIR / f"{case_name}.json") for case_name in QUANTIZED_CASE_NAMES
    ]

    assert [descriptor.kernel for descriptor in descriptors] == [
        "dot_i8",
        "matvec_i8",
        "matmul_i8",
        "dot_w4a8",
        "matvec_w4a8",
    ]
    assert descriptors[0].compare.quantization["rounding"] == "none"
    assert descriptors[3].inputs[1].layout["nibble_order"] == "low_even_high_odd"


def test_committed_transformer_helper_descriptors_validate() -> None:
    descriptors = [
        load_case_descriptor(CASES_DIR / f"{case_name}.json")
        for case_name in TRANSFORMER_HELPER_CASE_NAMES
    ]

    assert [descriptor.kernel for descriptor in descriptors] == [
        "add",
        "mul",
        "rms_norm",
        "softmax",
    ]
    assert descriptors[2].compare.quantization["epsilon"] == 1e-5
    assert all("riscv_rvv_256" in descriptor.targets for descriptor in descriptors)


def test_descriptor_validation_rejects_bad_matmul_shape() -> None:
    with (CASES_DIR / "matmul_f32_smoke.json").open() as handle:
        descriptor = json.load(handle)
    descriptor["outputs"][0]["shape"] = [4, 4]

    with pytest.raises(GoldenSchemaError, match="matmul output shape"):
        CaseDescriptor.from_json(descriptor)


def test_quantized_descriptor_requires_explicit_policy() -> None:
    with (CASES_DIR / "dot_i8_smoke.json").open() as handle:
        descriptor = json.load(handle)
    del descriptor["compare"]["rounding"]

    with pytest.raises(GoldenSchemaError, match="rounding"):
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


def test_generator_builds_i8_quantized_references() -> None:
    dot = load_case_descriptor(CASES_DIR / "dot_i8_smoke.json")
    matvec = load_case_descriptor(CASES_DIR / "matvec_i8_smoke.json")
    matmul = load_case_descriptor(CASES_DIR / "matmul_i8_smoke.json")

    dot_arrays = generate_case_arrays(dot)
    x = dot_arrays["input_X"].astype(np.int32) - dot.compare.quantization["input_zero_point"]
    w = dot_arrays["input_W"].astype(np.int32) - dot.compare.quantization["weight_zero_point"]
    np.testing.assert_array_equal(dot_arrays["expected_Y"], np.array([np.sum(x * w)], np.int32))

    matvec_arrays = generate_case_arrays(matvec)
    x = matvec_arrays["input_X"].astype(np.int32) - matvec.compare.quantization["input_zero_point"]
    w = matvec_arrays["input_W"].astype(np.int32) - matvec.compare.quantization["weight_zero_point"]
    np.testing.assert_array_equal(matvec_arrays["expected_Y"], np.sum(w * x, axis=1))

    matmul_arrays = generate_case_arrays(matmul)
    a = matmul_arrays["input_A"].astype(np.int32) - matmul.compare.quantization["input_zero_point"]
    b = matmul_arrays["input_B"].astype(np.int32) - matmul.compare.quantization["weight_zero_point"]
    np.testing.assert_array_equal(matmul_arrays["expected_C"], np.matmul(a, b))


def test_generator_builds_transformer_helper_references() -> None:
    add = load_case_descriptor(CASES_DIR / "add_f32_smoke.json")
    mul = load_case_descriptor(CASES_DIR / "mul_f32_smoke.json")
    rms_norm = load_case_descriptor(CASES_DIR / "rms_norm_f32_smoke.json")
    softmax = load_case_descriptor(CASES_DIR / "softmax_f32_smoke.json")

    add_arrays = generate_case_arrays(add)
    np.testing.assert_allclose(
        add_arrays["expected_Y"],
        add_arrays["input_X"] + add_arrays["input_Z"],
        rtol=1e-6,
        atol=1e-6,
    )

    mul_arrays = generate_case_arrays(mul)
    np.testing.assert_allclose(
        mul_arrays["expected_Y"],
        mul_arrays["input_X"] * mul_arrays["input_Z"],
        rtol=1e-6,
        atol=1e-6,
    )

    rms_arrays = generate_case_arrays(rms_norm)
    squared = rms_arrays["input_X"] * rms_arrays["input_X"]
    scale = 1.0 / np.sqrt(np.mean(squared, axis=-1, keepdims=True) + 1e-5)
    expected_rms = rms_arrays["input_X"] * scale * rms_arrays["input_W"].reshape((1, -1))
    np.testing.assert_allclose(rms_arrays["expected_Y"], expected_rms, rtol=1e-6, atol=1e-6)

    softmax_arrays = generate_case_arrays(softmax)
    assert np.all(softmax_arrays["expected_Y"] >= 0.0)
    np.testing.assert_allclose(
        np.sum(softmax_arrays["expected_Y"], axis=-1),
        np.ones((2,), dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def _unpack_i4(packed: np.ndarray, logical_shape: tuple[int, ...]) -> np.ndarray:
    rows = 1 if len(logical_shape) == 1 else logical_shape[0]
    cols = logical_shape[0] if len(logical_shape) == 1 else logical_shape[1]
    packed_rows = packed.reshape(rows, -1)
    unpacked = np.zeros((rows, cols), dtype=np.int8)
    for row in range(rows):
        for col in range(cols):
            byte = int(packed_rows[row, col // 2])
            nibble = (byte >> 4) if col & 1 else byte & 0x0F
            unpacked[row, col] = nibble - 0x10 if nibble & 0x08 else nibble
    if len(logical_shape) == 1:
        return unpacked.reshape(logical_shape)
    return unpacked


def test_generator_builds_w4a8_quantized_references() -> None:
    descriptor = load_case_descriptor(CASES_DIR / "matvec_w4a8_smoke.json")
    arrays = generate_case_arrays(descriptor)

    x = arrays["input_X"].astype(np.int32) - descriptor.compare.quantization["input_zero_point"]
    x_dequant = x.astype(np.float32) * descriptor.compare.quantization["input_scale"]
    weights = _unpack_i4(arrays["input_W_packed"], (3, 5)).astype(np.int32)
    scales = arrays["input_W_scales"]
    group_indices = np.arange(5) // descriptor.inputs[1].layout["group_size"]
    expected = np.sum(weights.astype(np.float32) * scales[:, group_indices] * x_dequant, axis=1)

    np.testing.assert_allclose(arrays["expected_Y"], expected.astype(np.float32), rtol=1e-6)


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


def test_quantized_manifest_records_policy_and_packed_layout(tmp_path: Path) -> None:
    descriptor = load_case_descriptor(CASES_DIR / "matvec_w4a8_smoke.json")
    manifest_path = generate_bundle(
        descriptor,
        tmp_path,
        descriptor_path=CASES_DIR / "matvec_w4a8_smoke.json",
    )

    manifest = validate_manifest(manifest_path)
    assert manifest["quantization_policy"]["accumulator_dtype"] == "float32"
    assert manifest["quantization_policy"]["input_scale"] == 0.125
    assert manifest["packed_layouts"] == [
        {
            "group_size": 2,
            "kind": "w4a8_packed",
            "logical_shape": [3, 5],
            "nibble_order": "low_even_high_odd",
            "packed_stride_bytes": 3,
            "tensor": "W_packed",
        }
    ]
    packed_entry = next(tensor for tensor in manifest["tensors"] if tensor["name"] == "W_packed")
    assert packed_entry["layout"]["logical_shape"] == [3, 5]


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

    exact_policy = {
        "accumulator_dtype": "int32",
        "mode": "quantized_exact",
        "rounding": "none",
        "saturation": "int32",
    }
    assert compare_arrays(exact, expected, exact_policy).passed
    assert not compare_arrays(off_by_one, expected, exact_policy).passed

    dequantized = compare_arrays(
        off_by_one,
        expected,
        {
            "accumulator_dtype": "int32",
            "mode": "dequantized_allclose",
            "rounding": "nearest_even",
            "saturation": "int8",
            "scale": 0.5,
            "zero_point": 0,
            "rtol": 0.0,
            "atol": 0.5,
        },
    )
    assert dequantized.passed


def test_verify_json_sanitizes_non_finite_metrics() -> None:
    assert _json_safe({"max_error": float("inf"), "nested": [float("nan")]}) == {
        "max_error": None,
        "nested": [None],
    }


def test_verify_rejects_undeclared_target(tmp_path: Path) -> None:
    with pytest.raises(VerifyError, match="target missing_target is not declared"):
        verify_case(
            case_path=CASES_DIR / "relu_f32_smoke.json",
            target="missing_target",
            manifest_dir=Path("tests/golden/generated"),
            output_dir=tmp_path,
            host_runner=None,
        )


def _write_fake_qemu(tmp_path: Path) -> Path:
    qemu = tmp_path / "fake-qemu"
    qemu.write_text(
        "#!/usr/bin/env python3\n"
        "import subprocess\n"
        "import sys\n"
        "args = sys.argv[1:]\n"
        "if args[:1] == ['--version']:\n"
        "    print('fake qemu')\n"
        "    raise SystemExit(0)\n"
        "if args[:1] == ['-cpu']:\n"
        "    args = args[2:]\n"
        "raise SystemExit(subprocess.run(args).returncode)\n"
    )
    qemu.chmod(0o755)
    return qemu


def _write_fake_descriptor_runner(tmp_path: Path, name: str, *, executable: bool) -> Path:
    runner = tmp_path / name
    runner.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        "import numpy as np\n"
        "kernel = sys.argv[1]\n"
        "if kernel in {'add', 'mul'}:\n"
        "    lhs_path, rhs_path, output_path, n = sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])\n"
        "    lhs = np.fromfile(lhs_path, dtype=np.float32, count=n)\n"
        "    rhs = np.fromfile(rhs_path, dtype=np.float32, count=n)\n"
        "    out = lhs + rhs if kernel == 'add' else lhs * rhs\n"
        "    out.astype(np.float32).tofile(output_path)\n"
        "elif kernel == 'relu':\n"
        "    input_path, output_path, n = sys.argv[2], sys.argv[3], int(sys.argv[4])\n"
        "    data = np.fromfile(input_path, dtype=np.float32, count=n)\n"
        "    np.maximum(data, 0).astype(np.float32).tofile(output_path)\n"
        "elif kernel == 'matmul':\n"
        "    lhs_path, rhs_path, output_path = sys.argv[2], sys.argv[3], sys.argv[4]\n"
        "    m, n, k = int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7])\n"
        "    lhs = np.fromfile(lhs_path, dtype=np.float32, count=m * k).reshape(m, k)\n"
        "    rhs = np.fromfile(rhs_path, dtype=np.float32, count=k * n).reshape(k, n)\n"
        "    np.matmul(lhs, rhs).astype(np.float32).tofile(output_path)\n"
        "elif kernel == 'rms_norm':\n"
        "    input_path, weight_path, output_path = sys.argv[2], sys.argv[3], sys.argv[4]\n"
        "    outer, inner, eps = int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])\n"
        "    data = np.fromfile(input_path, dtype=np.float32, count=outer * inner).reshape(outer, inner)\n"
        "    weight = np.fromfile(weight_path, dtype=np.float32, count=inner)\n"
        "    scale = 1.0 / np.sqrt(np.mean(data * data, axis=-1, keepdims=True) + eps)\n"
        "    (data * scale * weight.reshape(1, -1)).astype(np.float32).tofile(output_path)\n"
        "elif kernel == 'softmax':\n"
        "    input_path, output_path = sys.argv[2], sys.argv[3]\n"
        "    outer, inner = int(sys.argv[4]), int(sys.argv[5])\n"
        "    data = np.fromfile(input_path, dtype=np.float32, count=outer * inner).reshape(outer, inner)\n"
        "    shifted = data - np.max(data, axis=-1, keepdims=True)\n"
        "    exp = np.exp(shifted)\n"
        "    (exp / np.sum(exp, axis=-1, keepdims=True)).astype(np.float32).tofile(output_path)\n"
        "else:\n"
        "    print(f'unsupported kernel: {kernel}', file=sys.stderr)\n"
        "    raise SystemExit(2)\n"
    )
    if executable:
        runner.chmod(0o755)
    return runner


def test_verify_riscv_runs_qemu_vlens_and_compares_host_output(tmp_path: Path) -> None:
    descriptor_path = CASES_DIR / "relu_f32_smoke.json"
    descriptor = load_case_descriptor(descriptor_path)
    manifest_path = generate_bundle(
        descriptor,
        tmp_path / "generated",
        descriptor_path=descriptor_path,
    )
    host_runner = _write_fake_descriptor_runner(tmp_path, "fake-host-runner", executable=True)
    riscv_runner = _write_fake_descriptor_runner(tmp_path, "fake-riscv-runner", executable=False)

    results = verify_case(
        case_path=descriptor_path,
        target="riscv_rvv_256",
        manifest_dir=manifest_path.parent.parent,
        output_dir=tmp_path / "actual",
        host_runner=host_runner,
        riscv_runner=riscv_runner,
        qemu_binary=str(_write_fake_qemu(tmp_path)),
        vlens=(256, 512),
        timeout=5,
    )

    assert {result["vlen"] for result in results} == {256, 512}
    assert all(result["status"] == "PASS" for result in results)
    assert all(result["compare"]["mismatch_count"] == 0 for result in results)
    assert all(result["host_compare"]["mismatch_count"] == 0 for result in results)
    assert riscv_runner.stat().st_mode & stat.S_IXUSR


def test_benchmark_case_reports_host_and_riscv_samples(tmp_path: Path) -> None:
    descriptor_path = CASES_DIR / "matmul_f32_smoke.json"
    descriptor = load_case_descriptor(descriptor_path)
    manifest_path = generate_bundle(
        descriptor,
        tmp_path / "generated",
        descriptor_path=descriptor_path,
    )
    host_runner = _write_fake_descriptor_runner(tmp_path, "fake-host-runner", executable=True)
    riscv_runner = _write_fake_descriptor_runner(tmp_path, "fake-riscv-runner", executable=False)

    benchmarks = benchmark_case(
        case_path=descriptor_path,
        target="riscv_rvv_256",
        manifest_dir=manifest_path.parent.parent,
        output_dir=tmp_path / "actual",
        host_runner=host_runner,
        riscv_runner=riscv_runner,
        qemu_binary=str(_write_fake_qemu(tmp_path)),
        vlens=(256, 512),
        timeout=5,
        runs=2,
        warmup=1,
    )

    assert [benchmark["target"] for benchmark in benchmarks] == [
        "host_reference",
        "riscv_rvv_256",
        "riscv_rvv_256",
    ]
    assert {benchmark.get("vlen") for benchmark in benchmarks[1:]} == {256, 512}
    assert all(benchmark["kind"] == "kernelsmith_benchmark" for benchmark in benchmarks)
    assert all(benchmark["timed_runs"] == 2 for benchmark in benchmarks)
    assert all(len(benchmark["samples_ns"]) == 2 for benchmark in benchmarks)
    assert all(benchmark["min_ns"] <= benchmark["median_ns"] for benchmark in benchmarks)
    assert all(benchmark["median_ns"] <= benchmark["max_ns"] for benchmark in benchmarks)
    assert all(benchmark["command"] for benchmark in benchmarks)


def test_verify_riscv_rejects_vlen_below_profile(tmp_path: Path) -> None:
    descriptor_path = CASES_DIR / "matmul_f32_smoke.json"
    descriptor = load_case_descriptor(descriptor_path)
    manifest_path = generate_bundle(descriptor, tmp_path / "generated")

    with pytest.raises(VerifyError, match="VLEN 128 is below riscv_rvv_256 minimum VLEN 256"):
        verify_case(
            case_path=descriptor_path,
            target="riscv_rvv_256",
            manifest_dir=manifest_path.parent.parent,
            output_dir=tmp_path / "actual",
            host_runner=None,
            riscv_runner=_write_fake_descriptor_runner(
                tmp_path, "fake-riscv-runner", executable=True
            ),
            qemu_binary=str(_write_fake_qemu(tmp_path)),
            vlens=(128,),
        )
