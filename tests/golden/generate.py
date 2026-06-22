#!/usr/bin/env python3
"""Generate deterministic golden-reference bundles from case descriptors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .schema import (
        SCHEMA_VERSION,
        CaseDescriptor,
        TensorSpec,
        canonical_json_sha256,
        load_case_descriptor,
        sha256_file,
        validate_manifest,
        write_json,
    )
except ImportError:  # pragma: no cover - supports direct script execution.
    from schema import (  # type: ignore[no-redef]
        SCHEMA_VERSION,
        CaseDescriptor,
        TensorSpec,
        canonical_json_sha256,
        load_case_descriptor,
        sha256_file,
        validate_manifest,
        write_json,
    )


def _distribution_bounds(distribution: dict[str, Any]) -> tuple[float, float]:
    if distribution.get("kind", "uniform") != "uniform":
        raise ValueError("only uniform input distributions are supported")
    return float(distribution.get("low", -1.0)), float(distribution.get("high", 1.0))


def generate_f32_tensor(
    spec: TensorSpec,
    *,
    seed: int,
    distribution: dict[str, Any],
) -> np.ndarray:
    """Generate one deterministic float32 tensor."""
    if spec.dtype != "float32":
        raise ValueError(f"generate_f32_tensor requires float32, got {spec.dtype}")
    low, high = _distribution_bounds(distribution)
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=spec.shape).astype(np.float32)


def generate_i8_tensor(spec: TensorSpec, *, seed: int, distribution: dict[str, Any]) -> np.ndarray:
    """Generate one deterministic int8 tensor."""
    if spec.dtype != "int8":
        raise ValueError(f"generate_i8_tensor requires int8, got {spec.dtype}")
    bounds = distribution.get("int8", distribution)
    if bounds.get("kind", "integers") != "integers":
        raise ValueError("int8 tensors require an integers distribution")
    low = int(bounds.get("low", -8))
    high = int(bounds.get("high", 8))
    rng = np.random.default_rng(seed)
    return rng.integers(low, high, size=spec.shape, dtype=np.int8)


def generate_scale_tensor(
    spec: TensorSpec, *, seed: int, distribution: dict[str, Any]
) -> np.ndarray:
    """Generate one deterministic positive float32 scale tensor."""
    if spec.dtype != "float32":
        raise ValueError(f"generate_scale_tensor requires float32, got {spec.dtype}")
    bounds = distribution.get("scale", {"kind": "uniform", "low": 0.125, "high": 0.5})
    low, high = _distribution_bounds(bounds)
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, size=spec.shape).astype(np.float32)


def _generate_i4_values(shape: tuple[int, ...], *, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(-8, 8, size=shape, dtype=np.int8)


def _pack_i4_values(values: np.ndarray) -> np.ndarray:
    flat = values.astype(np.int8, copy=False).reshape(-1)
    packed = np.zeros((flat.size + 1) // 2, dtype=np.uint8)
    for index, value in enumerate(flat):
        nibble = np.uint8(int(value) & 0x0F)
        if index & 1:
            packed[index // 2] |= np.uint8(nibble << 4)
        else:
            packed[index // 2] |= nibble
    return packed


def _generate_w4_input(
    spec: TensorSpec,
    *,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    logical_shape = tuple(spec.layout["logical_shape"])
    values = _generate_i4_values(logical_shape, seed=seed)
    if len(logical_shape) == 1:
        packed = _pack_i4_values(values).reshape(spec.shape)
    elif len(logical_shape) == 2:
        packed = np.stack([_pack_i4_values(row) for row in values], axis=0).reshape(spec.shape)
    else:
        raise ValueError("W4A8 packed tensors support rank-1 or rank-2 logical shapes")
    return packed, values


def _quantization_value(descriptor: CaseDescriptor, key: str) -> Any:
    return descriptor.compare.quantization[key]


def _rms_norm_reference(input_tensor: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    squared = (input_tensor * input_tensor).astype(np.float32)
    sum_squares = np.sum(squared, axis=-1, keepdims=True, dtype=np.float32)
    mean_squares = sum_squares / np.float32(input_tensor.shape[-1])
    scale = (np.float32(1.0) / np.sqrt(mean_squares + np.float32(eps))).astype(np.float32)
    return (input_tensor * scale * weight.reshape((1, -1))).astype(np.float32)


def _softmax_reference(input_tensor: np.ndarray) -> np.ndarray:
    max_values = np.max(input_tensor, axis=-1, keepdims=True)
    shifted = input_tensor - max_values
    exp_values = np.exp(shifted, dtype=np.float32)
    sums = np.sum(exp_values, axis=-1, keepdims=True, dtype=np.float32)
    return (exp_values / sums).astype(np.float32)


def _i8_accumulate(lhs: np.ndarray, rhs: np.ndarray, *, lhs_zp: int, rhs_zp: int) -> np.ndarray:
    lhs_i32 = lhs.astype(np.int32) - lhs_zp
    rhs_i32 = rhs.astype(np.int32) - rhs_zp
    return np.sum(lhs_i32 * rhs_i32, axis=-1, dtype=np.int64).astype(np.int32)


def generate_case_arrays(descriptor: CaseDescriptor) -> dict[str, np.ndarray]:
    """Generate descriptor inputs and expected outputs."""
    arrays: dict[str, np.ndarray] = {}
    distribution = descriptor.generator.distribution

    if descriptor.kernel in {"add", "relu", "mul", "matmul", "rms_norm", "softmax"}:
        for index, tensor in enumerate(descriptor.inputs):
            arrays[f"input_{tensor.name}"] = generate_f32_tensor(
                tensor,
                seed=descriptor.generator.seed + index * 1009,
                distribution=distribution,
            )

        if descriptor.kernel == "relu":
            input_tensor = arrays[f"input_{descriptor.inputs[0].name}"]
            arrays[f"expected_{descriptor.outputs[0].name}"] = np.maximum(input_tensor, 0).astype(
                np.float32
            )
        elif descriptor.kernel == "add":
            lhs = arrays[f"input_{descriptor.inputs[0].name}"]
            rhs = arrays[f"input_{descriptor.inputs[1].name}"]
            arrays[f"expected_{descriptor.outputs[0].name}"] = (lhs + rhs).astype(np.float32)
        elif descriptor.kernel == "mul":
            lhs = arrays[f"input_{descriptor.inputs[0].name}"]
            rhs = arrays[f"input_{descriptor.inputs[1].name}"]
            arrays[f"expected_{descriptor.outputs[0].name}"] = (lhs * rhs).astype(np.float32)
        elif descriptor.kernel == "matmul":
            lhs = arrays[f"input_{descriptor.inputs[0].name}"]
            rhs = arrays[f"input_{descriptor.inputs[1].name}"]
            arrays[f"expected_{descriptor.outputs[0].name}"] = np.matmul(lhs, rhs).astype(
                np.float32
            )
        elif descriptor.kernel == "rms_norm":
            input_tensor = arrays[f"input_{descriptor.inputs[0].name}"]
            weight = arrays[f"input_{descriptor.inputs[1].name}"]
            eps = float(descriptor.compare.quantization["epsilon"])
            arrays[f"expected_{descriptor.outputs[0].name}"] = _rms_norm_reference(
                input_tensor, weight, eps
            )
        else:
            input_tensor = arrays[f"input_{descriptor.inputs[0].name}"]
            arrays[f"expected_{descriptor.outputs[0].name}"] = _softmax_reference(input_tensor)
        return arrays

    if descriptor.kernel in {"dot_i8", "matvec_i8", "matmul_i8"}:
        for index, tensor in enumerate(descriptor.inputs):
            arrays[f"input_{tensor.name}"] = generate_i8_tensor(
                tensor,
                seed=descriptor.generator.seed + index * 1009,
                distribution=distribution,
            )
        input_zp = int(_quantization_value(descriptor, "input_zero_point"))
        weight_zp = int(_quantization_value(descriptor, "weight_zero_point"))
        lhs = arrays[f"input_{descriptor.inputs[0].name}"]
        rhs = arrays[f"input_{descriptor.inputs[1].name}"]
        if descriptor.kernel == "dot_i8":
            expected = np.array([_i8_accumulate(lhs, rhs, lhs_zp=input_zp, rhs_zp=weight_zp)])
        elif descriptor.kernel == "matvec_i8":
            expected = _i8_accumulate(rhs, lhs, lhs_zp=weight_zp, rhs_zp=input_zp)
        else:
            lhs_i32 = lhs.astype(np.int32) - input_zp
            rhs_i32 = rhs.astype(np.int32) - weight_zp
            expected = np.matmul(lhs_i32, rhs_i32).astype(np.int32)
        arrays[f"expected_{descriptor.outputs[0].name}"] = expected.astype(np.int32)
        return arrays

    if descriptor.kernel in {"dot_w4a8", "matvec_w4a8"}:
        vector_spec, weight_spec, scale_spec = descriptor.inputs
        arrays[f"input_{vector_spec.name}"] = generate_i8_tensor(
            vector_spec,
            seed=descriptor.generator.seed,
            distribution=distribution,
        )
        packed, logical_weight = _generate_w4_input(
            weight_spec,
            seed=descriptor.generator.seed + 1009,
        )
        arrays[f"input_{weight_spec.name}"] = packed
        arrays[f"input_{scale_spec.name}"] = generate_scale_tensor(
            scale_spec,
            seed=descriptor.generator.seed + 2018,
            distribution=distribution,
        )

        input_scale = float(_quantization_value(descriptor, "input_scale"))
        input_zp = int(_quantization_value(descriptor, "input_zero_point"))
        weight_zp = int(_quantization_value(descriptor, "weight_zero_point"))
        group_size = int(weight_spec.layout["group_size"])
        vector = arrays[f"input_{vector_spec.name}"].astype(np.int32) - input_zp
        dequantized_vector = vector.astype(np.float32) * input_scale
        scales = arrays[f"input_{scale_spec.name}"]
        weights = logical_weight.astype(np.int32) - weight_zp
        group_indices = np.arange(vector_spec.shape[0]) // group_size
        if descriptor.kernel == "dot_w4a8":
            dequantized_weight = weights.astype(np.float32) * scales[group_indices]
            expected = np.array([np.sum(dequantized_vector * dequantized_weight)], dtype=np.float32)
        else:
            dequantized_weight = weights.astype(np.float32) * scales[:, group_indices]
            expected = np.sum(dequantized_weight * dequantized_vector, axis=1).astype(np.float32)
        arrays[f"expected_{descriptor.outputs[0].name}"] = expected.astype(np.float32)
        return arrays

    raise ValueError(f"unsupported kernel: {descriptor.kernel}")


def _tensor_manifest_entry(
    *,
    name: str,
    role: str,
    spec: TensorSpec,
    filename: str,
    path: Path,
) -> dict[str, Any]:
    array = np.load(path)
    return {
        "name": name,
        "role": role,
        "filename": filename,
        "shape": list(spec.shape),
        "dtype": spec.dtype,
        "layout": spec.layout,
        "size_bytes": int(array.nbytes),
        "sha256": sha256_file(path),
    }


def generate_bundle(
    descriptor: CaseDescriptor,
    output_root: Path,
    *,
    descriptor_path: Path | None = None,
) -> Path:
    """Generate a golden bundle and return the manifest path."""
    case_dir = output_root / descriptor.name
    case_dir.mkdir(parents=True, exist_ok=True)

    arrays = generate_case_arrays(descriptor)
    for name, array in arrays.items():
        np.save(case_dir / f"{name}.npy", array)

    tensor_entries = []
    for input_spec in descriptor.inputs:
        filename = f"input_{input_spec.name}.npy"
        tensor_entries.append(
            _tensor_manifest_entry(
                name=input_spec.name,
                role="input",
                spec=input_spec,
                filename=filename,
                path=case_dir / filename,
            )
        )
    for output_spec in descriptor.outputs:
        filename = f"expected_{output_spec.name}.npy"
        tensor_entries.append(
            _tensor_manifest_entry(
                name=output_spec.name,
                role="expected_output",
                spec=output_spec,
                filename=filename,
                path=case_dir / filename,
            )
        )

    descriptor_json = descriptor.to_json()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case": descriptor.name,
        "kernel": descriptor.kernel,
        "descriptor": str(descriptor_path) if descriptor_path else None,
        "descriptor_sha256": canonical_json_sha256(descriptor_json),
        "generator": {
            "backend": descriptor.generator.kind,
            "backend_version": np.__version__,
            "function": descriptor.generator.function,
            "seed": descriptor.generator.seed,
            "distribution": descriptor.generator.distribution,
        },
        "compare": descriptor.compare.to_json(),
        "quantization_policy": descriptor.compare.quantization,
        "packed_layouts": [
            {"tensor": tensor.name, **tensor.layout}
            for tensor in descriptor.inputs
            if tensor.layout.get("kind") == "w4a8_packed"
        ],
        "targets": list(descriptor.targets),
        "vlens": list(descriptor.vlens),
        "tensors": tensor_entries,
    }
    write_json(case_dir / "manifest.json", manifest)
    validate_manifest(case_dir / "manifest.json")
    return case_dir / "manifest.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="Path to a case descriptor JSON")
    parser.add_argument(
        "--output-dir",
        default=Path("tests/golden/generated"),
        type=Path,
        help="Directory where generated bundles are written",
    )
    parser.add_argument(
        "--print-manifest",
        action="store_true",
        help="Print the generated manifest JSON to stdout",
    )
    args = parser.parse_args(argv)

    descriptor = load_case_descriptor(args.case)
    manifest_path = generate_bundle(descriptor, args.output_dir, descriptor_path=args.case)
    if args.print_manifest:
        print(json.dumps(validate_manifest(manifest_path), indent=2, sort_keys=True))
    else:
        print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
