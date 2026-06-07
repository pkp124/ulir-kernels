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


def generate_case_arrays(descriptor: CaseDescriptor) -> dict[str, np.ndarray]:
    """Generate descriptor inputs and expected outputs."""
    arrays: dict[str, np.ndarray] = {}
    distribution = descriptor.generator.distribution
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
    elif descriptor.kernel == "matmul":
        lhs = arrays[f"input_{descriptor.inputs[0].name}"]
        rhs = arrays[f"input_{descriptor.inputs[1].name}"]
        arrays[f"expected_{descriptor.outputs[0].name}"] = np.matmul(lhs, rhs).astype(np.float32)
    else:
        raise ValueError(f"unsupported kernel: {descriptor.kernel}")

    return arrays


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
