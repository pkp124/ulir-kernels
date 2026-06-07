"""Descriptor and manifest schema helpers for golden-reference cases."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA_VERSION = 1
GENERATOR_BACKEND = "numpy"
SUPPORTED_DTYPES = {"float32", "float16", "int8", "uint8", "int32"}
SUPPORTED_COMPARE_MODES = {
    "exact",
    "allclose",
    "quantized_exact",
    "dequantized_allclose",
}


class GoldenSchemaError(ValueError):
    """Raised when a golden descriptor or manifest violates the schema."""


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest for a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: dict[str, Any]) -> str:
    """Hash a JSON-compatible value using the canonical manifest encoding."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _require_mapping(value: Any, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise GoldenSchemaError(f"{context} must be an object")
    return value


def _require_non_empty_string(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value:
        raise GoldenSchemaError(f"{context} must be a non-empty string")
    return value


def _require_bool(value: Any, context: str) -> bool:
    if not isinstance(value, bool):
        raise GoldenSchemaError(f"{context} must be a boolean")
    return value


def _validate_shape(value: Any, context: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise GoldenSchemaError(f"{context} must be a non-empty list")
    shape = []
    for dim in value:
        if not isinstance(dim, int) or dim <= 0:
            raise GoldenSchemaError(f"{context} dimensions must be positive integers")
        shape.append(dim)
    return tuple(shape)


@dataclass(frozen=True)
class TensorSpec:
    """A tensor declared by a golden descriptor."""

    name: str
    shape: tuple[int, ...]
    dtype: str

    @classmethod
    def from_json(cls, value: Any, context: str) -> TensorSpec:
        data = _require_mapping(value, context)
        dtype = _require_non_empty_string(data.get("dtype"), f"{context}.dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise GoldenSchemaError(f"{context}.dtype unsupported: {dtype}")
        return cls(
            name=_require_non_empty_string(data.get("name"), f"{context}.name"),
            shape=_validate_shape(data.get("shape"), f"{context}.shape"),
            dtype=dtype,
        )

    def to_json(self) -> dict[str, Any]:
        return {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}


@dataclass(frozen=True)
class GeneratorSpec:
    """The deterministic backend used to create a golden case."""

    kind: str
    function: str
    seed: int
    distribution: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: Any) -> GeneratorSpec:
        data = _require_mapping(value, "generator")
        kind = _require_non_empty_string(data.get("kind"), "generator.kind")
        if kind != GENERATOR_BACKEND:
            raise GoldenSchemaError(f"generator.kind unsupported: {kind}")
        function = _require_non_empty_string(data.get("function"), "generator.function")
        if function not in {"relu", "matmul"}:
            raise GoldenSchemaError(f"generator.function unsupported: {function}")
        seed = data.get("seed")
        if not isinstance(seed, int):
            raise GoldenSchemaError("generator.seed must be an integer")
        distribution = data.get("distribution", {"kind": "uniform", "low": -1.0, "high": 1.0})
        if not isinstance(distribution, dict):
            raise GoldenSchemaError("generator.distribution must be an object")
        return cls(kind=kind, function=function, seed=seed, distribution=distribution)

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "function": self.function,
            "seed": self.seed,
            "distribution": dict(self.distribution),
        }


@dataclass(frozen=True)
class ComparePolicy:
    """Comparison policy shared by host and target verification."""

    mode: str
    rtol: float = 0.0
    atol: float = 0.0
    strict_shape: bool = True
    strict_dtype: bool = True
    quantization: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: Any) -> ComparePolicy:
        data = _require_mapping(value, "compare")
        mode = _require_non_empty_string(data.get("mode"), "compare.mode")
        if mode not in SUPPORTED_COMPARE_MODES:
            raise GoldenSchemaError(f"compare.mode unsupported: {mode}")
        rtol = float(data.get("rtol", 0.0))
        atol = float(data.get("atol", 0.0))
        strict_shape = _require_bool(data.get("strict_shape", True), "compare.strict_shape")
        strict_dtype = _require_bool(data.get("strict_dtype", True), "compare.strict_dtype")
        quantization = {
            key: value
            for key, value in data.items()
            if key
            not in {
                "mode",
                "rtol",
                "atol",
                "strict_shape",
                "strict_dtype",
            }
        }
        if mode.startswith("quantized") or mode == "dequantized_allclose":
            if "rounding" not in quantization or "saturation" not in quantization:
                raise GoldenSchemaError(
                    "quantized compare policies require rounding and saturation fields"
                )
        return cls(
            mode=mode,
            rtol=rtol,
            atol=atol,
            strict_shape=strict_shape,
            strict_dtype=strict_dtype,
            quantization=quantization,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "rtol": self.rtol,
            "atol": self.atol,
            "strict_shape": self.strict_shape,
            "strict_dtype": self.strict_dtype,
            **self.quantization,
        }


@dataclass(frozen=True)
class CaseDescriptor:
    """A target-neutral golden-reference case descriptor."""

    name: str
    kernel: str
    source: str | None
    generator: GeneratorSpec
    inputs: tuple[TensorSpec, ...]
    outputs: tuple[TensorSpec, ...]
    compare: ComparePolicy
    targets: tuple[str, ...]
    vlens: tuple[int, ...] = ()

    @classmethod
    def from_json(cls, value: Any) -> CaseDescriptor:
        data = _require_mapping(value, "descriptor")
        inputs = data.get("inputs")
        outputs = data.get("outputs")
        targets = data.get("targets")
        vlens = data.get("vlens", [])
        if not isinstance(inputs, list) or not inputs:
            raise GoldenSchemaError("inputs must be a non-empty list")
        if not isinstance(outputs, list) or not outputs:
            raise GoldenSchemaError("outputs must be a non-empty list")
        if not isinstance(targets, list) or not targets:
            raise GoldenSchemaError("targets must be a non-empty list")
        if not isinstance(vlens, list):
            raise GoldenSchemaError("vlens must be a list")
        descriptor = cls(
            name=_require_non_empty_string(data.get("name"), "name"),
            kernel=_require_non_empty_string(data.get("kernel"), "kernel"),
            source=data.get("source"),
            generator=GeneratorSpec.from_json(data.get("generator")),
            inputs=tuple(
                TensorSpec.from_json(item, f"inputs[{index}]") for index, item in enumerate(inputs)
            ),
            outputs=tuple(
                TensorSpec.from_json(item, f"outputs[{index}]")
                for index, item in enumerate(outputs)
            ),
            compare=ComparePolicy.from_json(data.get("compare")),
            targets=tuple(_require_non_empty_string(item, "targets[]") for item in targets),
            vlens=tuple(vlen for vlen in vlens),
        )
        descriptor.validate_semantics()
        return descriptor

    def validate_semantics(self) -> None:
        if self.generator.function != self.kernel:
            raise GoldenSchemaError("generator.function must match kernel")
        if self.kernel == "relu":
            if len(self.inputs) != 1 or len(self.outputs) != 1:
                raise GoldenSchemaError("relu cases require one input and one output")
            if self.inputs[0].shape != self.outputs[0].shape:
                raise GoldenSchemaError("relu input and output shapes must match")
            if self.inputs[0].dtype != "float32" or self.outputs[0].dtype != "float32":
                raise GoldenSchemaError("relu smoke cases must use float32 tensors")
        elif self.kernel == "matmul":
            if len(self.inputs) != 2 or len(self.outputs) != 1:
                raise GoldenSchemaError("matmul cases require two inputs and one output")
            lhs, rhs = self.inputs
            out = self.outputs[0]
            if len(lhs.shape) != 2 or len(rhs.shape) != 2 or len(out.shape) != 2:
                raise GoldenSchemaError("matmul smoke cases require rank-2 tensors")
            if lhs.shape[1] != rhs.shape[0]:
                raise GoldenSchemaError("matmul inner dimensions must match")
            if out.shape != (lhs.shape[0], rhs.shape[1]):
                raise GoldenSchemaError("matmul output shape must be [M, N]")
            if {lhs.dtype, rhs.dtype, out.dtype} != {"float32"}:
                raise GoldenSchemaError("matmul smoke cases must use float32 tensors")
        else:
            raise GoldenSchemaError(f"kernel unsupported: {self.kernel}")
        for vlen in self.vlens:
            if not isinstance(vlen, int) or vlen <= 0:
                raise GoldenSchemaError("vlens must contain positive integers")

    def to_json(self) -> dict[str, Any]:
        value = {
            "name": self.name,
            "kernel": self.kernel,
            "generator": self.generator.to_json(),
            "inputs": [tensor.to_json() for tensor in self.inputs],
            "outputs": [tensor.to_json() for tensor in self.outputs],
            "compare": self.compare.to_json(),
            "targets": list(self.targets),
        }
        if self.source is not None:
            value["source"] = self.source
        if self.vlens:
            value["vlens"] = list(self.vlens)
        return value


def load_case_descriptor(path: Path) -> CaseDescriptor:
    """Load and validate a case descriptor from JSON."""
    with path.open() as handle:
        return CaseDescriptor.from_json(json.load(handle))


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Write stable, human-readable JSON."""
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    """Validate a generated manifest and return its parsed JSON."""
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise GoldenSchemaError("manifest schema_version is unsupported")
    tensors = manifest.get("tensors")
    if not isinstance(tensors, list) or not tensors:
        raise GoldenSchemaError("manifest tensors must be a non-empty list")
    base_dir = manifest_path.parent
    for tensor in tensors:
        data = _require_mapping(tensor, "manifest tensor")
        filename = _require_non_empty_string(data.get("filename"), "tensor.filename")
        expected_hash = _require_non_empty_string(data.get("sha256"), "tensor.sha256")
        tensor_path = base_dir / filename
        if not tensor_path.exists():
            raise GoldenSchemaError(f"tensor file missing: {filename}")
        actual_hash = sha256_file(tensor_path)
        if actual_hash != expected_hash:
            raise GoldenSchemaError(
                f"hash mismatch for {filename}: expected {expected_hash}, got {actual_hash}"
            )
        array = np.load(tensor_path)
        shape = _validate_shape(data.get("shape"), f"{filename}.shape")
        dtype = _require_non_empty_string(data.get("dtype"), f"{filename}.dtype")
        if tuple(array.shape) != shape:
            raise GoldenSchemaError(f"shape mismatch for {filename}")
        if str(array.dtype) != dtype:
            raise GoldenSchemaError(f"dtype mismatch for {filename}")
        if array.nbytes != data.get("size_bytes"):
            raise GoldenSchemaError(f"size_bytes mismatch for {filename}")
    return manifest
