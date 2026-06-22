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
SUPPORTED_GENERATOR_FUNCTIONS = {
    "add",
    "relu",
    "mul",
    "matmul",
    "rms_norm",
    "softmax",
    "dot_i8",
    "matvec_i8",
    "matmul_i8",
    "dot_w4a8",
    "matvec_w4a8",
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
    layout: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_json(cls, value: Any, context: str) -> TensorSpec:
        data = _require_mapping(value, context)
        dtype = _require_non_empty_string(data.get("dtype"), f"{context}.dtype")
        if dtype not in SUPPORTED_DTYPES:
            raise GoldenSchemaError(f"{context}.dtype unsupported: {dtype}")
        layout = data.get("layout", {})
        if not isinstance(layout, dict):
            raise GoldenSchemaError(f"{context}.layout must be an object")
        return cls(
            name=_require_non_empty_string(data.get("name"), f"{context}.name"),
            shape=_validate_shape(data.get("shape"), f"{context}.shape"),
            dtype=dtype,
            layout=layout,
        )

    def to_json(self) -> dict[str, Any]:
        value = {"name": self.name, "shape": list(self.shape), "dtype": self.dtype}
        if self.layout:
            value["layout"] = dict(self.layout)
        return value


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
        if function not in SUPPORTED_GENERATOR_FUNCTIONS:
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
            if (
                "rounding" not in quantization
                or "saturation" not in quantization
                or "accumulator_dtype" not in quantization
            ):
                raise GoldenSchemaError(
                    "quantized compare policies require rounding, saturation, "
                    "and accumulator_dtype fields"
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
        if self.kernel in {"add", "mul"}:
            self._validate_binary_f32()
        elif self.kernel == "relu":
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
        elif self.kernel == "rms_norm":
            self._validate_rms_norm()
        elif self.kernel == "softmax":
            self._validate_softmax()
        elif self.kernel == "dot_i8":
            self._validate_dot_i8()
        elif self.kernel == "matvec_i8":
            self._validate_matvec_i8()
        elif self.kernel == "matmul_i8":
            self._validate_matmul_i8()
        elif self.kernel == "dot_w4a8":
            self._validate_dot_w4a8()
        elif self.kernel == "matvec_w4a8":
            self._validate_matvec_w4a8()
        else:
            raise GoldenSchemaError(f"kernel unsupported: {self.kernel}")
        for vlen in self.vlens:
            if not isinstance(vlen, int) or vlen <= 0:
                raise GoldenSchemaError("vlens must contain positive integers")

    def _validate_binary_f32(self) -> None:
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise GoldenSchemaError(f"{self.kernel} cases require two inputs and one output")
        lhs, rhs = self.inputs
        out = self.outputs[0]
        if lhs.shape != rhs.shape or lhs.shape != out.shape:
            raise GoldenSchemaError(f"{self.kernel} input and output shapes must match")
        if {lhs.dtype, rhs.dtype, out.dtype} != {"float32"}:
            raise GoldenSchemaError(f"{self.kernel} cases must use float32 tensors")
        if self.compare.mode != "allclose":
            raise GoldenSchemaError(f"{self.kernel} cases require allclose comparison")

    def _validate_rms_norm(self) -> None:
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise GoldenSchemaError("rms_norm cases require input, weight, and output")
        input_tensor, weight = self.inputs
        out = self.outputs[0]
        if len(input_tensor.shape) != 2 or len(weight.shape) != 1:
            raise GoldenSchemaError(
                "rms_norm cases require input [outer, inner] and weight [inner]"
            )
        if out.shape != input_tensor.shape or weight.shape != (input_tensor.shape[1],):
            raise GoldenSchemaError(
                "rms_norm output/weight shapes must match input inner dimension"
            )
        if {input_tensor.dtype, weight.dtype, out.dtype} != {"float32"}:
            raise GoldenSchemaError("rms_norm cases must use float32 tensors")
        if self.compare.mode != "allclose":
            raise GoldenSchemaError("rms_norm cases require allclose comparison")
        eps = self.compare.quantization.get("epsilon")
        if not isinstance(eps, int | float) or float(eps) <= 0.0:
            raise GoldenSchemaError("rms_norm cases require positive compare.epsilon")

    def _validate_softmax(self) -> None:
        if len(self.inputs) != 1 or len(self.outputs) != 1:
            raise GoldenSchemaError("softmax cases require one input and one output")
        input_tensor = self.inputs[0]
        out = self.outputs[0]
        if len(input_tensor.shape) != 2 or out.shape != input_tensor.shape:
            raise GoldenSchemaError("softmax cases require matching rank-2 tensors")
        if input_tensor.dtype != "float32" or out.dtype != "float32":
            raise GoldenSchemaError("softmax cases must use float32 tensors")
        if self.compare.mode != "allclose":
            raise GoldenSchemaError("softmax cases require allclose comparison")

    def _require_quant_policy(self, *, output_dtype: str) -> None:
        policy = self.compare.quantization
        required = {
            "accumulator_dtype",
            "input_scale",
            "input_zero_point",
            "weight_zero_point",
            "rounding",
            "saturation",
        }
        missing = sorted(required - policy.keys())
        if missing:
            raise GoldenSchemaError(
                "quantized cases require compare policy fields: " + ", ".join(missing)
            )
        if policy["accumulator_dtype"] != output_dtype:
            raise GoldenSchemaError("accumulator_dtype must match quantized output dtype")
        if not isinstance(policy["input_zero_point"], int):
            raise GoldenSchemaError("input_zero_point must be an integer")
        if not isinstance(policy["weight_zero_point"], int):
            raise GoldenSchemaError("weight_zero_point must be an integer")
        if float(policy["input_scale"]) <= 0.0:
            raise GoldenSchemaError("input_scale must be positive")

    def _require_w4_layout(self, tensor: TensorSpec, *, rank: int) -> None:
        layout = tensor.layout
        if layout.get("kind") != "w4a8_packed":
            raise GoldenSchemaError(f"{tensor.name} requires w4a8_packed layout")
        logical_shape = layout.get("logical_shape")
        if not isinstance(logical_shape, list) or len(logical_shape) != rank:
            raise GoldenSchemaError(f"{tensor.name}.layout.logical_shape rank mismatch")
        if any(not isinstance(dim, int) or dim <= 0 for dim in logical_shape):
            raise GoldenSchemaError(f"{tensor.name}.layout.logical_shape must be positive")
        if layout.get("nibble_order") != "low_even_high_odd":
            raise GoldenSchemaError(f"{tensor.name}.layout.nibble_order unsupported")
        group_size = layout.get("group_size")
        if not isinstance(group_size, int) or group_size <= 0:
            raise GoldenSchemaError(f"{tensor.name}.layout.group_size must be positive")

    def _validate_dot_i8(self) -> None:
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise GoldenSchemaError("dot_i8 cases require two inputs and one output")
        lhs, rhs = self.inputs
        out = self.outputs[0]
        if len(lhs.shape) != 1 or len(rhs.shape) != 1 or out.shape != (1,):
            raise GoldenSchemaError("dot_i8 cases require rank-1 inputs and scalar output [1]")
        if lhs.shape != rhs.shape:
            raise GoldenSchemaError("dot_i8 input lengths must match")
        if lhs.dtype != "int8" or rhs.dtype != "int8" or out.dtype != "int32":
            raise GoldenSchemaError("dot_i8 cases require int8 inputs and int32 output")
        if self.compare.mode != "quantized_exact":
            raise GoldenSchemaError("dot_i8 cases require quantized_exact comparison")
        self._require_quant_policy(output_dtype="int32")

    def _validate_matvec_i8(self) -> None:
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise GoldenSchemaError("matvec_i8 cases require two inputs and one output")
        vector, weights = self.inputs
        out = self.outputs[0]
        if len(vector.shape) != 1 or len(weights.shape) != 2 or len(out.shape) != 1:
            raise GoldenSchemaError("matvec_i8 cases require vector, matrix, vector tensors")
        rows, cols = weights.shape
        if vector.shape[0] != cols or out.shape != (rows,):
            raise GoldenSchemaError(
                "matvec_i8 shape contract is input [cols], weights [rows, cols]"
            )
        if vector.dtype != "int8" or weights.dtype != "int8" or out.dtype != "int32":
            raise GoldenSchemaError("matvec_i8 cases require int8 inputs and int32 output")
        if self.compare.mode != "quantized_exact":
            raise GoldenSchemaError("matvec_i8 cases require quantized_exact comparison")
        self._require_quant_policy(output_dtype="int32")

    def _validate_matmul_i8(self) -> None:
        if len(self.inputs) != 2 or len(self.outputs) != 1:
            raise GoldenSchemaError("matmul_i8 cases require two inputs and one output")
        lhs, rhs = self.inputs
        out = self.outputs[0]
        if len(lhs.shape) != 2 or len(rhs.shape) != 2 or len(out.shape) != 2:
            raise GoldenSchemaError("matmul_i8 cases require rank-2 tensors")
        if lhs.shape[1] != rhs.shape[0] or out.shape != (lhs.shape[0], rhs.shape[1]):
            raise GoldenSchemaError("matmul_i8 output shape must be [M, N]")
        if lhs.dtype != "int8" or rhs.dtype != "int8" or out.dtype != "int32":
            raise GoldenSchemaError("matmul_i8 cases require int8 inputs and int32 output")
        if self.compare.mode != "quantized_exact":
            raise GoldenSchemaError("matmul_i8 cases require quantized_exact comparison")
        self._require_quant_policy(output_dtype="int32")

    def _validate_dot_w4a8(self) -> None:
        if len(self.inputs) != 3 or len(self.outputs) != 1:
            raise GoldenSchemaError("dot_w4a8 cases require input, packed weight, scales, output")
        vector, packed_weight, scales = self.inputs
        out = self.outputs[0]
        self._require_w4_layout(packed_weight, rank=1)
        logical_k = packed_weight.layout["logical_shape"][0]
        groups = (logical_k + packed_weight.layout["group_size"] - 1) // packed_weight.layout[
            "group_size"
        ]
        if vector.shape != (logical_k,) or packed_weight.shape != ((logical_k + 1) // 2,):
            raise GoldenSchemaError("dot_w4a8 packed layout does not match input length")
        if scales.shape != (groups,) or out.shape != (1,):
            raise GoldenSchemaError("dot_w4a8 scale/output shapes are invalid")
        if vector.dtype != "int8" or packed_weight.dtype != "uint8":
            raise GoldenSchemaError("dot_w4a8 requires int8 input and uint8 packed weight")
        if scales.dtype != "float32" or out.dtype != "float32":
            raise GoldenSchemaError("dot_w4a8 requires float32 scales and output")
        self._require_quant_policy(output_dtype="float32")

    def _validate_matvec_w4a8(self) -> None:
        if len(self.inputs) != 3 or len(self.outputs) != 1:
            raise GoldenSchemaError(
                "matvec_w4a8 cases require input, packed weights, scales, output"
            )
        vector, packed_weights, scales = self.inputs
        out = self.outputs[0]
        self._require_w4_layout(packed_weights, rank=2)
        rows, logical_cols = packed_weights.layout["logical_shape"]
        group_size = packed_weights.layout["group_size"]
        packed_cols = (logical_cols + 1) // 2
        groups = (logical_cols + group_size - 1) // group_size
        if vector.shape != (logical_cols,) or packed_weights.shape != (rows, packed_cols):
            raise GoldenSchemaError("matvec_w4a8 packed layout does not match input shape")
        if scales.shape != (rows, groups) or out.shape != (rows,):
            raise GoldenSchemaError("matvec_w4a8 scale/output shapes are invalid")
        if vector.dtype != "int8" or packed_weights.dtype != "uint8":
            raise GoldenSchemaError("matvec_w4a8 requires int8 input and uint8 packed weights")
        if scales.dtype != "float32" or out.dtype != "float32":
            raise GoldenSchemaError("matvec_w4a8 requires float32 scales and output")
        self._require_quant_policy(output_dtype="float32")

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
        layout = data.get("layout", {})
        if not isinstance(layout, dict):
            raise GoldenSchemaError(f"layout mismatch for {filename}")
    return manifest
