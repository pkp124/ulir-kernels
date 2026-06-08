#!/usr/bin/env python3
"""Run descriptor-backed functional verification for KernelSmith targets."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Direct CTest execution starts from the build tree, so local imports need the
# source root on sys.path before these module-level imports run.
from tests.functional_validator import compare_arrays  # noqa: E402
from tests.golden.schema import (  # noqa: E402
    CaseDescriptor,
    canonical_json_sha256,
    load_case_descriptor,
    sha256_file,
    validate_manifest,
)
from tests.riscv_runner import (  # noqa: E402
    DEFAULT_RVV_VLENS,
    RiscVRunner,
    RiscVRunnerError,
    validate_vlens,
)


class VerifyError(RuntimeError):
    """Raised when a verification setup or execution step fails."""


def _repo_root() -> Path:
    return REPO_ROOT


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _emit_result(result: dict[str, Any]) -> None:
    print(json.dumps(_json_safe(result), sort_keys=True, separators=(",", ":")))


def _product(shape: tuple[int, ...]) -> int:
    total = 1
    for dim in shape:
        total *= dim
    return total


def _tensor_entry(manifest: dict[str, Any], *, role: str, name: str) -> dict[str, Any]:
    for tensor in manifest["tensors"]:
        if tensor["role"] == role and tensor["name"] == name:
            return tensor
    raise VerifyError(f"manifest is missing {role} tensor {name}")


def _manifest_path(case: CaseDescriptor, manifest_dir: Path) -> Path:
    return manifest_dir / case.name / "manifest.json"


def _validate_descriptor_hash(case: CaseDescriptor, manifest: dict[str, Any]) -> None:
    expected = manifest.get("descriptor_sha256")
    actual = canonical_json_sha256(case.to_json())
    if expected != actual:
        raise VerifyError(f"descriptor hash mismatch: expected {expected}, got {actual}")


def _load_bundle_array(bundle_dir: Path, entry: dict[str, Any]) -> np.ndarray:
    path = bundle_dir / entry["filename"]
    array = np.load(path)
    return np.ascontiguousarray(array)


def _write_raw_input(
    *,
    bundle_dir: Path,
    raw_dir: Path,
    manifest: dict[str, Any],
    tensor_name: str,
) -> Path:
    entry = _tensor_entry(manifest, role="input", name=tensor_name)
    array = _load_bundle_array(bundle_dir, entry)
    raw_path = raw_dir / f"input_{tensor_name}.f32"
    array.astype(np.float32, copy=False).tofile(raw_path)
    return raw_path


def _load_expected_output(
    *,
    manifest: dict[str, Any],
    bundle_dir: Path,
    output_name: str,
) -> tuple[np.ndarray, Path, dict[str, Any]]:
    expected_entry = _tensor_entry(manifest, role="expected_output", name=output_name)
    expected_path = bundle_dir / expected_entry["filename"]
    return np.load(expected_path), expected_path, expected_entry


def _find_default_host_runner() -> Path | None:
    root = _repo_root()
    candidates = [
        root / "build" / "bin" / "host-reference-runner",
        root / "build" / "tests" / "host_reference" / "host-reference-runner",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _find_default_riscv_runner() -> Path | None:
    root = _repo_root()
    candidates = [
        root / "build-rvv" / "bin" / "riscv-golden-runner",
        root / "build" / "tests" / "riscv" / "riscv-golden-runner",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _case_runner_args(
    *,
    case: CaseDescriptor,
    manifest: dict[str, Any],
    bundle_dir: Path,
    raw_dir: Path,
    raw_output: Path,
) -> list[str]:
    if case.kernel == "relu":
        raw_input = _write_raw_input(
            bundle_dir=bundle_dir,
            raw_dir=raw_dir,
            manifest=manifest,
            tensor_name=case.inputs[0].name,
        )
        return [
            "relu",
            str(raw_input),
            str(raw_output),
            str(_product(case.inputs[0].shape)),
        ]

    if case.kernel == "matmul":
        lhs, rhs = case.inputs
        raw_lhs = _write_raw_input(
            bundle_dir=bundle_dir,
            raw_dir=raw_dir,
            manifest=manifest,
            tensor_name=lhs.name,
        )
        raw_rhs = _write_raw_input(
            bundle_dir=bundle_dir,
            raw_dir=raw_dir,
            manifest=manifest,
            tensor_name=rhs.name,
        )
        m, k = lhs.shape
        rhs_k, n = rhs.shape
        if k != rhs_k:
            raise VerifyError("matmul descriptor has mismatched inner dimensions")
        return [
            "matmul",
            str(raw_lhs),
            str(raw_rhs),
            str(raw_output),
            str(m),
            str(n),
            str(k),
        ]

    raise VerifyError(f"target runner does not support kernel {case.kernel}")


def _run_host_reference(
    *,
    case: CaseDescriptor,
    manifest: dict[str, Any],
    bundle_dir: Path,
    output_dir: Path,
    host_runner: Path,
) -> list[dict[str, Any]]:
    if not host_runner.exists():
        raise VerifyError(f"host runner not found: {host_runner}")

    case_dir = output_dir / case.name / "host_reference"
    raw_dir = case_dir / "raw"
    case_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    output = case.outputs[0]
    raw_output = raw_dir / f"actual_{output.name}.f32"
    command = [
        str(host_runner),
        *_case_runner_args(
            case=case,
            manifest=manifest,
            bundle_dir=bundle_dir,
            raw_dir=raw_dir,
            raw_output=raw_output,
        ),
    ]

    completed = subprocess.run(command, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise VerifyError(
            f"host runner failed with exit code {completed.returncode}: {completed.stderr.strip()}"
        )

    actual = np.fromfile(raw_output, dtype=np.float32)
    expected_size = _product(output.shape)
    if actual.size != expected_size:
        raise VerifyError(
            f"host output size mismatch for {output.name}: "
            f"got {actual.size}, expected {expected_size}"
        )
    actual = actual.reshape(output.shape).astype(np.dtype(output.dtype), copy=False)
    actual_path = case_dir / f"actual_{output.name}.npy"
    np.save(actual_path, actual)

    expected, expected_path, expected_entry = _load_expected_output(
        manifest=manifest,
        bundle_dir=bundle_dir,
        output_name=output.name,
    )
    comparison = compare_arrays(actual, expected, manifest["compare"])

    return [
        {
            "kind": "kernelsmith_result",
            "case": case.name,
            "target": "host_reference",
            "kernel": case.kernel,
            "output": output.name,
            "status": "PASS" if comparison.passed else "FAIL",
            "passed": comparison.passed,
            "actual_path": str(actual_path),
            "actual_sha256": sha256_file(actual_path),
            "expected_path": str(expected_path),
            "expected_sha256": expected_entry["sha256"],
            "shape": list(output.shape),
            "dtype": output.dtype,
            "compare": comparison.to_dict(),
        }
    ]


def _failure_result(
    *,
    case: CaseDescriptor,
    target: str,
    output_name: str,
    vlen: int,
    details: str,
    return_code: int | None = None,
    stdout: str = "",
    stderr: str = "",
) -> dict[str, Any]:
    return {
        "kind": "kernelsmith_result",
        "case": case.name,
        "target": target,
        "kernel": case.kernel,
        "output": output_name,
        "vlen": vlen,
        "status": "FAIL",
        "passed": False,
        "compare": {
            "passed": False,
            "max_error": float("inf"),
            "mean_error": float("inf"),
            "max_relative_error": float("inf"),
            "mismatch_count": 1,
            "details": details,
        },
        "qemu": {
            "return_code": return_code,
            "stdout": stdout,
            "stderr": stderr,
        },
    }


def _run_riscv_reference(
    *,
    case: CaseDescriptor,
    target: str,
    manifest: dict[str, Any],
    bundle_dir: Path,
    output_dir: Path,
    riscv_runner: Path,
    qemu_binary: str,
    vlens: tuple[int, ...],
    timeout: int,
    host_runner: Path | None,
) -> list[dict[str, Any]]:
    if not riscv_runner.exists():
        raise VerifyError(f"RISC-V runner not found: {riscv_runner}")

    host_results: dict[str, dict[str, Any]] = {}
    if host_runner is not None:
        for host_result in _run_host_reference(
            case=case,
            manifest=manifest,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            host_runner=host_runner,
        ):
            if not host_result["passed"]:
                raise VerifyError(
                    f"host_reference failed for {case.name}; refusing RISC-V comparison"
                )
            host_results[host_result["output"]] = host_result

    try:
        runner = RiscVRunner(qemu_binary)
    except RiscVRunnerError as error:
        raise VerifyError(str(error)) from error

    results = []
    output = case.outputs[0]
    expected, expected_path, expected_entry = _load_expected_output(
        manifest=manifest,
        bundle_dir=bundle_dir,
        output_name=output.name,
    )

    for vlen in vlens:
        case_dir = output_dir / case.name / target / f"vlen_{vlen}"
        raw_dir = case_dir / "raw"
        case_dir.mkdir(parents=True, exist_ok=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        raw_output = raw_dir / f"actual_{output.name}.f32"
        args = _case_runner_args(
            case=case,
            manifest=manifest,
            bundle_dir=bundle_dir,
            raw_dir=raw_dir,
            raw_output=raw_output,
        )
        run_result = runner.run(riscv_runner, vlen=vlen, args=args, timeout=timeout)
        if not run_result.success:
            results.append(
                _failure_result(
                    case=case,
                    target=target,
                    output_name=output.name,
                    vlen=vlen,
                    details=f"qemu failed with exit code {run_result.return_code}",
                    return_code=run_result.return_code,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                )
            )
            continue

        if not raw_output.exists():
            results.append(
                _failure_result(
                    case=case,
                    target=target,
                    output_name=output.name,
                    vlen=vlen,
                    details=f"runner did not write expected output: {raw_output}",
                    return_code=run_result.return_code,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                )
            )
            continue

        actual = np.fromfile(raw_output, dtype=np.float32)
        expected_size = _product(output.shape)
        if actual.size != expected_size:
            results.append(
                _failure_result(
                    case=case,
                    target=target,
                    output_name=output.name,
                    vlen=vlen,
                    details=(
                        f"RISC-V output size mismatch for {output.name}: "
                        f"got {actual.size}, expected {expected_size}"
                    ),
                    return_code=run_result.return_code,
                    stdout=run_result.stdout,
                    stderr=run_result.stderr,
                )
            )
            continue

        actual = actual.reshape(output.shape).astype(np.dtype(output.dtype), copy=False)
        actual_path = case_dir / f"actual_{output.name}.npy"
        np.save(actual_path, actual)

        comparison = compare_arrays(actual, expected, manifest["compare"])
        host_comparison = None
        if output.name in host_results:
            host_actual = np.load(host_results[output.name]["actual_path"])
            host_comparison = compare_arrays(actual, host_actual, manifest["compare"])

        passed = comparison.passed and (
            host_comparison is None or host_comparison.passed
        )
        result = {
            "kind": "kernelsmith_result",
            "case": case.name,
            "target": target,
            "kernel": case.kernel,
            "output": output.name,
            "vlen": vlen,
            "status": "PASS" if passed else "FAIL",
            "passed": passed,
            "actual_path": str(actual_path),
            "actual_sha256": sha256_file(actual_path),
            "expected_path": str(expected_path),
            "expected_sha256": expected_entry["sha256"],
            "shape": list(output.shape),
            "dtype": output.dtype,
            "compare": comparison.to_dict(),
            "qemu": {
                "return_code": run_result.return_code,
                "stdout": run_result.stdout,
                "stderr": run_result.stderr,
            },
        }
        if host_comparison is not None:
            result["host_compare"] = host_comparison.to_dict()
            result["host_actual_path"] = host_results[output.name]["actual_path"]
        results.append(result)

    return results


def verify_case(
    *,
    case_path: Path,
    target: str,
    manifest_dir: Path,
    output_dir: Path,
    host_runner: Path | None,
    riscv_runner: Path | None = None,
    qemu_binary: str = "qemu-riscv64",
    vlens: tuple[int, ...] | list[int] | None = None,
    timeout: int = 60,
) -> list[dict[str, Any]]:
    case = load_case_descriptor(case_path)
    if target not in case.targets:
        raise VerifyError(f"target {target} is not declared by case {case.name}")

    manifest_path = _manifest_path(case, manifest_dir)
    manifest = validate_manifest(manifest_path)
    _validate_descriptor_hash(case, manifest)
    bundle_dir = manifest_path.parent

    if target == "host_reference":
        runner = host_runner or _find_default_host_runner()
        if runner is None:
            raise VerifyError("host runner not found; pass --host-runner or build CTest targets")
        return _run_host_reference(
            case=case,
            manifest=manifest,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            host_runner=runner,
        )

    if target.startswith("riscv_rvv_"):
        selected_vlens = tuple(vlens or case.vlens or manifest.get("vlens") or DEFAULT_RVV_VLENS)
        try:
            selected_vlens = validate_vlens(target, selected_vlens)
        except RiscVRunnerError as error:
            raise VerifyError(str(error)) from error

        runner = riscv_runner or _find_default_riscv_runner()
        if runner is None:
            raise VerifyError("RISC-V runner not found; pass --riscv-runner or build CI artifacts")
        return _run_riscv_reference(
            case=case,
            target=target,
            manifest=manifest,
            bundle_dir=bundle_dir,
            output_dir=output_dir,
            riscv_runner=runner,
            qemu_binary=qemu_binary,
            vlens=selected_vlens,
            timeout=timeout,
            host_runner=host_runner,
        )

    raise VerifyError(f"unsupported target: {target}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path, help="Path to a case descriptor JSON")
    parser.add_argument("--target", required=True, help="Execution target to verify")
    parser.add_argument(
        "--manifest-dir",
        default=_repo_root() / "tests" / "golden" / "generated",
        type=Path,
        help="Directory containing generated golden bundles",
    )
    parser.add_argument(
        "--output-dir",
        default=_repo_root() / "build" / "golden-actual",
        type=Path,
        help="Directory where captured target outputs and reports are written",
    )
    parser.add_argument(
        "--host-runner",
        type=Path,
        help="Path to host-reference-runner for --target host_reference",
    )
    parser.add_argument(
        "--riscv-runner",
        type=Path,
        help="Path to riscv-golden-runner for --target riscv_rvv_*",
    )
    parser.add_argument("--qemu", default="qemu-riscv64", help="QEMU user-mode binary")
    parser.add_argument("--vlens", nargs="+", type=int, help="RVV VLEN values to run")
    parser.add_argument("--timeout", type=int, default=60, help="Per-QEMU-run timeout seconds")
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional JSON report path; defaults under --output-dir",
    )
    args = parser.parse_args(argv)

    try:
        results = verify_case(
            case_path=args.case,
            target=args.target,
            manifest_dir=args.manifest_dir,
            output_dir=args.output_dir,
            host_runner=args.host_runner,
            riscv_runner=args.riscv_runner,
            qemu_binary=args.qemu,
            vlens=args.vlens,
            timeout=args.timeout,
        )
    except Exception as error:
        result = {
            "kind": "kernelsmith_result",
            "case": args.case.stem,
            "target": args.target,
            "status": "ERROR",
            "passed": False,
            "details": str(error),
        }
        _emit_result(result)
        return 1

    report = {
        "kind": "kernelsmith_report",
        "target": args.target,
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    report_path = args.report or args.output_dir / f"{args.case.stem}_{args.target}_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(_json_safe(report), indent=2, sort_keys=True) + "\n")

    for result in results:
        _emit_result(result)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
