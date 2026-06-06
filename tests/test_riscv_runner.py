import importlib.util
import sys
from pathlib import Path

import pytest

_RUNNER_PATH = Path(__file__).resolve().parent / "riscv_runner.py"
_SPEC = importlib.util.spec_from_file_location("riscv_runner", _RUNNER_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_RUNNER = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _RUNNER
_SPEC.loader.exec_module(_RUNNER)

CaseResult = _RUNNER.CaseResult
validate_result_markers = _RUNNER.validate_result_markers
validate_vlens = _RUNNER.validate_vlens


def test_rvv_256_profile_rejects_vlen_below_baseline():
    case = {"name": "vector_relu_f32", "profile": "riscv_rvv_256"}

    with pytest.raises(ValueError, match="zvl256b baseline"):
        validate_vlens(case, [128])


def test_result_requires_pass_marker_and_error_marker():
    case = {"name": "vector_relu_f32", "tolerance": 1.0e-6, "benchmark": True}
    result = CaseResult(
        name="vector_relu_f32",
        vlen=256,
        return_code=0,
        stdout="MAX_ABS_ERROR: 0\nTIME_NS: 42\nPASS\n",
        stderr="",
        max_abs_error=0.0,
        time_ns=42,
    )

    assert result.passed
    assert validate_result_markers(case, [result])


def test_result_fails_when_error_exceeds_tolerance():
    case = {"name": "vector_relu_f32", "tolerance": 1.0e-6, "benchmark": True}
    result = CaseResult(
        name="vector_relu_f32",
        vlen=256,
        return_code=0,
        stdout="MAX_ABS_ERROR: 0.01\nTIME_NS: 42\nPASS\n",
        stderr="",
        max_abs_error=0.01,
        time_ns=42,
    )

    assert not validate_result_markers(case, [result])
