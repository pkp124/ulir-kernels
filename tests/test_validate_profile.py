from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = REPO_ROOT / "scripts" / "validate_profile.py"
RVV_PROFILE = REPO_ROOT / "target" / "riscv_rvv_256.h"
GENERIC_PROFILE = REPO_ROOT / "target" / "generic.h"


def run_validator(*profiles: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(VALIDATOR), *(str(profile) for profile in profiles)],
        check=False,
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def test_validate_profile_accepts_existing_profiles() -> None:
    result = run_validator(GENERIC_PROFILE, RVV_PROFILE)

    assert result.returncode == 0, result.stderr
    assert "target/generic.h: ok" in result.stdout
    assert "target/riscv_rvv_256.h: ok" in result.stdout


def test_validate_profile_requires_rvv_quantized_fields(tmp_path: Path) -> None:
    profile_text = "\n".join(
        line
        for line in RVV_PROFILE.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#define KS_QUANT_DOT_I8_TILE_K")
    )
    profile = tmp_path / "missing_quant_field.h"
    profile.write_text(profile_text, encoding="utf-8")

    result = run_validator(profile)

    assert result.returncode == 1
    assert "missing required RVV quantized field: KS_QUANT_DOT_I8_TILE_K" in result.stderr
