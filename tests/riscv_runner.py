#!/usr/bin/env python3
"""Run RISC-V Linux binaries under QEMU user-mode with RVV VLEN checks."""

from __future__ import annotations

import argparse
import json
import stat
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

DEFAULT_RVV_VLENS = (256, 512)


class RiscVRunnerError(RuntimeError):
    """Raised when RISC-V simulation setup or execution fails."""


@dataclass(frozen=True)
class RiscVRunResult:
    """Captured output from one QEMU user-mode execution."""

    vlen: int
    command: list[str]
    return_code: int
    stdout: str
    stderr: str

    @property
    def success(self) -> bool:
        return self.return_code == 0


def profile_min_vlen(profile: str) -> int:
    """Return the architectural minimum VLEN encoded by an RVV profile name."""
    prefix = "riscv_rvv_"
    if not profile.startswith(prefix):
        raise RiscVRunnerError(f"unsupported RISC-V profile: {profile}")
    try:
        return int(profile[len(prefix) :])
    except ValueError as error:
        raise RiscVRunnerError(f"unsupported RISC-V profile: {profile}") from error


def validate_vlens(profile: str, vlens: tuple[int, ...] | list[int] | None) -> tuple[int, ...]:
    """Reject VLENs below the compiled artifact baseline."""
    selected = tuple(vlens or DEFAULT_RVV_VLENS)
    if not selected:
        raise RiscVRunnerError("at least one VLEN must be provided")

    minimum = profile_min_vlen(profile)
    for vlen in selected:
        if vlen < minimum:
            raise RiscVRunnerError(f"VLEN {vlen} is below {profile} minimum VLEN {minimum}")
    return selected


def qemu_cpu_arg(vlen: int) -> str:
    """Build the QEMU RVV CPU argument for a specific vector length."""
    return f"rv64,v=true,vlen={vlen},vext_spec=v1.0"


def repair_executable_permission(path: Path) -> bool:
    """Restore executable bits commonly stripped from downloaded CI artifacts."""
    if not path.exists():
        raise RiscVRunnerError(f"RISC-V binary not found: {path}")

    executable_bits = stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH
    mode = path.stat().st_mode
    if mode & executable_bits:
        return False
    path.chmod(mode | executable_bits)
    return True


class RiscVRunner:
    """Execute RISC-V ELF binaries on QEMU user-mode."""

    def __init__(self, qemu_binary: str = "qemu-riscv64"):
        self.qemu_binary = qemu_binary
        self._check_qemu_available()

    def _check_qemu_available(self) -> None:
        try:
            subprocess.run([self.qemu_binary, "--version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as error:
            raise RiscVRunnerError(
                f"{self.qemu_binary} not found. Install with: apt install qemu-user"
            ) from error

    def command(self, binary_path: Path, vlen: int, args: list[str] | None = None) -> list[str]:
        command = [
            self.qemu_binary,
            "-cpu",
            qemu_cpu_arg(vlen),
            str(binary_path),
        ]
        if args:
            command.extend(args)
        return command

    def run(
        self,
        binary_path: Path,
        *,
        vlen: int,
        args: list[str] | None = None,
        timeout: int = 60,
    ) -> RiscVRunResult:
        repair_executable_permission(binary_path)
        command = self.command(binary_path, vlen, args)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            return RiscVRunResult(
                vlen=vlen,
                command=command,
                return_code=completed.returncode,
                stdout=completed.stdout,
                stderr=completed.stderr,
            )
        except subprocess.TimeoutExpired as error:
            return RiscVRunResult(
                vlen=vlen,
                command=command,
                return_code=-1,
                stdout=error.stdout or "",
                stderr=f"Timeout ({timeout}s) running {binary_path}",
            )


def _emit_run_result(binary: Path, result: RiscVRunResult) -> None:
    print(
        json.dumps(
            {
                "kind": "kernelsmith_riscv_run",
                "case": binary.stem,
                "vlen": result.vlen,
                "status": "PASS" if result.success else "FAIL",
                "return_code": result.return_code,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", required=True, type=Path, help="RISC-V ELF to run")
    parser.add_argument("--profile", default="riscv_rvv_256", help="Target profile name")
    parser.add_argument(
        "--vlens",
        nargs="+",
        type=int,
        default=list(DEFAULT_RVV_VLENS),
        help="RVV VLEN values to test",
    )
    parser.add_argument("--qemu", default="qemu-riscv64", help="QEMU user-mode binary")
    parser.add_argument("--timeout", type=int, default=60, help="Per-run timeout in seconds")
    parser.add_argument("binary_args", nargs=argparse.REMAINDER, help="Arguments after --")
    args = parser.parse_args(argv)

    try:
        vlens = validate_vlens(args.profile, args.vlens)
        runner = RiscVRunner(args.qemu)
    except RiscVRunnerError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1

    binary_args = args.binary_args
    if binary_args and binary_args[0] == "--":
        binary_args = binary_args[1:]

    passed = True
    for vlen in vlens:
        result = runner.run(args.binary, vlen=vlen, args=binary_args, timeout=args.timeout)
        _emit_run_result(args.binary, result)
        passed = passed and result.success
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
