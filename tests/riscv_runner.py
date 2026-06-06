#!/usr/bin/env python3
"""RISC-V simulator runner for KernelSmith functional tests."""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESCRIPTOR = REPO_ROOT / "tests" / "riscv" / "cases.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "build-rvv"


@dataclass(frozen=True)
class TargetProfile:
    """RISC-V target profile constraints relevant to simulator execution."""

    name: str
    min_vlen: int
    default_vlens: tuple[int, ...]


PROFILES = {
    "riscv_rvv_256": TargetProfile("riscv_rvv_256", min_vlen=256, default_vlens=(256, 512)),
}


@dataclass(frozen=True)
class CaseResult:
    """Normalized simulator result for one case and one VLEN."""

    name: str
    vlen: int
    return_code: int
    stdout: str
    stderr: str
    max_abs_error: float | None
    time_ns: int | None

    @property
    def passed(self) -> bool:
        return self.return_code == 0 and "PASS" in self.stdout


def repo_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def load_cases(descriptor: Path) -> list[dict]:
    with descriptor.open() as handle:
        data = json.load(handle)

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{descriptor} must contain a non-empty 'cases' list")

    cases = []
    for raw in raw_cases:
        case = dict(raw)
        name = case.get("name")
        profile_name = case.get("profile", "riscv_rvv_256")
        if not name:
            raise ValueError(f"{descriptor} contains a case without a name")
        if profile_name not in PROFILES:
            raise ValueError(f"{name}: unknown profile '{profile_name}'")
        for field in ("input_mlir", "harness", "entry_symbol"):
            if not case.get(field):
                raise ValueError(f"{name}: missing required field '{field}'")

        case["profile"] = profile_name
        case.setdefault("vlens", list(PROFILES[profile_name].default_vlens))
        case.setdefault("tolerance", 1e-5)
        case.setdefault("benchmark", True)
        validate_vlens(case, case["vlens"])
        cases.append(case)

    return cases


def validate_vlens(case: dict, vlens: list[int]) -> None:
    profile = PROFILES[case["profile"]]
    invalid = [vlen for vlen in vlens if vlen < profile.min_vlen]
    if invalid:
        raise ValueError(
            f"{case['name']}: VLEN {invalid} is below {profile.name}'s "
            f"zvl{profile.min_vlen}b baseline"
        )


def select_cases(cases: list[dict], names: list[str] | None) -> list[dict]:
    if not names:
        return cases

    selected = [case for case in cases if case["name"] in names]
    missing = sorted(set(names) - {case["name"] for case in selected})
    if missing:
        raise ValueError(f"unknown RISC-V functional case(s): {', '.join(missing)}")
    return selected


def check_tool(tool: str, install_hint: str) -> str:
    resolved = shutil.which(tool)
    if not resolved:
        raise RuntimeError(f"{tool} not found. {install_hint}")
    return resolved


def check_build_tools() -> None:
    ks_opt = REPO_ROOT / "build" / "bin" / "ks-opt"
    if not ks_opt.exists():
        raise RuntimeError("build/bin/ks-opt not found. Run: cmake --build build --parallel")
    check_tool("mlir-translate", "Install LLVM/MLIR 21 or set PATH to /usr/lib/llvm-21/bin.")
    check_tool("llc", "Install LLVM 21 or set PATH to /usr/lib/llvm-21/bin.")
    check_tool("riscv64-linux-gnu-gcc", "Run: ./scripts/setup-rvv-sim.sh")


def check_run_tools(qemu_binary: str) -> None:
    check_tool(qemu_binary, "Run: ./scripts/setup-rvv-sim.sh")


def case_paths(case: dict, output_dir: Path) -> tuple[Path, Path]:
    obj = output_dir / "obj" / f"{case['name']}.o"
    binary = output_dir / "bin" / case["name"]
    return obj, binary


def build_case(case: dict, output_dir: Path) -> Path:
    obj, binary = case_paths(case, output_dir)
    obj.parent.mkdir(parents=True, exist_ok=True)
    binary.parent.mkdir(parents=True, exist_ok=True)

    compile_cmd = [
        str(REPO_ROOT / "scripts" / "compile-rvv.sh"),
        str(repo_path(case["input_mlir"])),
        str(obj),
    ]
    run_command(compile_cmd)

    harness = repo_path(case["harness"])
    link_cmd = [
        "riscv64-linux-gnu-gcc",
        "-static",
        "-O2",
        "-march=rv64gcv",
        "-mabi=lp64d",
        f'-DKS_RISCV_TEST_NAME="{case["name"]}"',
        f"-DKS_RISCV_MAX_ERROR_FUNCTION={case['entry_symbol']}",
        f"-DKS_RISCV_TOLERANCE={case['tolerance']}f",
        str(harness),
        str(obj),
        "-o",
        str(binary),
        "-lm",
    ]
    run_command(link_cmd)
    return binary


def run_command(cmd: list[str]) -> None:
    print("+ " + " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def parse_result(name: str, vlen: int, return_code: int, stdout: str, stderr: str) -> CaseResult:
    max_abs_error = parse_float_marker(stdout, "MAX_ABS_ERROR")
    time_ns = parse_int_marker(stdout, "TIME_NS")
    return CaseResult(name, vlen, return_code, stdout, stderr, max_abs_error, time_ns)


def parse_float_marker(output: str, marker: str) -> float | None:
    match = re.search(rf"^{re.escape(marker)}:\s*([-+0-9.eE]+)\s*$", output, re.MULTILINE)
    return float(match.group(1)) if match else None


def parse_int_marker(output: str, marker: str) -> int | None:
    match = re.search(rf"^{re.escape(marker)}:\s*([0-9]+)\s*$", output, re.MULTILINE)
    return int(match.group(1)) if match else None


def run_case(case: dict, output_dir: Path, qemu_binary: str, vlens: list[int]) -> list[CaseResult]:
    validate_vlens(case, vlens)
    _, binary = case_paths(case, output_dir)
    if not binary.exists():
        raise FileNotFoundError(f"{binary} not found. Re-run with --build or --all.")
    ensure_executable(binary)

    results = []
    for vlen in vlens:
        cpu = f"rv64,v=true,vlen={vlen},vext_spec=v1.0"
        cmd = [qemu_binary, str(binary)]
        env = {**os.environ, "QEMU_CPU": cpu}
        print("+ QEMU_CPU=" + cpu + " " + " ".join(cmd), flush=True)
        completed = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60, check=False, env=env
        )
        if should_retry_with_static_qemu(qemu_binary, completed):
            static_qemu = "qemu-riscv64-static"
            cmd[0] = static_qemu
            print(f"Retrying with {static_qemu} after exit {completed.returncode}", flush=True)
            print("+ QEMU_CPU=" + cpu + " " + " ".join(cmd), flush=True)
            completed = subprocess.run(
                cmd, capture_output=True, text=True, timeout=60, check=False, env=env
            )
        result = parse_result(
            case["name"], vlen, completed.returncode, completed.stdout, completed.stderr
        )
        print_result(result)
        results.append(result)
    return results


def ensure_executable(binary: Path) -> None:
    mode = binary.stat().st_mode
    if mode & 0o111:
        return
    binary.chmod(mode | 0o111)
    print(f"Made RISC-V ELF executable: {binary}", flush=True)


def should_retry_with_static_qemu(qemu_binary: str, completed: subprocess.CompletedProcess) -> bool:
    if Path(qemu_binary).name != "qemu-riscv64":
        return False
    if completed.returncode == 0 or completed.stdout or completed.stderr:
        return False
    return shutil.which("qemu-riscv64-static") is not None


def print_result(result: CaseResult) -> None:
    status = "PASS" if result.passed else "FAIL"
    err = "nan" if result.max_abs_error is None else f"{result.max_abs_error:.9g}"
    time_ns = "0" if result.time_ns is None else str(result.time_ns)
    print(
        f"RESULT case={result.name} sim=qemu-user vlen={result.vlen} "
        f"status={status} return_code={result.return_code} "
        f"max_abs_error={err} time_ns={time_ns}",
        flush=True,
    )
    if result.stderr:
        print(result.stderr, file=sys.stderr, end="")


def validate_result_markers(case: dict, results: list[CaseResult]) -> bool:
    ok = True
    for result in results:
        if not result.passed:
            ok = False
        if result.max_abs_error is None:
            print(f"ERROR: {result.name} VLEN={result.vlen} did not print MAX_ABS_ERROR")
            ok = False
        elif result.max_abs_error > float(case["tolerance"]):
            print(
                f"ERROR: {result.name} VLEN={result.vlen} max_abs_error "
                f"{result.max_abs_error} exceeds tolerance {case['tolerance']}"
            )
            ok = False
        if case.get("benchmark", False) and result.time_ns is None:
            print(f"ERROR: {result.name} VLEN={result.vlen} did not print TIME_NS")
            ok = False
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Build and run RISC-V functional tests.")
    parser.add_argument("--descriptor", type=Path, default=DEFAULT_DESCRIPTOR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--case", nargs="+", help="Case name(s) to run.")
    parser.add_argument("--vlens", nargs="+", type=int, help="Override descriptor VLEN list.")
    parser.add_argument("--qemu", default="qemu-riscv64", help="QEMU user-mode binary.")
    parser.add_argument("--list", action="store_true", help="List descriptor cases and exit.")
    parser.add_argument("--check-only", action="store_true", help="Validate descriptors and exit.")
    parser.add_argument("--build", action="store_true", help="Build RISC-V ELF binaries.")
    parser.add_argument("--run", action="store_true", help="Run RISC-V ELF binaries under QEMU.")
    parser.add_argument("--all", action="store_true", help="Build and run all selected cases.")
    args = parser.parse_args()

    try:
        cases = select_cases(load_cases(args.descriptor), args.case)
        if args.list:
            for case in cases:
                print(f"{case['name']} profile={case['profile']} vlens={case['vlens']}")
            return 0
        if args.check_only:
            return 0

        do_build = args.all or args.build
        do_run = args.all or args.run
        if not do_build and not do_run:
            parser.error("choose --build, --run, --all, --list, or --check-only")

        if do_build:
            check_build_tools()
            for case in cases:
                build_case(case, args.output_dir)

        all_ok = True
        if do_run:
            if args.vlens is not None:
                for case in cases:
                    validate_vlens(case, args.vlens)
            check_run_tools(args.qemu)
            for case in cases:
                vlens = args.vlens if args.vlens is not None else case["vlens"]
                results = run_case(case, args.output_dir, args.qemu, vlens)
                all_ok = validate_result_markers(case, results) and all_ok
        return 0 if all_ok else 1
    except (FileNotFoundError, RuntimeError, ValueError, subprocess.CalledProcessError) as err:
        print(f"ERROR: {err}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
