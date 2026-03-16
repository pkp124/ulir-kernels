#!/usr/bin/env python3
"""
QEMU Runner for Multi-VLEN Testing (M4: RISC-V RVV Target)

Runs RISC-V kernels on QEMU with different vector lengths to validate
that the KernelSmith RVV pipeline produces correct results across VLEN
configurations (128, 256, 512 bits).

Usage:
    python tests/qemu_runner.py --binary build-rvv/bin/test_matmul_rvv
    python tests/qemu_runner.py --binary build-rvv/bin/test_matmul_rvv \
        --vlens 128 256 512 --benchmark
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


class QEMURunner:
    """Execute RISC-V binaries on QEMU with different VLEN configurations."""

    # Standard VLEN values to test (bits)
    STANDARD_VLENS = [128, 256, 512]

    def __init__(self, qemu_binary: str = "qemu-riscv64"):
        self.qemu_binary = qemu_binary
        self._check_qemu_available()

    def _check_qemu_available(self):
        """Check if QEMU is available."""
        try:
            subprocess.run([self.qemu_binary, "--version"],
                           capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError) as err:
            raise RuntimeError(
                f"{self.qemu_binary} not found. Install with: apt install qemu-user"
            ) from err

    def run_with_vlen(
        self,
        binary_path: str,
        vlen: int,
        args: list | None = None,
        timeout: int = 60,
    ) -> tuple:
        """
        Run a RISC-V binary on QEMU with specific VLEN.

        Args:
            binary_path: Path to RISC-V executable.
            vlen: Vector length in bits (128, 256, 512, etc.).
            args: Extra command line arguments to pass to the binary.
            timeout: Timeout in seconds.

        Returns:
            Tuple of (return_code, stdout, stderr).
        """
        if not Path(binary_path).exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        # QEMU user-mode with RVV enabled.
        # -cpu rv64,v=true,vlen=<N>  enables the V extension at the given VLEN.
        cmd = [
            self.qemu_binary,
            "-cpu", f"rv64,v=true,vlen={vlen},vext_spec=v1.0",
            binary_path,
        ]
        if args:
            cmd.extend(args)

        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timeout ({timeout}s) running {binary_path}"

    def run_multi_vlen(
        self,
        binary_path: str,
        vlens: list | None = None,
        args: list | None = None,
    ) -> dict:
        """
        Run binary on multiple VLEN configurations.

        Args:
            binary_path: Path to RISC-V executable.
            vlens: List of vector lengths to test (default: STANDARD_VLENS).
            args: Extra command line arguments.

        Returns:
            Dictionary mapping VLEN -> {return_code, stdout, stderr, success}.
        """
        if vlens is None:
            vlens = self.STANDARD_VLENS

        results = {}
        for vlen in vlens:
            print(f"  Running {Path(binary_path).name} with VLEN={vlen}...")
            ret_code, stdout, stderr = self.run_with_vlen(binary_path, vlen, args)
            results[vlen] = {
                "return_code": ret_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": ret_code == 0,
            }
            status = "PASS" if ret_code == 0 else f"FAIL (exit {ret_code})"
            print(f"    VLEN={vlen}: {status}")
            if ret_code != 0 and stderr:
                print(f"      stderr: {stderr[:300]}")

        return results

    def validate_consistent_results(self, results: dict) -> bool:
        """
        Check that all VLEN runs passed and produced identical stdout.

        Returns True only if every VLEN produced exit code 0 and the same
        standard output (bit-for-bit identical numerical results).
        """
        if not all(r["success"] for r in results.values()):
            return False

        outputs = [r["stdout"] for r in results.values()]
        if len(set(outputs)) > 1:
            print("WARNING: stdout differs across VLENs — numerical divergence?")
            for vlen, r in results.items():
                print(f"  VLEN={vlen}: {r['stdout'][:100]!r}")
            return False

        return True

    def benchmark_kernel(
        self,
        binary_path: str,
        vlen: int,
        runs: int = 5,
        warmup: int = 2,
    ) -> dict:
        """
        Benchmark a kernel on a specific VLEN.

        The binary must print a line "TIME_NS: <n>" to stdout.

        Args:
            binary_path: Path to RISC-V binary with built-in timing.
            vlen: Vector length in bits.
            runs: Number of timed runs.
            warmup: Number of warmup runs (not counted).

        Returns:
            Dictionary with timing statistics.
        """
        all_times = []

        for i in range(warmup + runs):
            t0 = time.perf_counter()
            ret_code, stdout, stderr = self.run_with_vlen(binary_path, vlen)
            t1 = time.perf_counter()

            if ret_code != 0:
                return {"error": f"Benchmark run failed: {stderr[:200]}"}

            wall_ms = (t1 - t0) * 1000
            parsed = None
            for line in stdout.splitlines():
                if line.startswith("TIME_NS:"):
                    try:
                        parsed = int(line.split(":")[1].strip()) / 1e6  # ns -> ms
                    except ValueError:
                        pass

            if i >= warmup:
                all_times.append(parsed if parsed is not None else wall_ms)

        return {
            "vlen": vlen,
            "runs": runs,
            "times_ms": all_times,
            "min_ms": min(all_times),
            "avg_ms": sum(all_times) / len(all_times),
            "max_ms": max(all_times),
        }


def run_matmul_correctness_test(runner: QEMURunner, binary: str,
                                 vlens: list) -> bool:
    """
    Run the RVV matmul binary across all VLENs and validate consistency.

    The binary is expected to print "PASS" to stdout on success.

    Returns True if all VLENs pass.
    """
    print(f"\n=== Matmul correctness test: {Path(binary).name} ===")
    results = runner.run_multi_vlen(binary, vlens)

    all_pass = True
    for vlen, r in results.items():
        if not r["success"]:
            print(f"  FAIL VLEN={vlen}: non-zero exit")
            all_pass = False
        elif "PASS" not in r["stdout"]:
            print(f"  FAIL VLEN={vlen}: 'PASS' not in output: {r['stdout'][:100]!r}")
            all_pass = False

    if all_pass:
        print("  All VLENs PASSED.")

    consistent = runner.validate_consistent_results(results)
    if not consistent:
        print("  WARNING: results inconsistent across VLENs")
        all_pass = False

    return all_pass


def run_benchmark(runner: QEMURunner, binary: str, vlens: list) -> None:
    """Print benchmark results for each VLEN."""
    print(f"\n=== Benchmark: {Path(binary).name} ===")
    for vlen in vlens:
        stats = runner.benchmark_kernel(binary, vlen)
        if "error" in stats:
            print(f"  VLEN={vlen}: ERROR — {stats['error']}")
        else:
            print(
                f"  VLEN={vlen}: min={stats['min_ms']:.2f}ms  "
                f"avg={stats['avg_ms']:.2f}ms  max={stats['max_ms']:.2f}ms"
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run RISC-V RVV kernels on QEMU with multiple VLEN configs."
    )
    parser.add_argument(
        "--binary", required=True,
        help="Path to the RISC-V ELF binary to test."
    )
    parser.add_argument(
        "--vlens", nargs="+", type=int,
        default=QEMURunner.STANDARD_VLENS,
        help="VLEN values to test (default: 128 256 512)."
    )
    parser.add_argument(
        "--benchmark", action="store_true",
        help="Run benchmark in addition to correctness test."
    )
    parser.add_argument(
        "--qemu", default="qemu-riscv64",
        help="Path or name of the QEMU user-mode binary."
    )
    args = parser.parse_args()

    try:
        runner = QEMURunner(args.qemu)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1

    passed = run_matmul_correctness_test(runner, args.binary, args.vlens)

    if args.benchmark:
        run_benchmark(runner, args.binary, args.vlens)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
