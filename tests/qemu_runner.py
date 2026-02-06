#!/usr/bin/env python3
"""
QEMU Runner for Multi-VLEN Testing
Runs RISC-V kernels on QEMU with different vector lengths
"""

import json
import os
import subprocess
from pathlib import Path
from typing import List, Tuple


class QEMURunner:
    """Execute RISC-V binaries on QEMU with different VLEN configurations"""

    # Standard VLEN values to test
    STANDARD_VLENS = [128, 256, 512]

    def __init__(self, qemu_binary: str = "qemu-riscv64"):
        self.qemu_binary = qemu_binary
        self._check_qemu_available()

    def _check_qemu_available(self):
        """Check if QEMU is available"""
        try:
            subprocess.run([self.qemu_binary, "-version"], capture_output=True, check=True)
        except (FileNotFoundError, subprocess.CalledProcessError):
            raise RuntimeError(
                f"{self.qemu_binary} not found. Install with: apt install qemu-user"
            )

    def run_with_vlen(self, binary_path: str, vlen: int, args: List[str] = None) -> Tuple[int, str, str]:
        """
        Run a RISC-V binary on QEMU with specific VLEN

        Args:
            binary_path: Path to RISC-V executable
            vlen: Vector length in bits (128, 256, 512, etc.)
            args: Command line arguments to pass to binary

        Returns:
            Tuple of (return_code, stdout, stderr)
        """
        if not Path(binary_path).exists():
            raise FileNotFoundError(f"Binary not found: {binary_path}")

        cmd = [
            self.qemu_binary,
            "-cpu", f"rv64,v=true,vlen={vlen}",
        ]

        if args:
            cmd.extend(args)
        else:
            cmd.append(binary_path)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", f"Timeout running {binary_path}"

    def run_multi_vlen(
        self, binary_path: str, vlens: List[int] = None, args: List[str] = None
    ) -> dict:
        """
        Run binary on multiple VLEN configurations

        Args:
            binary_path: Path to RISC-V executable
            vlens: List of vector lengths to test (default: STANDARD_VLENS)
            args: Command line arguments

        Returns:
            Dictionary with results for each VLEN
        """
        if vlens is None:
            vlens = self.STANDARD_VLENS

        results = {}

        for vlen in vlens:
            print(f"Running {Path(binary_path).name} with VLEN={vlen}...")
            ret_code, stdout, stderr = self.run_with_vlen(binary_path, vlen, args)

            results[vlen] = {
                "return_code": ret_code,
                "stdout": stdout,
                "stderr": stderr,
                "success": ret_code == 0,
            }

            if ret_code == 0:
                print(f"  ✓ VLEN={vlen}: PASSED")
            else:
                print(f"  ✗ VLEN={vlen}: FAILED (exit code {ret_code})")
                if stderr:
                    print(f"    Error: {stderr[:200]}")

        return results

    def validate_consistent_results(self, results: dict) -> bool:
        """
        Validate that results are consistent across different VLEN values

        Args:
            results: Dictionary from run_multi_vlen()

        Returns:
            True if all VLENs produce consistent results
        """
        all_passed = all(r["success"] for r in results.values())
        if not all_passed:
            return False

        # For numerical consistency, would compare actual output values here
        # This is a placeholder for basic sanity check
        return True

    def benchmark_kernel(self, binary_path: str, vlen: int, runs: int = 5) -> dict:
        """
        Benchmark a kernel on specific VLEN

        Args:
            binary_path: Path to RISC-V binary with timing
            vlen: Vector length in bits
            runs: Number of runs to average

        Returns:
            Dictionary with timing statistics
        """
        times = []

        for i in range(runs):
            ret_code, stdout, stderr = self.run_with_vlen(binary_path, vlen)
            if ret_code != 0:
                return {"error": f"Benchmark failed: {stderr}"}

            # Parse timing from output (assuming binary prints timing info)
            try:
                # Placeholder: would parse actual timing from stdout
                pass
            except Exception as e:
                return {"error": f"Failed to parse timing: {e}"}

        return {
            "vlen": vlen,
            "runs": runs,
            "times": times,
            "avg_time": sum(times) / len(times) if times else 0,
        }


def run_qemu_tests():
    """Run example QEMU tests"""
    runner = QEMURunner()

    print("QEMU Configuration:")
    print(f"  Binary: {runner.qemu_binary}")
    print(f"  Standard VLENs: {runner.STANDARD_VLENS}")
    print()

    # Example: test a simple binary
    # binary_path = "build/bin/some_test"
    # if Path(binary_path).exists():
    #     results = runner.run_multi_vlen(binary_path)
    #     consistent = runner.validate_consistent_results(results)
    #     print(f"Results consistent across VLENs: {consistent}")


if __name__ == "__main__":
    run_qemu_tests()
