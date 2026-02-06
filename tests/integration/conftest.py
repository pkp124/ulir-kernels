"""
Shared fixtures for integration tests.
"""

import shutil

import pytest


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests if RISC-V toolchain is unavailable."""
    riscv_gcc = shutil.which("riscv64-linux-gnu-gcc")
    qemu = shutil.which("qemu-riscv64")

    if riscv_gcc and qemu:
        return

    skip_marker = pytest.mark.skip(reason="RISC-V toolchain not available")
    for item in items:
        if "riscv" in item.nodeid.lower():
            item.add_marker(skip_marker)
