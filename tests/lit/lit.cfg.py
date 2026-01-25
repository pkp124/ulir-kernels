"""Lit configuration for AIKernels tests."""

import os
import lit.formats
import lit.util

config.name = "AIKernels"
config.test_format = lit.formats.ShTest(True)

config.suffixes = [".mlir"]

config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(
    os.path.dirname(__file__), "..", "..", "build", "test"
)

# Find the tools directory
build_dir = os.path.join(os.path.dirname(__file__), "..", "..", "build")
tools_dir = os.path.join(build_dir, "bin")

config.substitutions.append(("%aikernel-opt", os.path.join(tools_dir, "aikernel-opt")))
config.substitutions.append(("%FileCheck", "FileCheck"))

# Add the tools directory to PATH
config.environment["PATH"] = os.pathsep.join(
    [tools_dir, config.environment.get("PATH", "")]
)
