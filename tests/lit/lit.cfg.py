"""Lit configuration for KernelSmith tests."""

import os

import lit.formats
import lit.util

config.name = "KernelSmith"
config.test_format = lit.formats.ShTest(True)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)

# Get paths from site config or environment
tools_dir = getattr(config, "ks_tools_dir", None)
if not tools_dir:
    build_dir = os.environ.get(
        "KS_BUILD_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "build")
    )
    tools_dir = os.path.join(build_dir, "bin")

config.test_exec_root = os.path.join(tools_dir, "..", "test")

# Tool substitutions
config.substitutions.append(("%ks-opt", os.path.join(tools_dir, "ks-opt")))

# Find FileCheck
filecheck = lit.util.which("FileCheck")
if not filecheck:
    llvm_tools = getattr(config, "llvm_tools_dir", "/usr/lib/llvm-20/bin")
    filecheck = os.path.join(llvm_tools, "FileCheck")
config.substitutions.append(("%FileCheck", filecheck))

# Add tools to PATH
path = config.environment.get("PATH", "")
config.environment["PATH"] = os.pathsep.join([tools_dir, path])
