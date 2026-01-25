# TASK-001: Set up MLIR Dialect Infrastructure

## Status
[ ] Not Started

## Priority
P0 (Critical)

## Description

Set up the foundational infrastructure for MLIR dialects including:
- CMake build configuration for TableGen
- Base dialect and operation classes
- Pass registration infrastructure
- Tool driver (aikernel-opt)

This is a prerequisite for all kernel implementation work.

## Acceptance Criteria

- [ ] CMakeLists.txt properly configured for MLIR/LLVM
- [ ] TableGen targets generate header files
- [ ] Base Kernel dialect compiles
- [ ] `aikernel-opt` tool builds and runs
- [ ] Can parse and print a simple MLIR file
- [ ] CI pipeline passes

## Implementation Notes

### Directory Structure

```
src/
├── dialects/
│   └── kernel/
│       ├── CMakeLists.txt
│       ├── KernelDialect.td      # TableGen
│       ├── KernelDialect.h       # Header
│       └── KernelDialect.cpp     # Implementation
├── tools/
│   └── aikernel-opt/
│       ├── CMakeLists.txt
│       └── aikernel-opt.cpp
└── CMakeLists.txt
```

### Key Components

1. **CMake Configuration**
   - Find MLIR/LLVM packages
   - Set up TableGen targets
   - Configure include paths

2. **Dialect Registration**
   - Define dialect in TableGen
   - Implement registration function
   - Register with MLIR context

3. **Tool Driver**
   - Parse command line
   - Load MLIR module
   - Run passes
   - Output result

## Dependencies

None (this is the foundation)

## Verification

```bash
# Build succeeds
make build

# Tool runs
./build/bin/aikernel-opt --help

# Can parse empty module
echo 'module {}' | ./build/bin/aikernel-opt
```

## Log

### [Date TBD]
- Task created
