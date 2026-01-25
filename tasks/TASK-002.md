# TASK-002: Kernel Dialect Core

## Status
[ ] Not Started

## Current Phase
Backlog

## Priority
P0 (Critical) - Blocks all kernel operations

## Description

Build and verify the core kernel dialect infrastructure:
- Verify dialect registration works
- Verify basic operations parse and print
- Ensure verifiers function correctly
- Set up foundation for additional operations

## Dependencies

- [x] TASK-001: Infrastructure (Complete)

## Acceptance Criteria

- [ ] ks-opt builds successfully
- [ ] ks.relu parses and prints correctly
- [ ] ks.softmax parses and prints correctly
- [ ] ks.matmul parses and prints correctly
- [ ] Verifiers catch invalid inputs
- [ ] Lit tests pass via CTest
- [ ] Round-trip (parse → print → parse) works

## Workflow Tracking

### Phase Checklist

- [ ] **Specification**: Check specs exist for operations
- [ ] **Design**: Create design doc DES-002
- [ ] **Review**: Get design approved
- [ ] **Implement**: TDD - tests first
- [ ] **Verify**: All tests pass
- [ ] **Complete**: Close task

## Design Document

Create: `docs/design/DES-002-kernel-dialect-core.md`

## Test Files

- `tests/lit/Dialect/Kernel/basic.mlir`
- `tests/lit/Dialect/Kernel/matmul.mlir`
- `tests/lit/Dialect/Kernel/invalid.mlir` (verifier tests)

## Verification

```bash
cmake --build build --target ks-opt
ctest --test-dir build -R "Dialect"
```

## Progress Log

### 2025-01-25
- Task created
- Waiting to start
- **Next**: Create design document
- **Assigned to**: Architect Agent
