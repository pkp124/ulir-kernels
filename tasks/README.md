# Task Tracking

This directory contains task files for tracking development progress.

## Task Format

Each task is a markdown file with structured sections:

```markdown
# TASK-XXX: Task Title

## Status
[ ] Not Started / [~] In Progress / [x] Complete

## Priority
P0 (Critical) / P1 (High) / P2 (Medium) / P3 (Low)

## Description
What needs to be done and why.

## Acceptance Criteria
- [ ] Criterion 1
- [ ] Criterion 2

## Implementation Notes
Technical details, design decisions.

## Dependencies
- TASK-YYY (blocking)

## Verification
How to verify this task is complete.

## Log
Progress updates with dates.
```

## Task Categories

- `TASK-0XX`: Infrastructure and setup
- `TASK-1XX`: Kernel dialect and operations
- `TASK-2XX`: Transformation passes
- `TASK-3XX`: RISC-V RVV target
- `TASK-4XX`: Testing and verification
- `TASK-5XX`: Documentation and examples

## Current Tasks

| ID | Title | Status | Priority |
|----|-------|--------|----------|
| TASK-001 | Set up MLIR dialect infrastructure | Complete | P0 |
| TASK-002 | Implement Kernel dialect | Complete | P0 |
| TASK-003 | Implement matmul operation | Complete | P1 |
| TASK-004 | Implement tiling pass | In Progress | P1 |
| TASK-005 | Implement RVV lowering | Not Started | P1 |

## Working with Tasks

### Create a new task

```bash
make new-task ID=006 TITLE="My new task"
```

### Update task status

Edit the task file directly, updating:
1. Status checkbox
2. Log section with progress

### Complete a task

1. Ensure all acceptance criteria are met
2. Run `make verify`
3. Update status to `[x] Complete`
4. Add completion note to log
