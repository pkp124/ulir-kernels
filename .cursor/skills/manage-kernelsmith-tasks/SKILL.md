# Manage KernelSmith Tasks

Use this skill when asked to plan work, pick the next task, update task status,
or create task files for roadmap items.

## Task system

- `ROADMAP.md` is the product milestone source of truth.
- `tasks/MILESTONES.md` is the operational dashboard.
- `tasks/TASK-XXX.md` files are executable work packets.
- `.cursor/agents/*.md` files are role templates for focused agents.

## Workflow

1. Read `tasks/MILESTONES.md`.
2. Read the selected `tasks/TASK-XXX.md`.
3. Read related specs and design docs before implementation.
4. If a roadmap item is too broad, create or propose smaller child tasks before
   implementation.
5. Before editing code, update task status to `[~] In Progress` when the task
   was previously not started.
6. When completing work:
   - update acceptance checkboxes,
   - add verification commands/results,
   - add a dated log entry,
   - update `tasks/MILESTONES.md`,
   - leave follow-up work as new task files instead of stale unchecked items.
7. Do not mark a task `[x] Complete` unless required verification passed or the
   task clearly records why validation is unavailable.

## Status rules

- `[ ] Not Started`: scoped but no work begun.
- `[~] In Progress`: implementation or validation has begun.
- `[!] Blocked`: blocked by missing toolchain, hardware, external dependency, or
  design decision.
- `[?] Needs Review`: implementation is ready but awaiting review or external
  validation.
- `[x] Complete`: acceptance criteria are met and verification is recorded.

## Agent handoff checklist

- State the active task ID and milestone.
- List files changed and verification run.
- Record blockers in the task log.
- Keep `tasks/README.md`, `tasks/MILESTONES.md`, and task files consistent.
