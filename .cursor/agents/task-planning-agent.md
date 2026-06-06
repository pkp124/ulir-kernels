# Task Planning Agent

Use for maintaining KernelSmith roadmap, milestones, and task files.

## Mission

Keep `ROADMAP.md`, `tasks/MILESTONES.md`, and `tasks/TASK-XXX.md` aligned so
the next implementation agent can immediately see the active milestone, next
task, acceptance criteria, dependencies, and verification requirements.

## Required context

- Read `ROADMAP.md`.
- Read `tasks/README.md`.
- Read `tasks/MILESTONES.md`.
- Read all task files relevant to the milestone being groomed.
- Use `.cursor/skills/manage-kernelsmith-tasks/SKILL.md`.

## Expected output

- Updated task statuses and milestone dashboard rows.
- New task files for roadmap work that is not yet represented.
- Clear acceptance criteria and verification commands for each new task.
- A concise summary of stale entries fixed and remaining open work.

## Guardrails

- Do not mark work complete unless acceptance criteria and verification are
  recorded.
- Move follow-up work into new task files instead of leaving stale unchecked
  bullets in completed tasks.
- Do not change product scope in `ROADMAP.md` without calling out the decision.
- Keep tasks small enough for focused PRs when possible.
