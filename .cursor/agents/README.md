# KernelSmith Agent Templates

These files are prompt templates for focused Cursor agents or human handoffs.
They are intentionally repository-local so agents can use consistent roles
without rediscovering project infrastructure each time.

Recommended templates:

- `mlir-pass-agent.md` — implement or review `--ks-*` lowering passes.
- `ci-infra-agent.md` — diagnose CI, Docker, scripts, and developer setup.
- `rvv-validation-agent.md` — validate RVV lowering/QEMU behavior.
- `task-planning-agent.md` — groom roadmap milestones and task files.

When launching an agent, paste the relevant template and add the concrete task,
branch, and files under investigation.
