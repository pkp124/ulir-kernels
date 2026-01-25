# KernelSmith Agent System

This directory defines the agent skills, workflows, and processes for AI-assisted development of KernelSmith.

## Philosophy

KernelSmith uses a rigorous software engineering process with specialized agent roles. The **Manager Agent** coordinates all work to ensure proper sequencing.

## Agent Roles

| Agent | Role | Primary Focus |
|-------|------|---------------|
| **Manager** | Coordination & tracking | Workflow enforcement, task tracking |
| **Architect** | System design | Design docs, architecture decisions |
| **Reviewer** | Critical analysis | Design review, find issues |
| **Implementer** | TDD coding | Write tests first, then code |
| **Verifier** | Quality assurance | Test execution, spec compliance |

## Directory Structure

```
.agents/
├── README.md           # This file
├── skills/             # Agent skill definitions
│   ├── manager.md      # Workflow coordination
│   ├── architect.md    # Design and architecture
│   ├── reviewer.md     # Critical review
│   ├── implementer.md  # TDD implementation
│   └── verifier.md     # Testing and verification
└── workflows/          # Development workflows
    ├── feature.md      # New feature workflow
    ├── design-review.md # Design review process
    └── tdd.md          # Test-driven development
```

## Workflow Overview

The Manager Agent enforces this workflow:

```
┌─────────────────────────────────────────────────────────────────┐
│  MANAGER: Coordinates all phases, prevents out-of-order work   │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐    ┌───────────────┐    ┌───────────────┐
│ 1. SPEC       │    │ 2. DESIGN     │    │ 3. REVIEW     │
│ (Manager)     │ → │ (Architect)   │ → │ (Reviewer)    │
│               │    │               │    │               │
│ Check/create  │    │ Create design │    │ Critical      │
│ specification │    │ document      │    │ analysis      │
└───────────────┘    └───────────────┘    └───────────────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ Approved?     │
                                          └───────┬───────┘
                                             No ↙   ↘ Yes
                                    ┌──────────┐    │
                                    │ Revise   │    │
                                    └──────────┘    ▼
                                          ┌───────────────┐
                                          │ 4. IMPLEMENT  │
                                          │ (Implementer) │
                                          │               │
                                          │ TDD: tests    │
                                          │ first         │
                                          └───────────────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ 5. VERIFY     │
                                          │ (Verifier)    │
                                          │               │
                                          │ Full test     │
                                          │ suite         │
                                          └───────────────┘
                                                  │
                                                  ▼
                                          ┌───────────────┐
                                          │ 6. COMPLETE   │
                                          │ (Manager)     │
                                          │               │
                                          │ Update status │
                                          │ Close task    │
                                          └───────────────┘
```

## Using the Agent System

### Starting a Session

1. **Manager first**: Check current task status
2. **Identify phase**: What phase is current work in?
3. **Select agent**: Use appropriate agent for the phase
4. **Follow workflow**: Don't skip steps

### Agent Selection

| Current Phase | Agent to Use |
|---------------|--------------|
| Task planning | Manager |
| Specification | Manager + Architect |
| Design | Architect |
| Review | Reviewer |
| Implementation | Implementer |
| Testing | Verifier |
| Completion | Manager |

### Switching Agents

When switching between agents:
1. Complete current agent's checklist
2. Document handoff in task file
3. Switch to next agent
4. Continue from documented state

## Quick Reference

### Manager Commands

```bash
# Check active tasks
ls tasks/*.md

# Create new task
make new-task ID=XXX TITLE="Task name"

# View task status
cat tasks/TASK-XXX.md
```

### Workflow Gates

| Gate | Requirement |
|------|-------------|
| Start Design | Specification exists |
| Start Review | Design doc complete |
| Start Implementation | Design approved |
| Mark Complete | All tests pass |

### Red Flags

- ❌ Coding without design doc
- ❌ Writing code before tests
- ❌ Skipping review for "small" changes
- ❌ Not running full test suite
- ❌ No specification for new features
