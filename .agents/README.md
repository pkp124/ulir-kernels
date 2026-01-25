# KernelSmith Agent System

This directory defines the agent skills, workflows, and processes for AI-assisted development of KernelSmith.

## Philosophy

KernelSmith uses a rigorous software engineering process with specialized agent roles:

1. **Architect Agent**: High-level design and system decisions
2. **Reviewer Agent**: Critical review of designs and implementations
3. **Implementation Agent**: TDD-based coding
4. **Verification Agent**: Testing and quality assurance

## Directory Structure

```
.agents/
├── README.md           # This file
├── skills/             # Agent skill definitions
│   ├── architect.md    # Design and architecture skills
│   ├── reviewer.md     # Critical review skills
│   ├── implementer.md  # TDD implementation skills
│   └── verifier.md     # Testing and verification skills
└── workflows/          # Development workflows
    ├── feature.md      # New feature workflow
    ├── design-review.md # Design review process
    └── tdd.md          # Test-driven development
```

## Workflow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  1. SPECIFICATION                                                │
│     - Check/create spec in specs/                               │
│     - Define requirements and acceptance criteria               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  2. DESIGN (Architect Agent)                                     │
│     - Create design doc in docs/design/                         │
│     - Document alternatives and trade-offs                      │
│     - Define test strategy                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  3. DESIGN REVIEW (Reviewer Agent)                               │
│     - Critical analysis of design                               │
│     - Check requirements coverage                               │
│     - Identify risks and gaps                                   │
│     - Approve or request changes                                │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  4. TDD IMPLEMENTATION (Implementation Agent)                    │
│     - Write failing tests first                                 │
│     - Implement to pass tests                                   │
│     - Refactor with green tests                                 │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  5. VERIFICATION (Verification Agent)                            │
│     - Run full test suite                                       │
│     - Check edge cases                                          │
│     - Verify against specification                              │
└─────────────────────────────────────────────────────────────────┘
```

## Using Agents

When working on a feature, explicitly switch between agent roles:

1. Start with **Architect** to create design
2. Switch to **Reviewer** for critical analysis
3. Use **Implementer** for TDD coding
4. End with **Verifier** for final checks

Each agent has specific skills and focus areas defined in `skills/`.
