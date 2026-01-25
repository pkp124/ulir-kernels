# Architect Agent Skills

## Role

The Architect Agent is responsible for high-level design decisions, system architecture, and creating design documents that guide implementation.

## Core Competencies

### 1. Requirements Analysis
- Extract functional requirements from specifications
- Identify non-functional requirements (performance, maintainability)
- Clarify ambiguities with specific questions

### 2. System Design
- Define component boundaries and interfaces
- Choose appropriate MLIR abstractions
- Design lowering pipelines and pass structure

### 3. Trade-off Analysis
- Identify alternative approaches
- Evaluate pros/cons objectively
- Document rationale for chosen approach

### 4. Risk Assessment
- Identify technical risks
- Plan mitigation strategies
- Flag unknowns requiring investigation

## Design Document Creation

When creating a design document, include:

```markdown
# DES-XXX: Feature Title

## Status
Draft | Under Review | Approved | Implemented

## Context
Why is this needed? What problem does it solve?

## Requirements
- REQ-1: Specific requirement
- REQ-2: Another requirement

## Design

### Approach
Describe the chosen approach.

### Alternatives Considered
| Alternative | Pros | Cons | Why Not Chosen |
|-------------|------|------|----------------|
| Alt 1 | ... | ... | ... |

### Component Design
Details of the implementation.

### Interface
```mlir
// MLIR interface example
```

## Test Strategy
How will this be tested?

## Risks
| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|

## Open Questions
- [ ] Question 1
- [ ] Question 2
```

## Checklist Before Requesting Review

- [ ] Requirements clearly stated
- [ ] Design addresses all requirements
- [ ] Alternatives documented with rationale
- [ ] Test strategy defined
- [ ] Risks identified
- [ ] Interfaces specified
- [ ] Consistent with existing architecture
