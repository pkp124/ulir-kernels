# DES-XXX: [Feature Title]

## Metadata

| Field | Value |
|-------|-------|
| **Status** | Draft / Under Review / Approved / Implemented |
| **Author** | [Name] |
| **Created** | YYYY-MM-DD |
| **Approved** | YYYY-MM-DD |
| **Approver** | [Name] |

## Context

### Problem Statement
What problem does this solve? Why is it needed?

### Background
Relevant context, prior work, or related systems.

## Requirements

From specification: `specs/[path-to-spec].md`

| ID | Requirement | Priority |
|----|-------------|----------|
| REQ-1 | [Requirement text] | Must Have |
| REQ-2 | [Requirement text] | Should Have |
| REQ-3 | [Requirement text] | Nice to Have |

## Design

### Overview
High-level description of the approach.

### Component Design

#### [Component 1]
Details of this component.

#### [Component 2]
Details of this component.

### Interface

```mlir
// Example MLIR interface
func.func @example(%input: tensor<...>) -> tensor<...> {
  %output = ks.operation %input : ...
  return %output
}
```

### Data Flow

```
[Input] → [Component 1] → [Component 2] → [Output]
```

## Alternatives Considered

| Alternative | Pros | Cons | Decision |
|-------------|------|------|----------|
| [Alt 1] | [Pros] | [Cons] | [Why not chosen] |
| [Alt 2] | [Pros] | [Cons] | [Why not chosen] |
| [Chosen approach] | [Pros] | [Cons] | **Selected** |

### Alternative 1: [Name]
Detailed description and analysis.

### Alternative 2: [Name]
Detailed description and analysis.

### Rationale for Chosen Approach
Why this approach was selected.

## Test Strategy

### Unit Tests
- [ ] Test case 1: [Description]
- [ ] Test case 2: [Description]

### Lit Tests
- [ ] Parse/print round-trip
- [ ] Verifier tests (valid and invalid)
- [ ] Lowering transformation tests

### Edge Cases
- [ ] Empty input
- [ ] Single element
- [ ] Large input
- [ ] Dynamic shapes

### Integration Tests
- [ ] End-to-end pipeline test

## Risks

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| [Risk 1] | High/Med/Low | High/Med/Low | [Mitigation] |
| [Risk 2] | High/Med/Low | High/Med/Low | [Mitigation] |

## Open Questions

- [ ] Question 1?
- [ ] Question 2?

## Dependencies

- [Dependency 1]: [Description]
- [Dependency 2]: [Description]

## Implementation Plan

### Phase 1: [Name]
- [ ] Task 1
- [ ] Task 2

### Phase 2: [Name]
- [ ] Task 3
- [ ] Task 4

---

## Review History

### Review 1 (YYYY-MM-DD)
**Reviewer**: [Name]
**Decision**: [Approved / Request Changes / Rejected]

**Feedback**:
- [Feedback item 1]
- [Feedback item 2]

**Resolution**:
- [How feedback was addressed]
