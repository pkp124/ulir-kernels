# Design Review Workflow

This document describes the design review process for KernelSmith.

## When Design Review is Required

Design review is required for:
- New kernel operations
- New transformation passes
- New target backends
- Significant changes to existing components
- Changes affecting public interfaces

## Design Review Process

### 1. Prepare Design Document

**Author (Architect Agent)**

Create design document using template:
```bash
make new-design ID=XXX TITLE="Feature Title"
```

Required sections:
- Context and problem statement
- Requirements (linked to spec)
- Proposed design
- Alternatives considered
- Test strategy
- Risks and mitigations

### 2. Self-Review

Before requesting review:
- [ ] All sections complete
- [ ] Requirements traceable
- [ ] Alternatives genuinely evaluated
- [ ] No obvious gaps

### 3. Request Review

Mark document status as "Under Review"

### 4. Conduct Review

**Reviewer (Reviewer Agent)**

Review with critical mindset:

#### Requirements Check
- Are all spec requirements addressed?
- Are acceptance criteria testable?
- Are edge cases considered?

#### Design Evaluation
- Is the approach sound?
- Does it fit the architecture?
- Is it appropriately scoped?

#### Alternative Analysis
- Were alternatives fairly evaluated?
- Is the chosen approach justified?

#### Risk Assessment
- Are risks identified?
- Are mitigations adequate?

### 5. Document Feedback

Use structured format:

```markdown
## Review Feedback

### Decision
☐ Approved
☐ Request Changes
☐ Rejected

### Critical Issues
1. [Must fix before approval]

### Major Concerns
1. [Should address]

### Suggestions
1. [Nice to have]

### Questions
1. [Need clarification]
```

### 6. Address Feedback

**Author**

- Address all critical issues
- Discuss major concerns
- Consider suggestions
- Answer questions
- Update design document

### 7. Re-Review (if needed)

Repeat review cycle until approved.

### 8. Approval

**Reviewer**

- Mark as "Approved" with date
- Add approver name
- Document any conditions

## Review Criteria

### Must Have (Critical)
- All requirements addressed
- No correctness issues
- No safety/security issues

### Should Have (Major)
- Alternatives considered
- Risks mitigated
- Test strategy complete

### Nice to Have (Minor)
- Optimal design choices
- Comprehensive documentation
- Future extensibility

## Anti-Patterns to Watch

### Rubber Stamping
- Don't approve without thorough review
- Ask probing questions
- Challenge assumptions

### Bikeshedding
- Don't focus on trivial issues
- Prioritize by impact
- Keep feedback actionable

### Design by Committee
- Reviewer provides feedback, not redesign
- Author makes final decisions
- Respect different valid approaches

## Templates

### Quick Review (Minor Changes)

```markdown
## Review: DES-XXX

**Decision**: Approved

**Notes**: Minor change, LGTM.
- [Any small suggestions]
```

### Full Review (Significant Changes)

```markdown
## Review: DES-XXX

### Summary
[Overall assessment]

### Decision
[Approved / Request Changes / Reject]

### Critical Issues
1. **Issue**: [Description]
   **Impact**: [Why it matters]
   **Suggestion**: [How to fix]

### Major Concerns
1. [Description and suggestion]

### Minor Suggestions
1. [Optional improvements]

### Positive Aspects
- [What works well]

### Questions
1. [Clarifications needed]
```
