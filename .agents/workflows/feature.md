# Feature Development Workflow

This document describes the end-to-end workflow for developing a new feature in KernelSmith.

## Overview

```
Specification → Design → Review → Implement (TDD) → Verify → Merge
```

## Phase 1: Specification

**Agent**: Architect

### Steps

1. **Check existing spec** in `specs/`
2. **Create or update spec** if needed
3. **Define requirements** clearly

### Output

- Specification document in `specs/`
- Clear requirements list
- Acceptance criteria

### Checklist
- [ ] Spec exists or created
- [ ] Requirements numbered (REQ-1, REQ-2, ...)
- [ ] Acceptance criteria testable

---

## Phase 2: Design

**Agent**: Architect

### Steps

1. **Create design document**
   ```bash
   make new-design ID=001 TITLE="Feature Name"
   ```

2. **Document the approach**
   - How it fits in the architecture
   - Component design
   - Interfaces

3. **Analyze alternatives**
   - List at least 2-3 alternatives
   - Evaluate pros/cons
   - Justify chosen approach

4. **Define test strategy**
   - What tests are needed
   - Edge cases to cover

5. **Identify risks**
   - Technical risks
   - Mitigation plans

### Output

- Design document in `docs/design/DES-XXX-feature.md`
- Status: "Under Review"

### Checklist
- [ ] Design doc created from template
- [ ] All requirements addressed
- [ ] Alternatives documented
- [ ] Test strategy defined
- [ ] Risks identified

---

## Phase 3: Design Review

**Agent**: Reviewer

### Steps

1. **Critical review** of design document
2. **Check requirements coverage**
3. **Evaluate trade-offs**
4. **Identify gaps and risks**
5. **Provide feedback**

### Review Criteria

| Aspect | Questions |
|--------|-----------|
| Requirements | All addressed? Testable? |
| Architecture | Fits existing design? |
| Alternatives | Genuinely considered? |
| Risks | Identified and mitigated? |
| Testability | Can be tested? |

### Output

- Review comments in design doc
- Decision: Approve / Request Changes / Reject

### Checklist
- [ ] Requirements verified
- [ ] Architecture reviewed
- [ ] Risks acceptable
- [ ] Test strategy adequate
- [ ] Explicit approval given

---

## Phase 4: Implementation (TDD)

**Agent**: Implementer

### Steps

1. **Write failing tests first**
   ```bash
   # Create test file
   touch tests/lit/Dialect/Kernel/new-op.mlir
   
   # Run to confirm failure
   ctest --test-dir build -R "new-op"
   ```

2. **Implement minimal code** to pass tests

3. **Refactor** while keeping tests green

4. **Add edge case tests**

5. **Commit incrementally**
   ```bash
   git add -A
   git commit -m "feat(dialect): add ks.new_op operation"
   ```

### TDD Cycle

```
Write Test → See Fail → Implement → See Pass → Refactor → Repeat
```

### Output

- Passing tests
- Implementation code
- Incremental commits

### Checklist
- [ ] Tests written first
- [ ] All tests passing
- [ ] Edge cases covered
- [ ] Code formatted
- [ ] Commits atomic and descriptive

---

## Phase 5: Verification

**Agent**: Verifier

### Steps

1. **Run full test suite**
   ```bash
   ctest --test-dir build --output-on-failure
   ```

2. **Verify specification compliance**
   - Check each requirement
   - Confirm tests exist

3. **Quality checks**
   ```bash
   make lint
   make format
   ```

4. **Create verification report**

### Output

- Verification report
- All tests passing
- Quality checks clean

### Checklist
- [ ] All tests pass
- [ ] Spec requirements verified
- [ ] No lint errors
- [ ] Code formatted
- [ ] Verification report created

---

## Phase 6: Merge

### Steps

1. **Update design doc status** to "Implemented"
2. **Update task status** if applicable
3. **Final commit and push**

### Checklist
- [ ] Design doc status updated
- [ ] All changes committed
- [ ] Pushed to branch
