# KernelSmith Task Backlog

This file tracks all tasks and their current status. The Manager Agent maintains this file.

## Status Legend

| Symbol | Status |
|--------|--------|
| ⬜ | Backlog (not started) |
| 📋 | Specifying |
| 📐 | Designing |
| 🔍 | Under Review |
| 🔨 | Implementing |
| ✅ | Complete |
| ⏸️ | Blocked |

## Active Work

| Task | Title | Phase | Assignee | Blocker |
|------|-------|-------|----------|---------|
| - | - | - | - | - |

## Task Dependency Graph

```
TASK-001 Infrastructure ✅
    │
    ├─→ TASK-002 Kernel Dialect Core ⬜
    │       │
    │       ├─→ TASK-003 ks.matmul ⬜
    │       ├─→ TASK-004 ks.relu/gelu/softmax ⬜
    │       └─→ TASK-005 ks.attention ⬜
    │
    ├─→ TASK-010 Tiling Pass ⬜
    │       │
    │       └─→ TASK-011 Vectorization Pass ⬜
    │
    └─→ TASK-020 RVV Lowering ⬜
```

## Backlog by Priority

### P0 - Critical (Blocking)

| ID | Title | Status | Depends On |
|----|-------|--------|------------|
| TASK-002 | Kernel Dialect Core | ⬜ Backlog | TASK-001 ✅ |

### P1 - High (Core Functionality)

| ID | Title | Status | Depends On |
|----|-------|--------|------------|
| TASK-003 | ks.matmul Operation | ⬜ Backlog | TASK-002 |
| TASK-010 | Tiling Pass | ⬜ Backlog | TASK-002 |
| TASK-020 | RVV Lowering | ⬜ Backlog | TASK-010 |

### P2 - Medium (Important Features)

| ID | Title | Status | Depends On |
|----|-------|--------|------------|
| TASK-004 | Activation Ops (relu, gelu, softmax) | ⬜ Backlog | TASK-002 |
| TASK-005 | ks.attention Operation | ⬜ Backlog | TASK-002 |
| TASK-011 | Vectorization Pass | ⬜ Backlog | TASK-010 |

### P3 - Low (Nice to Have)

| ID | Title | Status | Depends On |
|----|-------|--------|------------|
| - | - | - | - |

## Completed Tasks

| ID | Title | Completed | Notes |
|----|-------|-----------|-------|
| TASK-001 | Infrastructure | 2025-01-25 | CMake, CTest, Agent system |

## Blocked Tasks

| ID | Title | Blocker | Owner |
|----|-------|---------|-------|
| - | - | - | - |

---

## Next Actions

**Recommended next task:** TASK-002 (Kernel Dialect Core)

**Prerequisites met:** ✅ Yes (TASK-001 complete)

**Required steps:**
1. Create design doc: `make new-design ID=002 TITLE="Kernel Dialect Core"`
2. Design review
3. TDD implementation
4. Verification
