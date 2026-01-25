# KernelSmith Task Backlog

This file tracks all tasks and their current status. The Manager Agent maintains this file and prioritizes milestone-aligned work.

## Current Milestone

**M2: Core Dialect** - Working kernel dialect with basic operations

See [ROADMAP.md](../ROADMAP.md) for full milestone details.

---

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

---

## Active Work

| Task | Title | Phase | Milestone | Blocker |
|------|-------|-------|-----------|---------|
| - | - | - | - | - |

---

## Milestone M2 Tasks (Current Priority)

These tasks are required for M2 completion. **Work on these first.**

| Task | Title | Status | Depends On | Required |
|------|-------|--------|------------|----------|
| TASK-002 | Kernel Dialect Core | ⬜ Backlog | TASK-001 ✅ | **Yes** |
| TASK-003 | ks.matmul Operation | ⬜ Backlog | TASK-002 | **Yes** |
| TASK-004 | Activation Ops | ⬜ Backlog | TASK-002 | **Yes** |
| TASK-005 | ks.attention Operation | ⬜ Backlog | TASK-002 | **Yes** |

### M2 Progress: 0/4 tasks complete

---

## Milestone M3 Tasks (Future)

Do not start until M2 is complete.

| Task | Title | Status | Depends On | Required |
|------|-------|--------|------------|----------|
| TASK-010 | Lower to Linalg Pass | ⬜ Backlog | M2 | **Yes** |
| TASK-011 | Tiling Pass | ⬜ Backlog | TASK-010 | **Yes** |
| TASK-012 | Vectorization Pass | ⬜ Backlog | TASK-011 | **Yes** |

---

## Milestone M4 Tasks (Future)

Do not start until M3 is complete.

| Task | Title | Status | Depends On | Required |
|------|-------|--------|------------|----------|
| TASK-020 | RVV Lowering Pass | ⬜ Backlog | M3 | **Yes** |
| TASK-021 | RVV Optimization | ⬜ Backlog | TASK-020 | **Yes** |
| TASK-022 | RVV Testing (QEMU) | ⬜ Backlog | TASK-020 | **Yes** |

---

## Task Dependency Graph

```
MILESTONE M1 ✅
    │
    └─→ MILESTONE M2 (Current)
            │
            ├─→ TASK-002 Kernel Dialect Core ⬜
            │       │
            │       ├─→ TASK-003 ks.matmul ⬜
            │       ├─→ TASK-004 Activation Ops ⬜
            │       └─→ TASK-005 ks.attention ⬜
            │
            └─→ MILESTONE M3 (Blocked by M2)
                    │
                    ├─→ TASK-010 Lower to Linalg ⬜
                    │       │
                    │       └─→ TASK-011 Tiling ⬜
                    │               │
                    │               └─→ TASK-012 Vectorize ⬜
                    │
                    └─→ MILESTONE M4 (Blocked by M3)
                            │
                            ├─→ TASK-020 RVV Lowering ⬜
                            ├─→ TASK-021 RVV Optimize ⬜
                            └─→ TASK-022 RVV Testing ⬜
```

---

## Completed Tasks

| Task | Title | Milestone | Completed |
|------|-------|-----------|-----------|
| TASK-001 | Infrastructure | M1 | 2025-01-25 |

---

## Blocked Tasks

| Task | Blocker | Waiting For |
|------|---------|-------------|
| TASK-003 | TASK-002 | Dialect infrastructure |
| TASK-004 | TASK-002 | Dialect infrastructure |
| TASK-005 | TASK-002 | Dialect infrastructure |
| M3 Tasks | M2 | M2 completion |
| M4 Tasks | M3 | M3 completion |

---

## Manager Decision Log

### 2025-01-25
- M1 complete
- Starting M2
- **Priority**: TASK-002 (unblocks all other M2 tasks)
- **Recommendation**: Begin TASK-002 with design phase

---

## Next Actions

**Immediate Priority:** TASK-002 (Kernel Dialect Core)

**Why:** 
- Required for M2 milestone
- Blocks TASK-003, TASK-004, TASK-005
- All prerequisites met (TASK-001 ✅)

**Steps:**
1. Create design doc: `make new-design ID=002 TITLE="Kernel Dialect Core"`
2. Design review (Reviewer agent)
3. TDD implementation (Implementer agent)
4. Verification (Verifier agent)
