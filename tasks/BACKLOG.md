# KernelSmith Task Backlog

Manager Agent maintains this file. **Milestone completion is the priority.**

---

## Current Milestone: M1

### Goal
**One kernel (matmul) running on RISC-V RVV simulator with 3 size variants.**

### Success Criteria (10 total)
- [ ] ks.matmul parses, prints, verifies
- [ ] Lowering pipeline produces valid LLVM IR
- [ ] Generated RISC-V assembly is valid
- [ ] Runs on QEMU with RVV
- [ ] 64x64 f32 output matches reference
- [ ] 128x128 f32 output matches reference
- [ ] 256x256 f32 output matches reference
- [ ] Works with VLEN=128
- [ ] Works with VLEN=256
- [ ] Works with VLEN=512

### Progress: 0/10

---

## M1 Task Breakdown

| ID | Task | Status | Depends | Blocks |
|----|------|--------|---------|--------|
| M1.1 | ks.matmul operation | ⬜ | - | M1.2 |
| M1.2 | --ks-lower-to-linalg pass | ⬜ | M1.1 | M1.3 |
| M1.3 | --ks-tile pass | ⬜ | M1.2 | M1.4 |
| M1.4 | --ks-vectorize pass | ⬜ | M1.3 | M1.5 |
| M1.5 | --ks-lower-to-rvv pass | ⬜ | M1.4 | M1.6 |
| M1.6 | QEMU test harness | ⬜ | M1.5 | M1.7 |
| M1.7 | Verify 64x64 matmul | ⬜ | M1.6 | M1.10 |
| M1.8 | Verify 128x128 matmul | ⬜ | M1.6 | M1.10 |
| M1.9 | Verify 256x256 matmul | ⬜ | M1.6 | M1.10 |
| M1.10 | Multi-VLEN testing | ⬜ | M1.7, M1.8, M1.9 | - |

---

## Active Work

| Task | Description | Phase | Assignee |
|------|-------------|-------|----------|
| - | - | - | - |

---

## Dependency Graph

```
M1.1 ks.matmul
  │
  └─→ M1.2 Lower to Linalg
        │
        └─→ M1.3 Tiling
              │
              └─→ M1.4 Vectorization
                    │
                    └─→ M1.5 RVV Lowering
                          │
                          └─→ M1.6 QEMU Harness
                                │
                        ┌───────┼───────┐
                        ▼       ▼       ▼
                      M1.7    M1.8    M1.9
                      64x64  128x128  256x256
                        │       │       │
                        └───────┼───────┘
                                ▼
                              M1.10
                          Multi-VLEN Test
```

---

## Next Action

**Immediate Priority:** M1.1 (ks.matmul operation)

**Why:** First task in dependency chain, blocks all other M1 work.

**Workflow:**
1. ✅ Specification exists: `specs/kernels/matmul.md`
2. ⬜ Create design doc: `make new-design ID=M1-1 TITLE="ks.matmul Operation"`
3. ⬜ Design review
4. ⬜ TDD implementation
5. ⬜ Verification

---

## Blocked Work

| Task | Blocked By | Notes |
|------|------------|-------|
| M1.2 - M1.10 | M1.1 | Waiting for matmul op |
| M2 tasks | M1 | Do not start until M1 complete |

---

## Completed

| Task | Completed | Milestone |
|------|-----------|-----------|
| Infrastructure | 2025-01-25 | (Pre-M1) |

---

## Manager Notes

### 2025-01-25
- Roadmap updated with quantifiable milestones
- M1 is end-to-end: matmul on QEMU with RVV
- All 10 M1 criteria must pass
- Starting with M1.1 (ks.matmul operation)
- **Do not skip to M2 until M1 is fully verified**
