# Design Documents

This directory contains design documents for KernelSmith features.

## Purpose

Design documents:
1. **Capture decisions** before implementation
2. **Enable review** of approaches
3. **Document rationale** for future reference
4. **Ensure alignment** with project goals

## When to Write a Design Doc

A design document is required for:
- New kernel operations
- New transformation passes
- New target backends
- Significant changes to existing components
- Changes affecting public interfaces

## Design Document Lifecycle

```
Draft → Under Review → Approved → Implemented
```

| Status | Description |
|--------|-------------|
| **Draft** | Initial creation, not ready for review |
| **Under Review** | Ready for critical review |
| **Approved** | Review passed, ready for implementation |
| **Implemented** | Implementation complete |

## Creating a Design Document

```bash
make new-design ID=001 TITLE="Feature Name"
```

This creates `docs/design/DES-001-feature-name.md` from the template.

## Naming Convention

```
DES-{ID}-{short-title}.md
```

Examples:
- `DES-001-matmul-operation.md`
- `DES-002-tiling-pass.md`
- `DES-003-rvv-lowering.md`

## Design Documents Index

| ID | Title | Status | Author |
|----|-------|--------|--------|
| [DES-001](DES-001-example.md) | Example | Template | - |

## Review Process

See `.agents/workflows/design-review.md` for the full review process.

Quick summary:
1. Create design doc (Author)
2. Self-review against checklist
3. Mark as "Under Review"
4. Critical review (Reviewer)
5. Address feedback
6. Get approval
7. Implement
