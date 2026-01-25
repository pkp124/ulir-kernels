#!/bin/bash
# ==============================================================================
# Create a new task file
# ==============================================================================

set -e

ID=$1
TITLE=$2

if [ -z "$ID" ] || [ -z "$TITLE" ]; then
    echo "Usage: $0 <task_id> \"<title>\""
    exit 1
fi

TASK_FILE="tasks/TASK-${ID}.md"

if [ -f "$TASK_FILE" ]; then
    echo "Task $TASK_FILE already exists!"
    exit 1
fi

cat > "$TASK_FILE" << EOF
# TASK-${ID}: ${TITLE}

## Status
[ ] Not Started

## Priority
P2 (Medium)

## Description

<!-- Describe what needs to be done and why -->

## Acceptance Criteria

- [ ] Criterion 1
- [ ] Criterion 2
- [ ] Tests passing
- [ ] Documentation updated

## Implementation Notes

<!-- Technical details, design decisions, approach -->

## Dependencies

<!-- List any blocking or related tasks -->

## Verification

\`\`\`bash
make verify
\`\`\`

## Log

<!-- Track progress, decisions, blockers -->

### $(date +%Y-%m-%d)
- Task created
EOF

echo "Created: $TASK_FILE"
echo ""
echo "Edit the task file to add details."
