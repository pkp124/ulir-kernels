#!/bin/bash
# ==============================================================================
# Create a new design document from template
# ==============================================================================

set -e

ID=$1
TITLE=$2

if [ -z "$ID" ] || [ -z "$TITLE" ]; then
    echo "Usage: $0 <id> \"<title>\""
    echo "Example: $0 001 \"Matrix Multiplication Operation\""
    exit 1
fi

# Convert title to filename format
FILENAME=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | tr ' ' '-' | tr -cd '[:alnum:]-')
DESIGN_FILE="docs/design/DES-${ID}-${FILENAME}.md"

if [ -f "$DESIGN_FILE" ]; then
    echo "Design document $DESIGN_FILE already exists!"
    exit 1
fi

# Copy template
cp docs/design/TEMPLATE.md "$DESIGN_FILE"

# Update title
sed -i "s/DES-XXX: \[Feature Title\]/DES-${ID}: ${TITLE}/" "$DESIGN_FILE"

# Update date
TODAY=$(date +%Y-%m-%d)
sed -i "s/YYYY-MM-DD/$TODAY/g" "$DESIGN_FILE"

echo "Created: $DESIGN_FILE"
echo ""
echo "Next steps:"
echo "  1. Fill in the design document sections"
echo "  2. Mark status as 'Under Review' when ready"
echo "  3. Request design review"
