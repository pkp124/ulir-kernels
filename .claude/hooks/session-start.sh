#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

# Install Python dev dependencies (linter, test tools, MLIR test infra)
pip install --quiet \
  "pytest>=7.0" \
  "ruff>=0.1.0" \
  "lit>=18.0" \
  "filecheck>=0.0.24" \
  "numpy>=1.24" \
  "cmake-format>=0.6"

# Make project scripts executable
chmod +x "$CLAUDE_PROJECT_DIR"/scripts/*.sh 2>/dev/null || true

# Export PYTHONPATH so pytest can find project modules
echo "export PYTHONPATH=\"$CLAUDE_PROJECT_DIR\"" >> "$CLAUDE_ENV_FILE"
