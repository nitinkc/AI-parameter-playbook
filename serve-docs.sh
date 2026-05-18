#!/bin/bash
# Docs server helper for AI Parameter Playbook
# Ensures mkdocs runs from the isolated .venv-docs environment

set -e

PLAYBOOK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DOCS="$PLAYBOOK_DIR/.venv-docs"

if [ ! -d "$VENV_DOCS" ]; then
  echo "Creating .venv-docs..."
  python3 -m venv "$VENV_DOCS"
fi

if [ ! -f "$VENV_DOCS/bin/mkdocs" ]; then
  echo "Installing docs dependencies..."
  "$VENV_DOCS/bin/pip" install -q -r "$PLAYBOOK_DIR/requirements-docs.txt"
fi

echo "Starting MkDocs server..."
echo "→ Open http://localhost:8000"
"$VENV_DOCS/bin/mkdocs" serve

