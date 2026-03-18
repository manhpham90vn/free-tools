#!/usr/bin/env bash
# Fix script - auto-fixes formatting and linting issues
# Usage: ./scripts/fix.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Ensure environment ──
ensure_venv

# ── Run fixes ──
run_step "Formatting..."  ruff format .

run_step "Linting..."  ruff check . --fix

ok "All fixes applied"
