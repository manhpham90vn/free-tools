#!/usr/bin/env bash
# Check script - runs all quality checks: compile, format, lint, type check
# Usage: ./scripts/check.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/common.sh"

# ── Ensure environment ──
ensure_venv

# ── Run checks ──
run_step "Formatting check..."  ruff format . --check

run_step "Linting..."  ruff check .

run_step "Type checking..."  mypy .

ok "All checks passed"