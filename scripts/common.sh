#!/usr/bin/env bash
# Common utilities shared across all scripts.
# Source this file: source "$(dirname "$0")/common.sh"

set -euo pipefail

# ── Project root (one level up from scripts/) ──
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
VENV_BIN="$VENV_DIR/bin"

# ── Colors ──
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

info()  { echo -e "${CYAN}▸${NC} $*"; }
ok()    { echo -e "${GREEN}✔${NC} $*"; }
warn()  { echo -e "${YELLOW}⚠${NC} $*"; }
fail()  { echo -e "${RED}✖${NC} $*"; exit 1; }

# ── Ensure python exists ──
ensure_python() {
  command -v python >/dev/null 2>&1 || fail "python not found in PATH"
}

# ── Ensure venv + deps are ready ──
ensure_venv() {
  ensure_python

  if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment..."
    python -m venv "$VENV_DIR"

    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
      info "Installing dependencies..."
      "$VENV_BIN/pip" install -q -r "$PROJECT_ROOT/requirements.txt"
    else
      warn "No requirements.txt found"
    fi
    ok "Virtual environment ready"
  fi
}

# ── Run a tool from the venv with a label ──
run_step() {
  local label="$1"; shift
  info "$label"
  "$VENV_BIN/$@"
}
