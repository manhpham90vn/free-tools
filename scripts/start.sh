#!/usr/bin/env bash
# Free Antigravity - Start Script
# Kills all Antigravity IDE processes and starts the MITM proxy

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

CYAN='\033[0;36m'
GREEN='\033[0;32m'
NC='\033[0m'

info() { echo -e "${CYAN}▸${NC} $*"; }
ok()   { echo -e "${GREEN}✔${NC} $*"; }

echo
info "Free Antigravity Start"
echo

# Kill all Antigravity related processes
info "Killing Antigravity processes..."
pkill -f "antigravity"     2>/dev/null || true
pkill -f "language_server" 2>/dev/null || true
pkill -f "chrome-devtools" 2>/dev/null || true
sleep 1
ok "Done."

echo
info "Starting MITM proxy on port 443..."
echo

cd "$PROJECT_ROOT"
.venv/bin/python main.py start
