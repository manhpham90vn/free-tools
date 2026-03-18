#!/usr/bin/env bash

if ! command -v python >/dev/null 2>&1; then
  echo "❌ python not found in PATH"
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  python -m venv .venv

  echo "⬆️ Installing dependencies..."
  if [ -f "requirements.txt" ]; then
    ./.venv/bin/pip install -r requirements.txt
  else
    echo "⚠️ No requirements.txt found"
  fi
fi

./.venv/bin/python -m compileall . -q
./.venv/bin/ruff format . --check
./.venv/bin/ruff check .
./.venv/bin/mypy .