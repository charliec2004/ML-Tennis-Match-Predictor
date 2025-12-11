#!/usr/bin/env bash
set -euo pipefail

# Simple one-time setup script for new clones.
# Creates a virtual environment in ./venv and installs project dependencies.

PYTHON_BIN="${PYTHON_BIN:-}"

choose_python() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "$PYTHON_BIN"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    echo "python3"
  elif command -v python >/dev/null 2>&1; then
    echo "python"
  else
    echo "Error: Python is not installed or not on PATH." >&2
    exit 1
  fi
}

PYTHON=$(choose_python)

echo "Using Python: $($PYTHON --version)"

if [[ ! -d "venv" ]]; then
  echo "Creating virtual environment in ./venv ..."
  $PYTHON -m venv venv
else
  echo "Virtual environment already exists at ./venv (skipping creation)."
fi

# Activate the venv for this shell session
# shellcheck source=/dev/null
source venv/bin/activate
echo "Venv activated. Python now: $(python --version)"

echo "Upgrading pip/setuptools/wheel ..."
python -m pip install --upgrade pip setuptools wheel

echo "Installing dependencies from requirements.txt ..."
python -m pip install -r requirements.txt

cat <<'EOF'

Setup complete.

To start working next time:
  source venv/bin/activate

If you are on Windows PowerShell:
  .\venv\Scripts\Activate.ps1

EOF
