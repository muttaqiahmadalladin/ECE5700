#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${1:-$ROOT_DIR/.venv}"

python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"

echo "Environment is ready at $VENV_DIR"
echo "Optional cache warm-up:"
echo "  source \"$VENV_DIR/bin/activate\""
echo "  python \"$ROOT_DIR/scripts/prefetch_assets.py\" --cache-dir \"\$HOME/.cache/huggingface\""

