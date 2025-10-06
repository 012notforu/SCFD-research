#!/bin/bash
set -euo pipefail
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev]"
pre-commit install
pytest -q || true  # don't fail bootstrap if tests missing initially
echo "Bootstrap OK. Activate with: source .venv/bin/activate"