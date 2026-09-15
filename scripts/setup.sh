#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv"

cd "${PROJECT_ROOT}"

echo "Creating ignored runtime folders..."
mkdir -p artifact artifacts saved_models data Notebook/catboost_info

if [ ! -f ".env" ]; then
  echo "Creating .env from .env.example..."
  cp .env.example .env
fi

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating virtual environment at .venv..."
  python3 -m venv "${VENV_DIR}"
fi

echo "Installing Python dependencies..."
"${VENV_DIR}/bin/python" -m pip install --upgrade pip
"${VENV_DIR}/bin/pip" install -r requirements.txt

echo "Setup complete."
echo "Activate the environment with: source .venv/bin/activate"
