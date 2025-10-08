#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${PROJECT_DIR}/.venv"

if ! command -v python3 >/dev/null 2>&1; then
  echo "[!] Python3 saknas. Installera Python 3.10 eller senare och försök igen." >&2
  exit 1
fi

PYTHON_BIN="python3"
if command -v python >/dev/null 2>&1; then
  PYTHON_CANDIDATE="$(command -v python)"
  if "${PYTHON_CANDIDATE}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
    PYTHON_BIN="${PYTHON_CANDIDATE}"
  fi
fi

"${PYTHON_BIN}" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' || {
  echo "[!] Python-versionen måste vara >= 3.10" >&2
  exit 1
}

if [ ! -d "${VENV_DIR}" ]; then
  echo "[*] Skapar virtuell miljö i ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "[*] Virtuell miljö finns redan i ${VENV_DIR}"
fi

# Aktivera miljön
if [ -f "${VENV_DIR}/bin/activate" ]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/bin/activate"
elif [ -f "${VENV_DIR}/Scripts/activate" ]; then
  # shellcheck source=/dev/null
  source "${VENV_DIR}/Scripts/activate"
else
  echo "[!] Kunde inte hitta aktiveringsskriptet för den virtuella miljön." >&2
  exit 1
fi

python -m pip install --upgrade pip
python -m pip install --upgrade wheel setuptools

if [ -f "${PROJECT_DIR}/requirements.txt" ]; then
  python -m pip install -r "${PROJECT_DIR}/requirements.txt"
else
  echo "[!] Hittar inte requirements.txt" >&2
  exit 1
fi

echo "[✓] Installation klar. Aktivera miljön med:"
echo "    source ${VENV_DIR}/bin/activate" 

echo "[i] Starta en träningssession med exempelvis:"
echo "    python train_dqn.py"
