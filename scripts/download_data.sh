#!/usr/bin/env bash
# Idempotent download of the Ling-Spam kernel and its dataset from Kaggle.
# Usage: scripts/download_data.sh [--force]
set -euo pipefail

KERNEL="sharmoul/ling-spam-dataset-eda"
DATASET="nitishabharathi/email-spam-dataset"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
KERNEL_DIR="$ROOT/notebooks/ling-spam-dataset-eda"
DATA_DIR="$ROOT/data/raw"

FORCE=0
[[ "${1:-}" == "--force" ]] && FORCE=1

# Prefer the project venv's CLI, fall back to PATH.
if [[ -x "$ROOT/.venv/bin/kaggle" ]]; then
  KAGGLE="$ROOT/.venv/bin/kaggle"
elif command -v kaggle >/dev/null 2>&1; then
  KAGGLE="$(command -v kaggle)"
else
  echo "ERROR: kaggle CLI not found. Run: python3 -m venv .venv && .venv/bin/pip install -r requirements.txt" >&2
  exit 1
fi

# Credentials: any one of these is accepted (never stored in this repo).
if [[ -z "${KAGGLE_API_TOKEN:-}" \
   && ( -z "${KAGGLE_USERNAME:-}" || -z "${KAGGLE_KEY:-}" ) \
   && ! -s "$HOME/.kaggle/access_token" \
   && ! -s "$HOME/.kaggle/kaggle.json" ]]; then
  cat >&2 <<MSG
ERROR: no Kaggle credentials found. Provide one of:
  - ~/.kaggle/access_token
  - KAGGLE_API_TOKEN env var
  - ~/.kaggle/kaggle.json (chmod 600)
  - KAGGLE_USERNAME and KAGGLE_KEY env vars
MSG
  exit 1
fi

if [[ $FORCE -eq 0 && -s "$KERNEL_DIR/kernel-metadata.json" ]]; then
  echo "Kernel already present in $KERNEL_DIR (use --force to re-download)"
else
  mkdir -p "$KERNEL_DIR"
  "$KAGGLE" kernels pull "$KERNEL" -p "$KERNEL_DIR" -m
fi

if [[ $FORCE -eq 0 && -s "$DATA_DIR/lingSpam.csv" ]]; then
  echo "Dataset already present in $DATA_DIR (use --force to re-download)"
else
  mkdir -p "$DATA_DIR"
  "$KAGGLE" datasets download "$DATASET" -p "$DATA_DIR" --unzip --force
fi

echo "Done."
