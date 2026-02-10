#!/usr/bin/env bash
set -euo pipefail

# Simple demo runner for the repo.
# Assumes:
# - Python 3 is installed
# - OPENAI_API_KEY is set in your environment
# - (optional) MANIFEST_PATH set; defaults to manifest.example.yaml

MANIFEST_PATH_DEFAULT="manifest.example.yaml"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_PATH_DEFAULT}"

echo "[demo] Using MANIFEST_PATH=$MANIFEST_PATH"

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "[demo] ERROR: OPENAI_API_KEY is not set."
  echo "       Export it first: export OPENAI_API_KEY=..."
  exit 1
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export MANIFEST_PATH="$MANIFEST_PATH"

echo "[demo] Running retrieval eval -> reports/latest/"
python eval_retrieval.py --golden data/golden.sample.jsonl --k 5 --out-dir reports/latest || true

echo "[demo] Starting Streamlit app"
echo "       streamlit run app.py"
streamlit run app.py
