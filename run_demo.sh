#!/usr/bin/env bash
set -euo pipefail

# Simple demo runner for the repo.
# Assumes:
# - Python 3 is installed
# - Provider keys are set in your environment:
#     - PROVIDER=openai  -> OPENAI_API_KEY
#     - PROVIDER=gemini  -> GOOGLE_API_KEY
# - (optional) MANIFEST_PATH set; defaults to manifest.example.yaml

MANIFEST_PATH_DEFAULT="manifest.example.yaml"
MANIFEST_PATH="${MANIFEST_PATH:-$MANIFEST_PATH_DEFAULT}"

echo "[demo] Using MANIFEST_PATH=$MANIFEST_PATH"

PROVIDER="${PROVIDER:-openai}"

if [[ "$PROVIDER" == "gemini" ]]; then
  if [[ -z "${GOOGLE_API_KEY:-}" ]]; then
    echo "[demo] ERROR: GOOGLE_API_KEY is not set (required for PROVIDER=gemini)."
    echo "       Export it first: export GOOGLE_API_KEY=..."
    exit 1
  fi
else
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[demo] ERROR: OPENAI_API_KEY is not set (required for PROVIDER=openai)."
    echo "       Export it first: export OPENAI_API_KEY=..."
    exit 1
  fi
fi

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

export MANIFEST_PATH="$MANIFEST_PATH"
export PROVIDER="${PROVIDER:-openai}"

echo "[demo] Running retrieval eval -> reports/latest/"
python eval_retrieval.py --golden data/golden.sample.jsonl --k 5 --out-dir reports/latest || true

echo "[demo] Starting Streamlit app"
echo "       streamlit run app.py"
streamlit run app.py
