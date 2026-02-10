"""Retrieval evaluation harness (local).

Golden set format (JSONL), one per line:
{
  "question": "...",
  "expected_sources": ["returns.md", "qc_checklist.pdf"]
}

We compute recall@k over expected_sources based on retrieved chunk metadata.source.

Usage:
  python eval_retrieval.py --golden data/golden.jsonl --k 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from rag_pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--k", type=int, default=config.K_DOCUMENTS)
    ap.add_argument("--threshold", type=float, default=getattr(config, "RETRIEVAL_THRESHOLD", 0.35))
    args = ap.parse_args()

    pipe = RAGPipeline(config.OPENAI_API_KEY)

    # For evaluation you must have indexed docs already; simplest is to provide a manifest via env.
    manifest = getattr(config, "MANIFEST_PATH", None)
    if not manifest:
        raise SystemExit("Set MANIFEST_PATH in config.py (or env) to load documents for evaluation.")

    manifest_path = Path(manifest)
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    pipe.load_manifest_paths(m["documents"])

    total = 0
    hit = 0

    for line in Path(args.golden).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = rec["question"]
        expected = set(rec.get("expected_sources", []))

        retrieved = pipe._retrieve(q, k=args.k)
        got = {r.doc.metadata.get("source") for r in retrieved}

        total += len(expected)
        hit += len(expected & got)

    recall = hit / total if total else 0.0
    print(json.dumps({"k": args.k, "expected_total": total, "hit": hit, "recall": recall}, indent=2))


if __name__ == "__main__":
    main()
