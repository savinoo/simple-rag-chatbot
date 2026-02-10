"""Retrieval evaluation harness (local).

Golden set format (JSONL), one per line:
{
  "question": "...",
  "expected_sources": ["returns.md", "qc_checklist.pdf"]
}

We compute recall@k over expected_sources based on retrieved chunk metadata.source.

Usage:
  export MANIFEST_PATH=manifest.example.json
  python eval_retrieval.py --golden data/golden.jsonl --k 5 --out-dir reports/latest
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
from manifest_loader import load_manifest
from rag_pipeline import RAGPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", required=True)
    ap.add_argument("--k", type=int, default=config.K_DOCUMENTS)
    ap.add_argument("--out-dir", default="reports/latest")
    args = ap.parse_args()

    pipe = RAGPipeline(config.OPENAI_API_KEY)

    manifest = getattr(config, "MANIFEST_PATH", None)
    if not manifest:
        raise SystemExit("Set MANIFEST_PATH (env) to a JSON/YAML manifest to load documents for evaluation.")

    docs = load_manifest(manifest)
    pipe.load_manifest_docs(docs)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_q = []
    expected_total = 0
    hit_total = 0

    for line in Path(args.golden).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        q = rec["question"]
        expected = set(rec.get("expected_sources", []))

        retrieved = pipe._retrieve(q, k=args.k)
        got = [r.doc.metadata.get("source") for r in retrieved]
        got_set = set(got)

        hits = sorted(expected & got_set)
        misses = sorted(expected - got_set)

        expected_total += len(expected)
        hit_total += len(hits)

        per_q.append(
            {
                "question": q,
                "expected_sources": sorted(expected),
                "retrieved_sources": got,
                "hits": hits,
                "misses": misses,
            }
        )

    recall = hit_total / expected_total if expected_total else 0.0

    report = {
        "k": args.k,
        "expected_total": expected_total,
        "hit_total": hit_total,
        "recall_at_k": recall,
        "per_question": per_q,
    }

    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md_lines = [
        f"# Retrieval Report (recall@{args.k})",
        "",
        f"- expected_total: **{expected_total}**",
        f"- hit_total: **{hit_total}**",
        f"- recall@{args.k}: **{recall:.3f}**",
        "",
        "## Per-question breakdown",
        "",
    ]

    for i, row in enumerate(per_q, start=1):
        md_lines.append(f"### {i}. {row['question']}")
        md_lines.append(f"- Expected: {', '.join(row['expected_sources']) if row['expected_sources'] else '(none)'}")
        md_lines.append(f"- Retrieved: {', '.join(row['retrieved_sources']) if row['retrieved_sources'] else '(none)'}")
        md_lines.append(f"- Hits: {', '.join(row['hits']) if row['hits'] else '(none)'}")
        md_lines.append(f"- Misses: {', '.join(row['misses']) if row['misses'] else '(none)'}")
        md_lines.append("")

    (out_dir / "report.md").write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({"out_dir": str(out_dir), "recall_at_k": recall}, indent=2))


if __name__ == "__main__":
    main()
