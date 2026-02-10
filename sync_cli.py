"""Sync CLI stub (local-first).

Goal:
- Provide a cron-friendly command that reads a manifest and (re)indexes documents.
- Track sync runs and per-document state in SQLite.

Notes:
- In this local MVP we rebuild the index from the manifest for correctness.
- The SQLite `doc_state` table enables future incremental upserts (hash diff).

Usage:
  python sync_cli.py --manifest manifest.example.yaml

Exit codes:
  0 success
  1 failure
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import config
from audit_sqlite import SQLiteAudit
from manifest_loader import load_manifest
from rag_pipeline import RAGPipeline


def file_hash(path: str) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    args = ap.parse_args()

    audit = SQLiteAudit(getattr(config, "AUDIT_DB_PATH", "logs/audit.db"))
    audit.ensure_sync_tables()

    docs = load_manifest(args.manifest)

    # Compute hashes (best-effort; may be slow for huge PDFs, but OK for MVP)
    doc_states = []
    errors = []
    changed = 0

    for d in docs:
        try:
            h = file_hash(d.path)
            prev = audit.get_doc_state(d.id or d.path)
            is_changed = prev is None or prev.get("content_hash") != h
            if is_changed:
                changed += 1
            doc_states.append({"doc_id": d.id or d.path, "path": d.path, "content_hash": h})
        except Exception as e:
            errors.append({"doc": getattr(d, "path", None), "error": str(e)})

    # Rebuild index (correctness-first MVP)
    pipe = RAGPipeline()
    pipe.load_manifest_docs(docs)

    # Persist doc states
    for s in doc_states:
        audit.upsert_doc_state(s["doc_id"], s["path"], s["content_hash"])

    audit.log_sync_run(
        manifest_path=args.manifest,
        docs_total=len(docs),
        docs_indexed=len(docs) - len(errors),
        docs_failed=len(errors),
        errors=errors,
        changed_docs=changed,
    )

    print(
        json.dumps(
            {
                "ok": len(errors) == 0,
                "manifest": args.manifest,
                "docs_total": len(docs),
                "changed_docs": changed,
                "errors": errors,
            },
            indent=2,
        )
    )

    return 0 if len(errors) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
