# Screenshots checklist (for Upwork)

Goal: capture 3–5 screenshots that prove: correctness, citations, refusal behavior, and admin/auditability.

## 1) Answer with citations
- Ask a question that is clearly covered by a markdown SOP.
- Screenshot:
  - Answer text with inline citations like `[S1]`
  - `Sources` section showing doc + section/page.

## 2) “Not in KB yet” refusal
- Ask a question that is **not** in the docs.
- Screenshot the exact refusal:
  - `Not in KB yet.`

## 3) Manifest-driven ingestion
- In sidebar, load from `MANIFEST_PATH` using `manifest.example.yaml`.
- Screenshot the success toast showing number of docs loaded.

## 4) Admin: Audit log
- Go to **Admin** tab.
- Screenshot:
  - The audit table (status, best_score, sources)
  - An opened answer detail by log id.

## 4b) Role filtering (demo)
- In the sidebar, set Role to `cs`.
- Ask a question about Returns.
- Screenshot the answer + citations.
- Switch Role to `warehouse` and repeat.
- Screenshot the difference (if you add a doc restricted to warehouse).

## 5) Eval report
- Run eval:
  - `export MANIFEST_PATH=manifest.example.yaml`
  - `python eval_retrieval.py --golden data/golden.sample.jsonl --k 5 --out-dir reports/latest`
- Screenshot `reports/latest/report.md` or paste it into README.

## Suggested demo docs
Create a tiny demo doc: `docs/policies/returns.md` with headings (so section citations show).
