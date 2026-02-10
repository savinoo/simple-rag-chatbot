# ROADMAP — Upgrade `simple-rag-chatbot` to “Knowledge Brain” MVP

Goal: Turn this repo into a credible MVP for the FitnessSuperstore-style RAG job.

## Phase 0 (DONE — local)
- Grounded answers with citations ([S1]… + Sources section)
- Retrieval confidence gating + explicit fallback: “Not in KB yet.”
- Manifest-driven ingestion (local JSON)
- Audit logging (JSONL)
- Retrieval evaluation script (recall@k)

## Phase 1 (NEXT — make it portfolio-grade)
1) **Manifest v1.1 (YAML + metadata)**
   - Support YAML manifest with per-doc metadata:
     - title, path/url, tags/department, allowed_roles, doc_id
   - Keep JSON supported.

2) **Structured audit log (SQLite)**
   - Store: ts, user/session id (optional), question, best_score, top_sources, answer, status
   - Add a tiny viewer page in Streamlit (admin tab) to browse logs.

3) **Section-level citations (best-effort)**
   - For markdown: infer section heading path during chunking.
   - For PDF: include page number.
   - For txt: fallback to chunk number.

4) **Admin controls**
   - Manual re-index button + sync status.

5) **Golden set runner**
   - Improve eval harness to report per-question hits/misses.

## Phase 2 (Stretch — aligns with “nice-to-have”)
- Connectors skeleton:
  - Google Drive/Docs/Sheets interfaces (no credentials required in repo; documented placeholders)
- Scheduled sync stub (cron-friendly CLI command)
- Role-based filtering based on manifest metadata

## Output artifacts (for Upwork)
- Screenshot(s) showing:
  - Answer with citations + Sources section
  - “Not in KB yet” refusal behavior
  - Admin log viewer
  - Manifest ingestion UI
- README updated with a crisp “Why this is safe” section

## Approval gates
- I’ll implement locally freely.
- **I will only push to GitHub after Lucas explicitly says: “OK, push to GitHub”.**
