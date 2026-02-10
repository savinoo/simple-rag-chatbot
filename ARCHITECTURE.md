# Architecture — `simple-rag-chatbot` as a “Knowledge Brain” MVP

This document describes the target architecture for upgrading `simple-rag-chatbot` into a portfolio-grade internal RAG assistant aligned with the FitnessSuperstore-style spec:
- Answer **only** from indexed documents (no guessing)
- Always include citations (doc + section/page where possible)
- "Not in KB yet." fallback when evidence is insufficient
- Manifest-driven ingestion + sync
- Private/internal orientation (auth + access control roadmap)
- Audit logging (Q/A + sources)
- Evaluation harness (golden set; recall@k)

## 1) Scope & Non-goals

### MVP scope (local-first)
- Local document ingestion (PDF/TXT/MD)
- Local manifest (JSON/YAML) is the *single source of truth* for what gets indexed
- Retrieval confidence gating + refusal mode
- Citations + sources section
- Admin: manual re-index + view logs
- Audit logging to SQLite (plus JSONL optional)
- Evaluation script for retrieval recall@k

### Non-goals (for MVP)
- Google Workspace credentials or deployment (we add clean stubs)
- Complex multimodal (OCR/images) beyond best-effort
- Production SSO (we outline it; keep local-only demo)

## 2) Components

### UI (Streamlit)
- **Chat**: user question → answer + citations
- **Admin panel**:
  - load manifest
  - manual re-index
  - view sync status
  - view audit logs
  - run quick eval (optional button)

### Core RAG Engine
- **Ingestion**: load docs from manifest; normalize; chunk; embed; index
- **Retriever**: similarity search (top-k) with relevance scores
- **Gating**: if best score < threshold → return `Not in KB yet.`
- **Grounded generation**: LLM sees only retrieved chunks with stable source IDs `[S1]..`
- **Citation enforcement**: require `[S#]` markers + append a `Sources` section

### Storage
- **Vector store**: Chroma (in-memory for demo; optionally persistent dir)
- **Relational store**: SQLite for audit logs + sync metadata

## 3) Data flow

1. **Manifest** (JSON/YAML) lists documents + metadata
2. **Indexer** loads docs → chunks with metadata:
   - `source` (filename)
   - `title` (optional)
   - `section_path` (best-effort for MD)
   - `page` (PDF)
   - `chunk` (per-source ordinal)
   - `tags`, `allowed_roles`, `doc_id` (from YAML manifest)
3. **Embed + upsert** into vector store
4. **Query**:
   - retrieve top-k chunks + scores
   - gate on threshold
   - generate answer grounded in retrieved chunks only
   - citations appear as `[S1]` etc.
5. **Audit log**:
   - store question, scores, selected sources, answer, status, timestamp

## 4) Module layout (target)

```
simple-rag-chatbot/
  app.py
  config.py
  rag_pipeline.py
  manifest_loader.py
  storage/
    audit_sqlite.py
    sync_state.py
  connectors/
    google_drive.py        # stub
    google_docs.py         # stub
    google_sheets.py       # stub
  eval/
    eval_retrieval.py
    schemas.py
  docs/
    ARCHITECTURE.md
    ROADMAP.md
```

## 5) SQLite schema (MVP)

### Table: `qa_logs`
- `id` INTEGER PRIMARY KEY
- `ts` TEXT (ISO)
- `question` TEXT
- `status` TEXT (`answered` | `not_in_kb` | `error`)
- `best_score` REAL
- `k` INTEGER
- `sources_json` TEXT (JSON array of source refs)
- `answer` TEXT

### Table: `sync_runs`
- `id` INTEGER PRIMARY KEY
- `ts` TEXT
- `manifest_path` TEXT
- `docs_total` INTEGER
- `docs_indexed` INTEGER
- `docs_failed` INTEGER
- `errors_json` TEXT

### Table: `doc_state`
- `doc_id` TEXT (from manifest; fallback to path)
- `path` TEXT
- `last_modified` TEXT (optional)
- `content_hash` TEXT
- `indexed_at` TEXT

## 6) Citations strategy

### Today (implemented)
- `[S#]` references map to retrieved chunks.
- `Sources` section lists `source (chunk N, page P)`.

### Next (planned)
- For Markdown:
  - preserve heading hierarchy during parsing
  - store `section_path` metadata and cite it
- For PDFs:
  - cite page number when loader provides it
- For Sheets/structured rows (future):
  - cite tab + row keys

## 7) Hallucination prevention

Layered approach:
- **Retrieval confidence gating** (refuse before generation if weak evidence)
- **Strict system prompt**: use *only* provided sources; otherwise `Not in KB yet.`
- **Citation requirement**: claims must have `[S#]` markers
- Optional v2: post-generation verification (claim-to-source check)

## 8) Access control roadmap

### v1 (MVP)
- Local-only demo: no auth
- Optional: “role” selector in UI (for demo) + manifest `allowed_roles`
- Enforce filtering at retrieval time (metadata filter)

### v2
- Google SSO (workspace domain allowlist)
- Doc-level permission sync from Drive
- Role-based retrieval enforced by metadata filtering

## 9) Sync strategy

### MVP (local)
- Manual re-index in UI
- CLI command `python -m sync --manifest ...` (cron-friendly)
- Change detection via `content_hash` to skip unchanged docs

### v2 (Google)
- Daily scheduled sync reading Google Sheets manifest
- Optional webhook-based push notifications

## 10) Quality & evaluation

- Golden set JSONL: `{question, expected_sources[]}`
- Metric: recall@k over expected sources
- Optional: track refusal rate, average best_score

## 11) Implementation milestones

1) SQLite audit logging + Streamlit log viewer
2) Markdown section-path parsing for better citations
3) Role filtering (demo) using manifest YAML `allowed_roles`
4) Sync CLI stub + sync_runs table
5) Eval report output (JSON + markdown)

