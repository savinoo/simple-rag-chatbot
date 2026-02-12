# Enterprise Knowledge Base RAG System

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_DB-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-UI-red)

A production-ready Retrieval-Augmented Generation (RAG) system built for **enterprise internal knowledge bases** (SOPs, policies, procedures).

## 🎯 Why This Matters

Unlike generic RAG chatbots that "guess" answers, this system is engineered for **trust and compliance**:

- **✅ Mandatory Citations** — Every answer links back to source documents (`[S1]`, `[S2]`)
- **✅ No Hallucinations** — "Not in KB yet" fallback when retrieval confidence is low
- **✅ Full Audit Trail** — JSONL + SQLite logging of every Q&A interaction with sources
- **✅ Quantitative Evaluation** — Built-in recall@k metrics via golden dataset
- **✅ Manifest-Driven Ingestion** — Controlled, reproducible KB updates (JSON/YAML)

## Features

- 📚 Document ingestion (PDF, TXT, Markdown)
- 🔍 Vector search with ChromaDB
- 🤖 OpenAI GPT integration
- 💬 Interactive Streamlit interface
- ✅ **Grounded answers with citations** (`[S1]`, `[S2]`)
- 🛑 **Safety**: if retrieval confidence is below a threshold, respond:
  - `Not in KB yet. Please add the relevant SOP/policy document to the knowledge base.`
- 🧾 Audit logs written to `logs/qa.jsonl` + SQLite audit DB (`logs/audit.db`) with Admin viewer

## Tech Stack

- **LangChain** - RAG pipeline orchestration
- **ChromaDB** - Vector database
- **OpenAI API** - Language model
- **Streamlit** - Web interface
- **Python 3.9+**

## Installation

```bash
git clone https://github.com/savinoo/simple-rag-chatbot.git
cd simple-rag-chatbot
pip install -r requirements.txt
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

```bash
streamlit run app.py
```

## Demo (1 command)

This repo ships with small demo docs + a golden set so you can generate a report and launch the UI quickly.

### Environment variables (LLM + embeddings)

Minimum required:

**Gemini**
```bash
export PROVIDER=gemini
export GOOGLE_API_KEY=...
# recommended (avoids Gemini embeddings 404s)
export EMBEDDINGS_PROVIDER=local
```

**OpenAI**
```bash
export PROVIDER=openai
export OPENAI_API_KEY=...
export EMBEDDINGS_PROVIDER=openai
```

Optional:
- `MODEL_NAME` overrides the LLM model name (applies to both providers).
- For Gemini, you may need to use a model your key supports (often with `models/...` prefix).

```bash
# Option A (OpenAI)
export PROVIDER=openai
export OPENAI_API_KEY=...
./run_demo.sh

# Option B (Gemini)
export PROVIDER=gemini
export GOOGLE_API_KEY=...

# Recommended: use local embeddings (avoids Gemini embeddings 404s)
export EMBEDDINGS_PROVIDER=local

./run_demo.sh
```

> Gemini uses `GOOGLE_API_KEY` (not `GEMINI_API_KEY`).
>
> If Gemini embeddings are not available for your key/account (common), use local embeddings:
>
> ```bash
> export EMBEDDINGS_PROVIDER=local
> ```
>
> If you get a Gemini model NOT_FOUND (404), you likely need to set a model your key supports:
>
> ```bash
> python -c "import os, google.generativeai as genai; genai.configure(api_key=os.environ['GOOGLE_API_KEY']); print('\\n'.join([m.name + ' :: ' + ','.join(m.supported_generation_methods) for m in genai.list_models()]))"
> export GEMINI_MODEL=models/gemini-1.0-pro
> ```

What it does:
- Loads docs from `manifest.example.yaml`
- Runs `eval_retrieval.py` and writes `reports/latest/report.md`
- Starts the Streamlit UI (Chat + Admin)

### Optional
Use a different manifest:

```bash
export MANIFEST_PATH=path/to/your-manifest.yaml
./run_demo.sh
```

You can either:
1) Upload documents in the sidebar, or
2) Load documents from a local manifest (see below).

## Manifest-driven ingestion (local)

Create a manifest JSON file (example: `manifest.example.json`):

```json
{ "documents": ["docs/policies/returns.md", "docs/sops/qc_checklist.pdf"] }
```

Then set `MANIFEST_PATH` (env var) or paste it in the sidebar:

```bash
export MANIFEST_PATH=manifest.example.json
```

## Configuration

Configuration is via env vars (see `config.py`):

Core:
- `PROVIDER` = `openai` | `gemini`
- `MODEL_NAME` (default depends on provider)
- `OPENAI_API_KEY` (when `PROVIDER=openai`)
- `GOOGLE_API_KEY` (when `PROVIDER=gemini`)

Embeddings:
- `EMBEDDINGS_PROVIDER` = `openai` | `gemini` | `local` (recommended for Gemini)
- `LOCAL_EMBEDDINGS_MODEL` (default: `all-MiniLM-L6-v2`)

RAG:
- `K_DOCUMENTS` (default: `5`)
- `RETRIEVAL_THRESHOLD` (default: `0.35`)

Logging:
- `LOG_PATH` (default: `logs/qa.jsonl`)
- `AUDIT_DB_PATH` (default: `logs/audit.db`)

## Retrieval evaluation (recall@k)

Golden set JSONL format (one per line):

```json
{"question":"What is the return window?","expected_sources":["returns.md"]}
```

Run:

```bash
python eval_retrieval.py --golden data/golden.sample.jsonl --k 5 --out-dir reports/latest
```

> Note: for evaluation, the pipeline loads docs via `MANIFEST_PATH`.

## Roadmap (next upgrades)

To fully match the Upwork job requirements, the next steps are:
- Google Drive/Docs/Sheets ingestion (via Google APIs)
- Scheduled daily sync + manual re-index controls
- Doc-level / role-based access control
- Slack bot interface
- Better section-level citations (heading-aware parsing)

## Project Structure

```
simple-rag-chatbot/
├── app.py
├── rag_pipeline.py
├── config.py
├── eval_retrieval.py
├── manifest.example.json
├── requirements.txt
└── README.md
```

## License

MIT

## Author

Lucas Lorenzo Savino  
AI Engineer | Agent Development & MLOps
