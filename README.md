# Simple RAG Chatbot (Upgraded)

A lightweight Retrieval-Augmented Generation (RAG) chatbot built with LangChain and Streamlit.

This project is being upgraded to better match a **private internal “knowledge brain”** spec:
- **Answer only from sources** (no guessing)
- **Citations in every answer**
- **“Not in KB yet.”** fallback when retrieval is weak
- **Manifest-driven ingestion (local)** as a stepping stone toward Google Sheets/Drive manifests
- **Audit logging** of Q/A + sources (JSONL)
- **Retrieval evaluation** (golden set → recall@k)

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
- `MODEL_NAME` (default: `gpt-4o-mini`)
- `K_DOCUMENTS` (default: `5`)
- `RETRIEVAL_THRESHOLD` (default: `0.35`)
- `LOG_PATH` (default: `logs/qa.jsonl`)

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
