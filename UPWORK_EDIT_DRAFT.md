# Upwork “Edit proposal” draft (paste-ready)

Since submitting, I’ve upgraded my RAG project to better match your “knowledge brain” requirements:
- **Evidence-only behavior** with a strict refusal mode (returns **“Not in KB yet.”** when retrieval confidence is low)
- **Mandatory citations** in every answer with a `Sources` section (chunk/page references where available)
- **Manifest-driven ingestion** (JSON/YAML) so the index is controlled by a single source of truth (similar to your SOP Index + Docs Directory approach)
- **Audit logging** for every Q/A with the retrieved sources (for review and improvement)
- A small **evaluation harness** (golden set → recall@k) to measure retrieval quality over time

Repo: https://github.com/savinoo/simple-rag-chatbot

Happy to walk through the architecture and show the grounded/citation output on a short demo.
