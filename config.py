"""Configuration for RAG Chatbot"""

import os

# Provider (LLM)
# - openai: requires OPENAI_API_KEY
# - gemini: requires GOOGLE_API_KEY
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# Embeddings provider
# - openai: OpenAI embeddings
# - gemini: Gemini embeddings (often not enabled on some keys/accounts)
# - local: sentence-transformers (offline)
EMBEDDINGS_PROVIDER = os.getenv(
    "EMBEDDINGS_PROVIDER",
    "openai" if PROVIDER == "openai" else "local",
).lower()

LOCAL_EMBEDDINGS_MODEL = os.getenv("LOCAL_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
# Embeddings model: "models/embedding-001" is the most compatible default.
# (Some accounts/regions/APIs may not support text-embedding-004 on v1beta.)
GEMINI_EMBEDDINGS_MODEL = os.getenv("GEMINI_EMBEDDINGS_MODEL", "models/embedding-001")

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", OPENAI_MODEL if PROVIDER == "openai" else GEMINI_MODEL)
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

# RAG settings
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
K_DOCUMENTS = int(os.getenv("K_DOCUMENTS", "5"))

# Retrieval safety
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.35"))

# Optional: manifest-driven ingestion (local)
MANIFEST_PATH = os.getenv("MANIFEST_PATH")

# Logging
LOG_PATH = os.getenv("LOG_PATH", "logs/qa.jsonl")
AUDIT_DB_PATH = os.getenv("AUDIT_DB_PATH", "logs/audit.db")

# Streamlit settings
PAGE_TITLE = os.getenv("PAGE_TITLE", "RAG Chatbot")
PAGE_ICON = os.getenv("PAGE_ICON", "🤖")
