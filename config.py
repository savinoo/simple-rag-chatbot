"""Configuration for RAG Chatbot"""

import os

# Provider
# - openai: requires OPENAI_API_KEY
# - gemini: requires GOOGLE_API_KEY
PROVIDER = os.getenv("PROVIDER", "openai").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Gemini
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_EMBEDDINGS_MODEL = os.getenv("GEMINI_EMBEDDINGS_MODEL", "text-embedding-004")

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
