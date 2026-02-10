"""Configuration for RAG Chatbot"""

import os

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-mini")
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

# Streamlit settings
PAGE_TITLE = os.getenv("PAGE_TITLE", "RAG Chatbot")
PAGE_ICON = os.getenv("PAGE_ICON", "🤖")
