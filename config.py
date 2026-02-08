"""
Configuration for RAG Chatbot
"""

import os

# OpenAI API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# Model settings
MODEL_NAME = "gpt-3.5-turbo"
TEMPERATURE = 0.7

# RAG settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
K_DOCUMENTS = 4

# Streamlit settings
PAGE_TITLE = "RAG Chatbot"
PAGE_ICON = "🤖"
