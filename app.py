"""
Simple RAG Chatbot with Streamlit
Author: Lucas Lorenzo Savino
"""

import streamlit as st
from rag_pipeline import RAGPipeline
import config

# Page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if 'rag' not in st.session_state:
    st.session_state.rag = RAGPipeline(config.OPENAI_API_KEY)
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'documents_loaded' not in st.session_state:
    st.session_state.documents_loaded = False

# Title
st.title("🤖 RAG Chatbot")
st.markdown("Upload documents and ask questions about them.")

# Sidebar for document upload
with st.sidebar:
    st.header("📚 Knowledge Base")

    # Option A: upload files
    st.subheader("Upload documents")
    uploaded_files = st.file_uploader(
        "Upload documents",
        accept_multiple_files=True,
        type=['pdf', 'txt', 'md']
    )

    if uploaded_files and st.button("Process Uploads"):
        with st.spinner("Processing documents..."):
            try:
                st.session_state.rag.load_documents(uploaded_files)
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded {len(uploaded_files)} document(s)")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Option B: manifest-driven ingestion (local)
    st.subheader("Manifest-driven ingestion (local)")
    st.caption("Use a local manifest JSON like manifest.example.json and set MANIFEST_PATH env var.")
    manifest_path = st.text_input("MANIFEST_PATH", value=(config.MANIFEST_PATH or ""))
    if manifest_path and st.button("Load from Manifest"):
        with st.spinner("Loading manifest + indexing..."):
            try:
                import json
                from pathlib import Path

                m = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
                st.session_state.rag.load_manifest_paths(m["documents"])
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded {len(m['documents'])} document(s) from manifest")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.1)
    k_docs = st.slider("Retrieved Documents", 1, 10, config.K_DOCUMENTS)
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main chat interface
if not st.session_state.documents_loaded:
    st.info("👈 Upload documents in the sidebar to get started.")
else:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = st.session_state.rag.query(
                        prompt,
                        temperature=temperature,
                        k=k_docs
                    )
                    st.markdown(response["answer"])

                    # Show sources (refs) + retrieval debug
                    with st.expander("📄 Sources"):
                        for i, ref in enumerate(response.get("sources", []), 1):
                            st.markdown(f"**Source {i}:** {ref}")

                    with st.expander("🔎 Retrieval (debug)"):
                        st.json(response.get("retrieval", []))

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"]
                    })
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown(
    """
    <div style='text-align: center; color: gray;'>
    Built with LangChain, ChromaDB, and OpenAI | 
    <a href='https://github.com/savinoo/simple-rag-chatbot'>GitHub</a>
    </div>
    """,
    unsafe_allow_html=True
)
