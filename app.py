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
    st.caption("Supports JSON (manifest.example.json) and YAML (manifest.example.yaml).")
    manifest_path = st.text_input("MANIFEST_PATH", value=(config.MANIFEST_PATH or ""))
    if manifest_path and st.button("Load from Manifest"):
        with st.spinner("Loading manifest + indexing..."):
            try:
                from manifest_loader import load_manifest

                docs = load_manifest(manifest_path)
                st.session_state.rag.load_manifest_docs(docs)
                st.session_state.documents_loaded = True
                st.success(f"✅ Loaded {len(docs)} document(s) from manifest")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    st.divider()
    
    # Settings
    st.header("⚙️ Settings")
    temperature = st.slider("Temperature", 0.0, 1.0, config.TEMPERATURE, 0.1)
    k_docs = st.slider("Retrieved Documents", 1, 10, config.K_DOCUMENTS)

    st.header("🔐 Access (demo)")
    role = st.selectbox(
        "Role (used to filter retrieval when manifest provides allowed_roles)",
        options=["(all)", "cs", "warehouse", "qc", "management"],
        index=0,
    )
    st.session_state["role"] = role
    
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main interface (tabs)
chat_tab, admin_tab = st.tabs(["💬 Chat", "🛠️ Admin"])

with chat_tab:
    if not st.session_state.documents_loaded:
        st.info("👈 Upload documents in the sidebar to get started.")
    else:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask a question about your documents..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag.query(
                            prompt,
                            temperature=temperature,
                            k=k_docs,
                            role=st.session_state.get("role"),
                        )
                        st.markdown(response["answer"])

                        with st.expander("📄 Sources"):
                            for i, ref in enumerate(response.get("sources", []), 1):
                                st.markdown(f"**Source {i}:** {ref}")

                        with st.expander("🔎 Retrieval (debug)"):
                            st.json(response.get("retrieval", []))

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response["answer"],
                        })
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")

with admin_tab:
    st.subheader("Audit log (SQLite)")
    st.caption("Shows the latest Q/A entries with status, confidence, and sources.")

    try:
        recent = st.session_state.rag.audit.recent(limit=25)
        st.dataframe(recent, use_container_width=True)

        selected = st.number_input("View answer by log id", min_value=0, value=0, step=1)
        if selected:
            ans = st.session_state.rag.audit.get_answer(int(selected))
            if ans:
                st.markdown("### Answer")
                st.code(ans)
            else:
                st.info("No answer found for that id.")
    except Exception as e:
        st.error(f"Audit log error: {e}")

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
