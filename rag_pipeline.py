"""
RAG Pipeline Implementation

Upgraded to support:
- Manifest-driven ingestion (local files)
- Answer-only-from-sources behavior with a "Not in KB yet." fallback
- Citations (source + chunk reference) in every answer
- Lightweight structured logging (JSONL)

Author: Lucas Lorenzo Savino
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from markdown_loader import load_markdown_with_sections
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Optional Gemini support
try:
    from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
except Exception:  # pragma: no cover
    ChatGoogleGenerativeAI = None  # type: ignore
    GoogleGenerativeAIEmbeddings = None  # type: ignore

import config
from audit_sqlite import QALogRecord, SQLiteAudit, now_iso


@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    idx: int  # 1-based index for citations


# (moved to audit_sqlite.now_iso)


class JSONLLogger:
    """Append-only JSONL logger (no external services)."""

    def __init__(self, path: str = "logs/qa.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"ts": now_iso(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class RAGPipeline:
    """RAG pipeline for document Q&A."""

    def __init__(self, logger: Optional[JSONLLogger] = None):
        self.provider = getattr(config, "PROVIDER", "openai").lower()
        self.emb_provider = getattr(config, "EMBEDDINGS_PROVIDER", "openai").lower()

        # Embeddings
        if self.emb_provider == "local":
            self.embeddings = HuggingFaceEmbeddings(
                model_name=getattr(config, "LOCAL_EMBEDDINGS_MODEL", "all-MiniLM-L6-v2")
            )
        elif self.emb_provider == "gemini":
            if not getattr(config, "GOOGLE_API_KEY", ""):
                raise ValueError("GOOGLE_API_KEY is required when EMBEDDINGS_PROVIDER=gemini")
            if GoogleGenerativeAIEmbeddings is None:
                raise ImportError(
                    "Gemini dependencies missing. Install: langchain-google-genai google-generativeai"
                )
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model=getattr(config, "GEMINI_EMBEDDINGS_MODEL", "models/embedding-001"),
                google_api_key=getattr(config, "GOOGLE_API_KEY"),
            )
        else:
            # openai
            if not getattr(config, "OPENAI_API_KEY", ""):
                raise ValueError("OPENAI_API_KEY is required when EMBEDDINGS_PROVIDER=openai")
            self.embeddings = OpenAIEmbeddings(openai_api_key=getattr(config, "OPENAI_API_KEY"))

        self.vectorstore: Optional[Chroma] = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

        if self.provider == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model=getattr(config, "MODEL_NAME", "gemini-1.5-flash"),
                temperature=config.TEMPERATURE,
                google_api_key=getattr(config, "GOOGLE_API_KEY"),
            )
        else:
            self.llm = ChatOpenAI(
                model_name=getattr(config, "MODEL_NAME", "gpt-4o-mini"),
                temperature=config.TEMPERATURE,
                openai_api_key=getattr(config, "OPENAI_API_KEY"),
            )

        self.logger = logger or JSONLLogger(getattr(config, "LOG_PATH", "logs/qa.jsonl"))
        self.audit = SQLiteAudit(getattr(config, "AUDIT_DB_PATH", "logs/audit.db"))

    # ----------------------
    # Ingestion
    # ----------------------

    def load_documents(self, uploaded_files) -> None:
        """Load and process Streamlit UploadedFile objects."""
        docs: List[Document] = []

        for uploaded_file in uploaded_files:
            suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name

            try:
                docs.extend(self._load_path(tmp_path, source_name=uploaded_file.name))
            finally:
                os.unlink(tmp_path)

        self._index_documents(docs)

    def load_manifest_paths(self, paths: Iterable[str]) -> None:
        """Load and index documents from local file paths (manifest-driven ingestion)."""
        docs: List[Document] = []
        for p in paths:
            p = os.path.expanduser(p)
            docs.extend(self._load_path(p, source_name=os.path.basename(p)))
        self._index_documents(docs)

    def load_manifest_docs(self, manifest_docs: Iterable[object]) -> None:
        """Load and index documents from manifest entries that include metadata.

        Expected fields (duck-typed): path, id, title, tags, allowed_roles.
        """
        docs: List[Document] = []
        for md in manifest_docs:
            path = os.path.expanduser(getattr(md, "path"))
            source_name = os.path.basename(path)
            loaded = self._load_path(path, source_name=source_name)
            for d in loaded:
                d.metadata = d.metadata or {}
                d.metadata["doc_id"] = getattr(md, "id", None)
                d.metadata["title"] = getattr(md, "title", None)
                d.metadata["tags"] = list(getattr(md, "tags", []) or [])
                d.metadata["allowed_roles"] = list(getattr(md, "allowed_roles", []) or [])
            docs.extend(loaded)
        self._index_documents(docs)

    def _load_path(self, path: str, source_name: str) -> List[Document]:
        if path.lower().endswith(".md"):
            loaded = load_markdown_with_sections(path, source_name=source_name)
            return loaded

        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        else:
            return []

        loaded = loader.load()
        # Normalize metadata so we can cite it later
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", source_name)
        return loaded

    def _index_documents(self, documents: List[Document]) -> None:
        if not documents:
            raise ValueError("No supported documents found.")

        splits = self.text_splitter.split_documents(documents)

        # Attach chunk index per source for more stable citations
        per_source_counts = {}
        for d in splits:
            src = (d.metadata or {}).get("source", "unknown")
            per_source_counts[src] = per_source_counts.get(src, 0) + 1
            d.metadata["chunk"] = per_source_counts[src]

        self.vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=self.embeddings,
            persist_directory=None,  # in-memory
        )

    # ----------------------
    # Retrieval + Generation
    # ----------------------

    def _retrieve(self, question: str, k: int, role: Optional[str] = None) -> List[RetrievedChunk]:
        if not self.vectorstore:
            raise ValueError("No documents loaded. Please upload documents first.")

        # Retrieve extra then filter (Chroma metadata filters vary by backend)
        raw_k = max(k * 3, k)

        pairs: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=raw_k
        )

        def allowed(doc: Document) -> bool:
            if not role or role == "(all)":
                return True
            roles = (doc.metadata or {}).get("allowed_roles") or []
            # If no roles are set, treat as public within the app
            if not roles:
                return True
            return role in roles

        filtered = [(d, s) for (d, s) in pairs if allowed(d)][:k]

        out: List[RetrievedChunk] = []
        for i, (doc, score) in enumerate(filtered, start=1):
            out.append(RetrievedChunk(doc=doc, score=float(score), idx=i))
        return out

    def query(
        self,
        question: str,
        temperature: Optional[float] = None,
        k: Optional[int] = None,
        role: Optional[str] = None,
    ):
        """Query the RAG system.

        Behavior:
        - Retrieve top-k chunks with scores
        - If best score below threshold => "Not in KB yet." (no guessing)
        - Otherwise generate grounded answer with citations
        """

        k = int(k or config.K_DOCUMENTS)
        threshold = float(getattr(config, "RETRIEVAL_THRESHOLD", 0.35))

        retrieved = self._retrieve(question, k=k, role=role)
        best_score = retrieved[0].score if retrieved else 0.0

        if not retrieved or best_score < threshold:
            answer = "Not in KB yet. Please add the relevant SOP/policy document to the knowledge base."
            sources = []
            self.logger.log(
                {
                    "question": question,
                    "best_score": best_score,
                    "k": k,
                    "status": "not_in_kb",
                    "sources": sources,
                    "answer": answer,
                }
            )
            self.audit.log(
                QALogRecord(
                    ts=now_iso(),
                    question=question,
                    status="not_in_kb",
                    best_score=best_score,
                    k=k,
                    sources=sources,
                    answer=answer,
                )
            )
            return {"answer": answer, "sources": sources, "retrieval": self._serialize_retrieval(retrieved)}

        # Build context block with stable citation IDs
        ctx_lines = []
        source_map = []
        for r in retrieved:
            md = r.doc.metadata or {}
            src = md.get("source", "unknown")
            title = md.get("title")
            section = md.get("section_path")
            chunk = md.get("chunk")
            page = md.get("page")

            label = title or src
            parts = [f"chunk {chunk}"] if chunk is not None else []
            if page is not None:
                parts.append(f"page {page}")
            if section:
                parts.append(f"section {section}")

            ref = f"{label} ({', '.join(parts)})" if parts else label
            source_map.append({"id": r.idx, "ref": ref, "metadata": md})
            ctx_lines.append(f"[S{r.idx}] {ref}\n{r.doc.page_content}")

        context = "\n\n".join(ctx_lines)

        system = (
            "You are an internal assistant that MUST answer strictly using the provided SOURCES. "
            "Do not use outside knowledge. If the answer is not supported by the sources, reply exactly: 'Not in KB yet.'\n\n"
            "When you answer, include citations like [S1], [S2] next to each claim. "
            "At the end, include a 'Sources' section listing each used source id with its ref." 
        )

        user = f"Question: {question}\n\nSOURCES:\n{context}\n\nAnswer:" 

        if temperature is not None:
            self.llm.temperature = float(temperature)

        msg = self.llm.invoke([{"role": "system", "content": system}, {"role": "user", "content": user}])
        answer_text = msg.content.strip()

        # If model didn't cite anything, enforce safe behavior
        if "[S" not in answer_text and "Not in KB yet" not in answer_text:
            answer_text = "Not in KB yet. Please add the relevant SOP/policy document to the knowledge base."

        used_ids = sorted({int(x[2:]) for x in _extract_citation_tokens(answer_text)})
        used_sources = [s for s in source_map if s["id"] in used_ids] if used_ids else source_map

        # Append sources section if missing
        if "Sources" not in answer_text:
            lines = ["\n\nSources:"]
            for s in used_sources:
                lines.append(f"- [S{s['id']}] {s['ref']}")
            answer_text = answer_text + "\n" + "\n".join(lines)

        self.logger.log(
            {
                "question": question,
                "best_score": best_score,
                "k": k,
                "status": "answered",
                "sources": [s["ref"] for s in used_sources],
                "answer": answer_text,
            }
        )
        self.audit.log(
            QALogRecord(
                ts=now_iso(),
                question=question,
                status="answered",
                best_score=best_score,
                k=k,
                sources=[s["ref"] for s in used_sources],
                answer=answer_text,
            )
        )

        return {
            "answer": answer_text,
            "sources": [s["ref"] for s in used_sources],
            "retrieval": self._serialize_retrieval(retrieved),
        }

    def _serialize_retrieval(self, retrieved: List[RetrievedChunk]) -> List[dict]:
        out = []
        for r in retrieved:
            md = r.doc.metadata or {}
            out.append(
                {
                    "id": r.idx,
                    "score": r.score,
                    "source": md.get("source"),
                    "chunk": md.get("chunk"),
                    "page": md.get("page"),
                }
            )
        return out


def _extract_citation_tokens(text: str) -> List[str]:
    # naive extraction of [S1], [S2]...
    out = []
    i = 0
    while True:
        i = text.find("[S", i)
        if i == -1:
            break
        j = text.find("]", i)
        if j == -1:
            break
        token = text[i + 1 : j]
        if token.startswith("S") and token[1:].isdigit():
            out.append("[" + token + "]")
        i = j + 1
    return out
