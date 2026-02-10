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

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, UnstructuredMarkdownLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

import config


@dataclass
class RetrievedChunk:
    doc: Document
    score: float
    idx: int  # 1-based index for citations


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


class JSONLLogger:
    """Append-only JSONL logger (no external services)."""

    def __init__(self, path: str = "logs/qa.jsonl"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)

    def log(self, record: dict) -> None:
        record = {"ts": _now_iso(), **record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class RAGPipeline:
    """RAG pipeline for document Q&A."""

    def __init__(self, api_key: str, logger: Optional[JSONLLogger] = None):
        self.api_key = api_key
        self.embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        self.vectorstore: Optional[Chroma] = None

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

        self.llm = ChatOpenAI(
            model_name=config.MODEL_NAME,
            temperature=config.TEMPERATURE,
            openai_api_key=self.api_key,
        )

        self.logger = logger or JSONLLogger(getattr(config, "LOG_PATH", "logs/qa.jsonl"))

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

    def _load_path(self, path: str, source_name: str) -> List[Document]:
        if path.lower().endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif path.lower().endswith(".txt"):
            loader = TextLoader(path, encoding="utf-8")
        elif path.lower().endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        else:
            return []

        loaded = loader.load()
        # Normalize metadata so we can cite it later
        for d in loaded:
            d.metadata = d.metadata or {}
            d.metadata.setdefault("source", source_name)
            if "page" in d.metadata:
                # keep as-is
                pass
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

    def _retrieve(self, question: str, k: int) -> List[RetrievedChunk]:
        if not self.vectorstore:
            raise ValueError("No documents loaded. Please upload documents first.")

        # Chroma returns relevance scores in [0,1] (higher is better)
        pairs: List[Tuple[Document, float]] = self.vectorstore.similarity_search_with_relevance_scores(
            question, k=k
        )

        out: List[RetrievedChunk] = []
        for i, (doc, score) in enumerate(pairs, start=1):
            out.append(RetrievedChunk(doc=doc, score=float(score), idx=i))
        return out

    def query(self, question: str, temperature: Optional[float] = None, k: Optional[int] = None):
        """Query the RAG system.

        Behavior:
        - Retrieve top-k chunks with scores
        - If best score below threshold => "Not in KB yet." (no guessing)
        - Otherwise generate grounded answer with citations
        """

        k = int(k or config.K_DOCUMENTS)
        threshold = float(getattr(config, "RETRIEVAL_THRESHOLD", 0.35))

        retrieved = self._retrieve(question, k=k)
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
            return {"answer": answer, "sources": sources, "retrieval": self._serialize_retrieval(retrieved)}

        # Build context block with stable citation IDs
        ctx_lines = []
        source_map = []
        for r in retrieved:
            md = r.doc.metadata or {}
            src = md.get("source", "unknown")
            chunk = md.get("chunk")
            page = md.get("page")
            ref = f"{src} (chunk {chunk}" + (f", page {page}" if page is not None else "") + ")"
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
