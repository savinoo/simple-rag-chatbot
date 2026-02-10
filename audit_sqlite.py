"""SQLite-backed audit logging.

Stores every Q/A with sources used so the system is reviewable and improvable.
Local-first: no external services.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


@dataclass
class QALogRecord:
    ts: str
    question: str
    status: str  # answered|not_in_kb|error
    best_score: float
    k: int
    sources: list[str]
    answer: str


class SQLiteAudit:
    def __init__(self, path: str = "logs/audit.db"):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path)
        con.row_factory = sqlite3.Row
        return con

    def _init_db(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS qa_logs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  question TEXT NOT NULL,
                  status TEXT NOT NULL,
                  best_score REAL NOT NULL,
                  k INTEGER NOT NULL,
                  sources_json TEXT NOT NULL,
                  answer TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_qa_logs_ts ON qa_logs(ts)")

    def ensure_sync_tables(self) -> None:
        with self._connect() as con:
            con.execute(
                """
                CREATE TABLE IF NOT EXISTS sync_runs (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  ts TEXT NOT NULL,
                  manifest_path TEXT NOT NULL,
                  docs_total INTEGER NOT NULL,
                  docs_indexed INTEGER NOT NULL,
                  docs_failed INTEGER NOT NULL,
                  changed_docs INTEGER NOT NULL,
                  errors_json TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_sync_runs_ts ON sync_runs(ts)")

            con.execute(
                """
                CREATE TABLE IF NOT EXISTS doc_state (
                  doc_id TEXT PRIMARY KEY,
                  path TEXT NOT NULL,
                  content_hash TEXT NOT NULL,
                  indexed_at TEXT NOT NULL
                )
                """
            )
            con.execute("CREATE INDEX IF NOT EXISTS idx_doc_state_path ON doc_state(path)")

    def log(self, rec: QALogRecord) -> None:
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO qa_logs (ts, question, status, best_score, k, sources_json, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.ts,
                    rec.question,
                    rec.status,
                    float(rec.best_score),
                    int(rec.k),
                    json.dumps(rec.sources, ensure_ascii=False),
                    rec.answer,
                ),
            )

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute(
                "SELECT id, ts, status, best_score, k, question, sources_json FROM qa_logs ORDER BY id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            out.append(
                {
                    "id": r["id"],
                    "ts": r["ts"],
                    "status": r["status"],
                    "best_score": r["best_score"],
                    "k": r["k"],
                    "question": r["question"],
                    "sources": json.loads(r["sources_json"] or "[]"),
                }
            )
        return out

    def get_answer(self, row_id: int) -> Optional[str]:
        with self._connect() as con:
            row = con.execute(
                "SELECT answer FROM qa_logs WHERE id = ?",
                (int(row_id),),
            ).fetchone()
        return row["answer"] if row else None

    # -----------------------------
    # Sync tracking (MVP)
    # -----------------------------

    def log_sync_run(
        self,
        manifest_path: str,
        docs_total: int,
        docs_indexed: int,
        docs_failed: int,
        errors: list[dict],
        changed_docs: int,
    ) -> None:
        self.ensure_sync_tables()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO sync_runs (ts, manifest_path, docs_total, docs_indexed, docs_failed, changed_docs, errors_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now_iso(),
                    manifest_path,
                    int(docs_total),
                    int(docs_indexed),
                    int(docs_failed),
                    int(changed_docs),
                    json.dumps(errors, ensure_ascii=False),
                ),
            )

    def upsert_doc_state(self, doc_id: str, path: str, content_hash: str) -> None:
        self.ensure_sync_tables()
        with self._connect() as con:
            con.execute(
                """
                INSERT INTO doc_state (doc_id, path, content_hash, indexed_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  path=excluded.path,
                  content_hash=excluded.content_hash,
                  indexed_at=excluded.indexed_at
                """,
                (doc_id, path, content_hash, now_iso()),
            )

    def get_doc_state(self, doc_id: str) -> Optional[dict[str, Any]]:
        self.ensure_sync_tables()
        with self._connect() as con:
            row = con.execute(
                "SELECT doc_id, path, content_hash, indexed_at FROM doc_state WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "doc_id": row["doc_id"],
            "path": row["path"],
            "content_hash": row["content_hash"],
            "indexed_at": row["indexed_at"],
        }
