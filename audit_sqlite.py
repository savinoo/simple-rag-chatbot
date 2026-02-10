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
            con.execute(
                "CREATE INDEX IF NOT EXISTS idx_qa_logs_ts ON qa_logs(ts)"
            )

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
