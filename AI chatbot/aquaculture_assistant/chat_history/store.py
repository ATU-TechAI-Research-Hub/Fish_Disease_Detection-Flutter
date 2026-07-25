"""SQLite-backed chat and prediction context storage."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from pathlib import Path
from typing import Any

from ..models import ChatMessage, PredictionContext, utc_now_iso


class ChatHistoryStore:
    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._lock = threading.RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.database_path,
            timeout=10,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    model TEXT,
                    sources_json TEXT NOT NULL DEFAULT '[]'
                );
                CREATE INDEX IF NOT EXISTS idx_messages_session_time
                    ON messages(session_id, created_at);

                CREATE TABLE IF NOT EXISTS prediction_contexts (
                    session_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS deleted_sessions (
                    session_id TEXT PRIMARY KEY,
                    deleted_at TEXT NOT NULL
                );
                """
            )

    def append(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        sources: list[dict[str, Any]] | None = None,
    ) -> ChatMessage:
        message = ChatMessage(
            id=str(uuid.uuid4()),
            session_id=session_id,
            role=role,
            content=content.strip(),
            created_at=utc_now_iso(),
            model=model,
            sources=tuple(sources or ()),
        )
        with self._lock, self._connect() as connection:
            deleted = connection.execute(
                "SELECT 1 FROM deleted_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if deleted is not None:
                return message
            connection.execute(
                """
                INSERT INTO messages
                    (id, session_id, role, content, created_at, model, sources_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    message.id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.created_at,
                    message.model,
                    json.dumps(list(message.sources), ensure_ascii=False),
                ),
            )
        return message

    def get_history(self, session_id: str, limit: int = 50) -> list[ChatMessage]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT id, session_id, role, content, created_at,
                           model, sources_json
                    FROM messages
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                    LIMIT ?
                )
                ORDER BY created_at ASC
                """,
                (session_id, max(1, min(limit, 200))),
            ).fetchall()
        return [
            ChatMessage(
                id=row["id"],
                session_id=row["session_id"],
                role=row["role"],
                content=row["content"],
                created_at=row["created_at"],
                model=row["model"],
                sources=tuple(json.loads(row["sources_json"] or "[]")),
            )
            for row in rows
        ]

    def clear(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )

    def delete_session(self, session_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                "DELETE FROM messages WHERE session_id = ?", (session_id,)
            )
            connection.execute(
                "DELETE FROM prediction_contexts WHERE session_id = ?",
                (session_id,),
            )
            connection.execute(
                """
                INSERT INTO deleted_sessions (session_id, deleted_at)
                VALUES (?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    deleted_at = excluded.deleted_at
                """,
                (session_id, utc_now_iso()),
            )

    def delete_last_assistant(self, session_id: str) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT id FROM messages
                WHERE session_id = ? AND role = 'assistant'
                ORDER BY created_at DESC LIMIT 1
                """,
                (session_id,),
            ).fetchone()
            if row is None:
                return False
            connection.execute("DELETE FROM messages WHERE id = ?", (row["id"],))
            return True

    def last_user_message(self, session_id: str) -> ChatMessage | None:
        history = self.get_history(session_id, limit=50)
        return next(
            (message for message in reversed(history) if message.role == "user"),
            None,
        )

    def set_prediction(
        self, session_id: str, prediction: PredictionContext
    ) -> None:
        with self._lock, self._connect() as connection:
            deleted = connection.execute(
                "SELECT 1 FROM deleted_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if deleted is not None:
                return
            connection.execute(
                """
                INSERT INTO prediction_contexts
                    (session_id, payload_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    payload_json = excluded.payload_json,
                    updated_at = excluded.updated_at
                """,
                (
                    session_id,
                    json.dumps(prediction.to_dict(), ensure_ascii=False),
                    utc_now_iso(),
                ),
            )

    def get_prediction(self, session_id: str) -> PredictionContext | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT payload_json FROM prediction_contexts
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return PredictionContext.from_dict(json.loads(row["payload_json"]))
