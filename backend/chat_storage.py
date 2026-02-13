"""
Modular chat history storage layer for the Knowledge Space Agent.

Provides a simple abstract interface with two implementations:
  - InMemoryChatStorage  : dict-backed, matches original behaviour (default)
  - SQLiteChatStorage    : file-backed via stdlib sqlite3, zero extra deps

Default behaviour is unchanged.  Persistence is opt-in via the
ENABLE_PERSISTENCE env var.

    from chat_storage import create_storage
    storage = create_storage()   # reads env vars
"""

import os
import json
import sqlite3
import threading
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class ChatStorage(ABC):
    """Minimal interface that any chat-history backend must satisfy."""

    @abstractmethod
    def get_history(self, session_id: str) -> List[str]:
        ...

    @abstractmethod
    def append(self, session_id: str, entry: str) -> None:
        ...

    @abstractmethod
    def set_history(self, session_id: str, history: List[str]) -> None:
        ...

    @abstractmethod
    def clear(self, session_id: str) -> None:
        ...

    @abstractmethod
    def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        ...

    @abstractmethod
    def set_session_memory(self, session_id: str, memory: Dict[str, Any]) -> None:
        ...

    @abstractmethod
    def clear_session_memory(self, session_id: str) -> None:
        ...

    @abstractmethod
    def list_sessions(self) -> List[str]:
        ...


# ---------------------------------------------------------------------------
# In-memory (default — identical to original behaviour)
# ---------------------------------------------------------------------------

class InMemoryChatStorage(ChatStorage):

    def __init__(self) -> None:
        self._history: Dict[str, List[str]] = {}
        self._memory: Dict[str, Dict[str, Any]] = {}

    def get_history(self, session_id: str) -> List[str]:
        return list(self._history.get(session_id, []))

    def append(self, session_id: str, entry: str) -> None:
        self._history.setdefault(session_id, []).append(entry)

    def set_history(self, session_id: str, history: List[str]) -> None:
        self._history[session_id] = list(history)

    def clear(self, session_id: str) -> None:
        self._history.pop(session_id, None)

    def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        return dict(self._memory.get(session_id, {}))

    def set_session_memory(self, session_id: str, memory: Dict[str, Any]) -> None:
        self._memory[session_id] = dict(memory)

    def clear_session_memory(self, session_id: str) -> None:
        self._memory.pop(session_id, None)

    def list_sessions(self) -> List[str]:
        return list(set(list(self._history) + list(self._memory)))


# ---------------------------------------------------------------------------
# SQLite (opt-in persistence)
# ---------------------------------------------------------------------------

_DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "chat_history.db"
)


class SQLiteChatStorage(ChatStorage):
    """
    SQLite-backed persistent chat storage.

    Uses stdlib ``sqlite3`` — zero extra dependencies.  Thread-safe via a
    lock with ``check_same_thread=False``.
    """

    def __init__(self, db_path: Optional[str] = None) -> None:
        self._db_path = db_path or os.getenv("CHAT_DB_PATH", _DEFAULT_DB_PATH)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            self._db_path, timeout=5, check_same_thread=False
        )
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._init_db()

    def _init_db(self) -> None:
        with self._lock:
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS chat_history (
                    id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    entry      TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            self._conn.execute(
                """CREATE TABLE IF NOT EXISTS session_memory (
                    session_id TEXT PRIMARY KEY,
                    data       TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )"""
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chat_session "
                "ON chat_history(session_id)"
            )
            self._conn.commit()

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            self._conn.close()

    # -- interface -----------------------------------------------------------

    def get_history(self, session_id: str) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT entry FROM chat_history WHERE session_id=? ORDER BY id",
                (session_id,),
            ).fetchall()
            return [r[0] for r in rows]

    def append(self, session_id: str, entry: str) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat_history(session_id, entry, created_at) VALUES(?,?,?)",
                (session_id, entry, now),
            )
            self._conn.commit()

    def set_history(self, session_id: str, history: List[str]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            self._conn.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
            self._conn.executemany(
                "INSERT INTO chat_history(session_id, entry, created_at) VALUES(?,?,?)",
                [(session_id, e, now) for e in history],
            )
            self._conn.commit()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM chat_history WHERE session_id=?", (session_id,))
            self._conn.commit()

    def get_session_memory(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            row = self._conn.execute(
                "SELECT data FROM session_memory WHERE session_id=?", (session_id,)
            ).fetchone()
            if row:
                try:
                    return json.loads(row[0])
                except (json.JSONDecodeError, TypeError):
                    return {}
            return {}

    def set_session_memory(self, session_id: str, memory: Dict[str, Any]) -> None:
        now = datetime.now(timezone.utc).isoformat()
        # Exclude bulky all_results from DB; store a count instead.
        serializable = {k: v for k, v in memory.items() if k != "all_results"}
        if "all_results" in memory:
            serializable["all_results_count"] = len(memory["all_results"])
        data = json.dumps(serializable, default=str)
        with self._lock:
            self._conn.execute(
                """INSERT INTO session_memory(session_id, data, updated_at)
                   VALUES(?,?,?)
                   ON CONFLICT(session_id)
                   DO UPDATE SET data=excluded.data, updated_at=excluded.updated_at""",
                (session_id, data, now),
            )
            self._conn.commit()

    def clear_session_memory(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM session_memory WHERE session_id=?", (session_id,))
            self._conn.commit()

    def list_sessions(self) -> List[str]:
        with self._lock:
            rows = self._conn.execute(
                """SELECT DISTINCT session_id FROM (
                       SELECT session_id FROM chat_history
                       UNION
                       SELECT session_id FROM session_memory
                   )"""
            ).fetchall()
            return [r[0] for r in rows]


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_storage(backend: Optional[str] = None, **kwargs: Any) -> ChatStorage:
    """
    Create a ChatStorage from configuration.

    Reads ``CHAT_STORAGE_BACKEND`` or ``ENABLE_PERSISTENCE`` from env.
    Defaults to in-memory (original behaviour).  Falls back to in-memory
    gracefully if SQLite initialisation fails.
    """
    if backend is None:
        backend = os.getenv("CHAT_STORAGE_BACKEND", "").strip().lower()
        if not backend:
            flag = os.getenv("ENABLE_PERSISTENCE", "").strip().lower()
            backend = "sqlite" if flag in {"1", "true", "yes"} else "memory"

    if backend == "sqlite":
        try:
            storage = SQLiteChatStorage(**kwargs)
            print("[chat_storage] Using SQLite persistent storage")
            return storage
        except Exception as exc:
            print(f"[chat_storage] SQLite init failed ({exc}), falling back to in-memory")
            return InMemoryChatStorage()

    print("[chat_storage] Using in-memory storage (default)")
    return InMemoryChatStorage()
