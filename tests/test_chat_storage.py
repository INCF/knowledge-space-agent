"""Tests for chat_storage module."""

import os
import sys
import tempfile
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from chat_storage import InMemoryChatStorage, SQLiteChatStorage, create_storage


class TestInMemoryChatStorage:
    def setup_method(self):
        self.s = InMemoryChatStorage()

    def test_empty_history(self):
        assert self.s.get_history("s1") == []

    def test_append_and_get(self):
        self.s.append("s1", "User: hi")
        self.s.append("s1", "Assistant: hello")
        assert self.s.get_history("s1") == ["User: hi", "Assistant: hello"]

    def test_sessions_isolated(self):
        self.s.append("a", "User: x")
        self.s.append("b", "User: y")
        assert self.s.get_history("a") == ["User: x"]
        assert self.s.get_history("b") == ["User: y"]

    def test_set_history_replaces(self):
        self.s.append("s1", "old")
        self.s.set_history("s1", ["new"])
        assert self.s.get_history("s1") == ["new"]

    def test_clear(self):
        self.s.append("s1", "x")
        self.s.clear("s1")
        assert self.s.get_history("s1") == []

    def test_session_memory_round_trip(self):
        self.s.set_session_memory("s1", {"page": 2})
        assert self.s.get_session_memory("s1")["page"] == 2

    def test_clear_session_memory(self):
        self.s.set_session_memory("s1", {"a": 1})
        self.s.clear_session_memory("s1")
        assert self.s.get_session_memory("s1") == {}

    def test_list_sessions(self):
        self.s.append("s1", "x")
        self.s.set_session_memory("s2", {})
        assert set(self.s.list_sessions()) == {"s1", "s2"}


class TestSQLiteChatStorage:
    def setup_method(self):
        self.tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmp.close()
        self.s = SQLiteChatStorage(db_path=self.tmp.name)

    def teardown_method(self):
        self.s.close()
        try:
            os.unlink(self.tmp.name)
        except OSError:
            pass

    def test_append_and_get(self):
        self.s.append("s1", "User: hi")
        self.s.append("s1", "Assistant: hello")
        assert self.s.get_history("s1") == ["User: hi", "Assistant: hello"]

    def test_persistence_across_instances(self):
        self.s.append("s1", "User: persisted")
        self.s.set_session_memory("s1", {"k": "v"})
        fresh = SQLiteChatStorage(db_path=self.tmp.name)
        assert fresh.get_history("s1") == ["User: persisted"]
        assert fresh.get_session_memory("s1")["k"] == "v"
        fresh.close()

    def test_set_history_replaces(self):
        self.s.append("s1", "old")
        self.s.set_history("s1", ["new"])
        assert self.s.get_history("s1") == ["new"]

    def test_clear(self):
        self.s.append("s1", "x")
        self.s.clear("s1")
        assert self.s.get_history("s1") == []

    def test_session_memory_upsert(self):
        self.s.set_session_memory("s1", {"page": 1})
        self.s.set_session_memory("s1", {"page": 2})
        assert self.s.get_session_memory("s1")["page"] == 2

    def test_list_sessions(self):
        self.s.append("a", "x")
        self.s.set_session_memory("b", {})
        assert set(self.s.list_sessions()) == {"a", "b"}


class TestFactory:
    def test_default_is_memory(self, monkeypatch):
        monkeypatch.delenv("ENABLE_PERSISTENCE", raising=False)
        monkeypatch.delenv("CHAT_STORAGE_BACKEND", raising=False)
        assert isinstance(create_storage(), InMemoryChatStorage)

    def test_persistence_gives_sqlite(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ENABLE_PERSISTENCE", "true")
        monkeypatch.delenv("CHAT_STORAGE_BACKEND", raising=False)
        monkeypatch.setenv("CHAT_DB_PATH", str(tmp_path / "t.db"))
        assert isinstance(create_storage(), SQLiteChatStorage)

    def test_graceful_fallback_on_bad_path(self, monkeypatch):
        """If SQLite can't create the DB, factory falls back to InMemory."""
        monkeypatch.setenv("CHAT_DB_PATH", "/nonexistent/dir/impossible.db")
        storage = create_storage(backend="sqlite")
        assert isinstance(storage, InMemoryChatStorage)
