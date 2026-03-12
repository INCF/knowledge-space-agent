"""Tests for response_cache module."""

import os
import sys
import time
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from response_cache import ResponseCache, create_cache


class TestResponseCache:
    def test_disabled_returns_none(self):
        c = ResponseCache(enabled=False)
        c.put("q", "r")
        assert c.get("q") is None

    def test_put_and_get(self):
        c = ResponseCache(enabled=True)
        c.put("What is EEG?", "EEG is…")
        resp, _ = c.get("What is EEG?")
        assert resp == "EEG is…"

    def test_case_insensitive(self):
        c = ResponseCache(enabled=True)
        c.put("  What  IS  EEG?  ", "r")
        assert c.get("what is eeg?") is not None

    def test_miss(self):
        c = ResponseCache(enabled=True)
        assert c.get("unknown") is None

    def test_session_scoped(self):
        """Same query in different sessions should be cached separately."""
        c = ResponseCache(enabled=True)
        c.put("q", "resp-A", session_id="sessA")
        c.put("q", "resp-B", session_id="sessB")
        rA, _ = c.get("q", session_id="sessA")
        rB, _ = c.get("q", session_id="sessB")
        assert rA == "resp-A"
        assert rB == "resp-B"

    def test_max_size_eviction(self):
        c = ResponseCache(enabled=True, max_size=2)
        c.put("q1", "r1")
        c.put("q2", "r2")
        c.put("q3", "r3")
        assert c.size <= 2

    def test_ttl_expiry(self):
        c = ResponseCache(enabled=True, ttl_seconds=0)
        c.put("q", "r")
        time.sleep(0.05)  # Windows timer resolution is ~15ms
        assert c.get("q") is None

    def test_invalidate(self):
        c = ResponseCache(enabled=True)
        c.put("q", "r")
        c.invalidate("q")
        assert c.get("q") is None

    def test_clear(self):
        c = ResponseCache(enabled=True)
        c.put("a", "1")
        c.put("b", "2")
        c.clear()
        assert c.size == 0

    def test_stats(self):
        c = ResponseCache(enabled=True, max_size=64, ttl_seconds=300)
        c.put("q", "r")
        s = c.stats()
        assert s["enabled"] is True
        assert s["size"] == 1


class TestFactory:
    def test_default_disabled(self, monkeypatch):
        monkeypatch.delenv("ENABLE_CACHE", raising=False)
        assert create_cache().enabled is False

    def test_enabled(self, monkeypatch):
        monkeypatch.setenv("ENABLE_CACHE", "true")
        assert create_cache().enabled is True
