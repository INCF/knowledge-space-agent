"""
Lightweight in-memory response cache for the Knowledge Space Agent.

Reduces redundant LLM calls by caching responses for identical queries.
Opt-in via ``ENABLE_CACHE=true``.  Zero extra dependencies.

    from response_cache import create_cache
    cache = create_cache()          # reads env vars
    hit   = cache.get("what is EEG?")
    cache.put("what is EEG?", "EEG is …")
"""

import os
import time
import hashlib
import threading
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple


class ResponseCache:
    """Exact-match response cache with TTL expiry and LRU eviction."""

    def __init__(
        self,
        max_size: int = 128,
        ttl_seconds: int = 3600,
        enabled: bool = True,
    ) -> None:
        self._enabled = enabled
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        # Ordered by insertion/access time (LRU order).
        # key -> (response, metadata, monotonic_timestamp)
        self._store: OrderedDict[str, Tuple[str, Dict[str, Any], float]] = OrderedDict()

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _normalise(query: str) -> str:
        return " ".join(query.lower().split())

    @staticmethod
    def _key(session_id: str, query: str) -> str:
        raw = f"{session_id}:{query}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, (_, _, ts) in self._store.items() if now - ts > self._ttl]
        for k in expired:
            del self._store[k]

    def _evict_oldest(self) -> None:
        while len(self._store) >= self._max_size:
            self._store.popitem(last=False)  # O(1) — pop oldest

    # -- public API ----------------------------------------------------------

    def get(self, query: str, session_id: str = "") -> Optional[Tuple[str, Dict[str, Any]]]:
        if not self._enabled:
            return None
        key = self._key(session_id, self._normalise(query))
        with self._lock:
            self._evict_expired()
            entry = self._store.get(key)
            if entry is None:
                return None
            resp, meta, _ = entry
            # Move to end (most-recently-used) and refresh timestamp
            self._store.move_to_end(key)
            self._store[key] = (resp, meta, time.monotonic())
            return (resp, meta)

    def put(self, query: str, response: str, session_id: str = "", metadata: Optional[Dict[str, Any]] = None) -> None:
        if not self._enabled:
            return
        key = self._key(session_id, self._normalise(query))
        with self._lock:
            self._evict_expired()
            self._evict_oldest()
            self._store[key] = (response, metadata or {}, time.monotonic())

    def invalidate(self, query: str, session_id: str = "") -> None:
        if not self._enabled:
            return
        key = self._key(session_id, self._normalise(query))
        with self._lock:
            self._store.pop(key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    @property
    def size(self) -> int:
        with self._lock:
            return len(self._store)

    @property
    def enabled(self) -> bool:
        return self._enabled

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "enabled": self._enabled,
                "size": len(self._store),
                "max_size": self._max_size,
                "ttl_seconds": self._ttl,
            }


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_cache(**kwargs: Any) -> ResponseCache:
    """Create a ResponseCache from env configuration."""
    enabled = os.getenv("ENABLE_CACHE", "").strip().lower() in {"1", "true", "yes"}
    max_size = int(os.getenv("CACHE_MAX_SIZE", "128"))
    ttl = int(os.getenv("CACHE_TTL_SECONDS", "3600"))

    if enabled:
        print(f"[response_cache] Enabled (max_size={max_size}, ttl={ttl}s)")
    else:
        print("[response_cache] Disabled (default)")

    return ResponseCache(max_size=max_size, ttl_seconds=ttl, enabled=enabled, **kwargs)
