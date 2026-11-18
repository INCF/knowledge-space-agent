"""
Regression and concurrency tests for thread-safe GenAI client initialization in backend/agents.py.
Validates Issue #82: _GENAI_CLIENT singleton thread safety during concurrent cold starts.
"""

import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, patch

import pytest

import agents
from agents import _get_genai_client, _reset_genai_client


@pytest.fixture(autouse=True)
def reset_singleton_state():
    """Ensure singleton client state is reset before and after every test."""
    _reset_genai_client()
    yield
    _reset_genai_client()


def test_concurrent_cold_start_singleton():
    """
    Test A: Concurrent cold-start test.
    Synchronizes 20 worker threads via threading.Barrier to simulate simultaneous cold-start requests.
    Asserts:
    1. genai.Client constructor is called exactly once.
    2. All 20 threads receive the exact same singleton client instance.
    """
    num_threads = 20
    barrier = threading.Barrier(num_threads)
    results = [None] * num_threads

    mock_client_instance = MagicMock()
    mock_client_instance.close = MagicMock()

    def delayed_client_init(*args, **kwargs):
        # Controlled delay to increase the probability of overlapping initialization attempts
        time.sleep(0.02)
        return mock_client_instance

    with patch.object(agents, "_use_vertex", return_value=False), \
         patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
         patch("agents.genai.Client", side_effect=delayed_client_init) as mock_client_cls:

        def worker(thread_idx: int):
            # Wait for all threads to be ready so they hit _get_genai_client simultaneously
            barrier.wait()
            results[thread_idx] = _get_genai_client()

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Constructor must be called exactly once
        assert mock_client_cls.call_count == 1, (
            f"Expected exactly 1 client initialization, but was called {mock_client_cls.call_count} times."
        )

        # All threads must receive the exact same client instance
        assert all(res is mock_client_instance for res in results), (
            "Not all threads received the same singleton client instance."
        )


def test_warm_start_singleton():
    """
    Test B: Warm-start test.
    Verifies that after the client is initialized, subsequent calls return the cached
    instance immediately without re-invoking the constructor.
    """
    mock_client_instance = MagicMock()

    with patch.object(agents, "_use_vertex", return_value=False), \
         patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
         patch("agents.genai.Client", return_value=mock_client_instance) as mock_client_cls:

        first_client = _get_genai_client()
        assert mock_client_cls.call_count == 1
        assert first_client is mock_client_instance

        # Subsequent warm calls
        for _ in range(5):
            client = _get_genai_client()
            assert client is mock_client_instance

        # Constructor should still only have been called once
        assert mock_client_cls.call_count == 1


def test_vertex_ai_initialization():
    """
    Test C: Vertex AI initialization test.
    Verifies that when _use_vertex() is True, Vertex credentials and region configurations
    are correctly forwarded to genai.Client.
    """
    mock_client_instance = MagicMock()

    with patch.object(agents, "_use_vertex", return_value=True), \
         patch.object(agents, "_ensure_google_creds_for_vertex") as mock_ensure_creds, \
         patch.dict(os.environ, {"GCP_PROJECT_ID": "my-vertex-project", "GCP_REGION": "us-central1"}), \
         patch("agents.genai.Client", return_value=mock_client_instance) as mock_client_cls:

        client = _get_genai_client()

        assert client is mock_client_instance
        mock_ensure_creds.assert_called_once()
        mock_client_cls.assert_called_once_with(
            vertexai=True,
            project="my-vertex-project",
            location="us-central1",
        )


def test_api_key_initialization():
    """
    Test D: API-Key initialization test.
    Verifies that when _use_vertex() is False, the API key is passed correctly
    and the resulting client is cached.
    """
    mock_client_instance = MagicMock()

    with patch.object(agents, "_use_vertex", return_value=False), \
         patch.dict(os.environ, {"GOOGLE_API_KEY": "ai-studio-api-key-xyz"}), \
         patch("agents.genai.Client", return_value=mock_client_instance) as mock_client_cls:

        client = _get_genai_client()

        assert client is mock_client_instance
        mock_client_cls.assert_called_once_with(api_key="ai-studio-api-key-xyz")


def test_initialization_failure_and_retry():
    """
    Test E: Initialization failure and retry test.
    Ensures that if client construction raises an exception:
    1. The exception propagates out.
    2. _GENAI_CLIENT remains None (not in a corrupted/half-baked state).
    3. A subsequent request is able to retry and successfully initialize the client.
    """
    mock_success_instance = MagicMock()

    with patch.object(agents, "_use_vertex", return_value=False), \
         patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
         patch("agents.genai.Client", side_effect=[RuntimeError("Transient auth failure"), mock_success_instance]) as mock_client_cls:

        # First attempt fails
        with pytest.raises(RuntimeError, match="Transient auth failure"):
            _get_genai_client()

        # Singleton must remain None
        assert agents._GENAI_CLIENT is None

        # Second attempt succeeds
        client = _get_genai_client()
        assert client is mock_success_instance
        assert agents._GENAI_CLIENT is mock_success_instance
        assert mock_client_cls.call_count == 2


def test_reset_genai_client():
    """
    Test F: Reset test.
    Verifies that _reset_genai_client():
    1. Calls client.close() if available to prevent resource leaks.
    2. Sets _GENAI_CLIENT back to None.
    3. Allows subsequent calls to initialize a fresh client instance.
    """
    mock_instance_1 = MagicMock()
    mock_instance_1.close = MagicMock()

    mock_instance_2 = MagicMock()
    mock_instance_2.close = MagicMock()

    with patch.object(agents, "_use_vertex", return_value=False), \
         patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}), \
         patch("agents.genai.Client", side_effect=[mock_instance_1, mock_instance_2]):

        first_client = _get_genai_client()
        assert first_client is mock_instance_1

        # Reset singleton
        _reset_genai_client()
        mock_instance_1.close.assert_called_once()
        assert agents._GENAI_CLIENT is None

        # Next call initializes a fresh instance
        second_client = _get_genai_client()
        assert second_client is mock_instance_2
