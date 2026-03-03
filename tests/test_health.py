import os
import sys
from unittest.mock import MagicMock

# Add backend folder to path
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "backend"))
)

# Mock assistant BEFORE importing main
import builtins
import types

mock_agents = types.ModuleType("agents")
mock_agents.NeuroscienceAssistant = MagicMock
sys.modules["agents"] = mock_agents

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    assert "KnowledgeSpace AI Backend is running" in response.json()["message"]


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_api_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200

    data = response.json()
    assert data["status"] == "healthy"
    assert "components" in data
    assert "version" in data


def test_chat_endpoint_success(monkeypatch):
    async def mock_handle_chat(session_id, query, reset):
        return "Mocked response"

    # Patch the assistant instance inside main
    from main import assistant

    monkeypatch.setattr(assistant, "handle_chat", mock_handle_chat)

    response = client.post(
        "/api/chat", json={"query": "What is neuroscience?", "session_id": "test123"}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Mocked response"
    assert "metadata" in data


def test_chat_validation_error():
    response = client.post(
        "/api/chat", json={"session_id": "abc"}  # Missing required 'query'
    )

    assert response.status_code == 422
