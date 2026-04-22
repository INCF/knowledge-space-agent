from unittest.mock import patch, AsyncMock

def test_root_endpoint(test_client):
    response = test_client.get("/")
    assert response.status_code == 200
    assert "KnowledgeSpace AI Backend is running" in response.json()["message"]

def test_health_check_endpoint(test_client):
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("main.asyncio.wait_for")
def test_api_health_endpoint(mock_wait_for, test_client):
    mock_wait_for.return_value = True
    
    response = test_client.get("/api/health")
    assert response.status_code == 200
    
    data = response.json()
    assert data["status"] == "healthy"
    assert data["components"]["vector_search"] == "enabled"
    assert data["components"]["llm"] == "enabled"  # enabled via conftest env patch

@patch("main.assistant")
def test_chat_endpoint_success(mock_assistant, test_client):
    mock_assistant.handle_chat = AsyncMock(return_value="Found 3 datasets for rat hippocampus...")
    
    payload = {
        "query": "find rat hippocampus data",
        "session_id": "session_123",
        "reset": False
    }
    
    response = test_client.post("/api/chat", json=payload)
    
    assert response.status_code == 200
    data = response.json()
    assert data["response"] == "Found 3 datasets for rat hippocampus..."
    assert "process_time" in data["metadata"]
    assert data["metadata"]["session_id"] == "session_123"
    assert data["metadata"]["reset"] is False
    
    mock_assistant.handle_chat.assert_called_once_with(
        session_id="session_123",
        query="find rat hippocampus data",
        reset=False
    )

@patch("main.assistant")
def test_session_reset_endpoint(mock_assistant, test_client):
    response = test_client.post("/api/session/reset", json={"session_id": "session_456"})
    
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    
    mock_assistant.reset_session.assert_called_once_with("session_456")
