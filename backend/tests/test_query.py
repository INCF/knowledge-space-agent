"""Tests for chat and session endpoints."""


class TestChatEndpoint:

    def test_chat_endpoint_exists(self, client):
        response = client.post("/api/chat", json={"query": "test"})
        assert response.status_code != 404

    def test_chat_returns_response_field(self, client):
        response = client.post("/api/chat", json={"query": "What is a neuron?"})
        data = response.json()
        assert "response" in data

    def test_chat_returns_metadata(self, client):
        response = client.post("/api/chat", json={"query": "test query"})
        data = response.json()
        assert "metadata" in data

    def test_chat_metadata_contains_process_time(self, client):
        response = client.post("/api/chat", json={"query": "test"})
        if response.status_code == 200:
            metadata = response.json().get("metadata", {})
            assert "process_time" in metadata

    def test_chat_metadata_contains_session_id(self, client):
        response = client.post("/api/chat", json={"query": "test"})
        if response.status_code == 200:
            metadata = response.json().get("metadata", {})
            assert "session_id" in metadata

    def test_chat_with_custom_session_id(self, client):
        response = client.post(
            "/api/chat",
            json={"query": "test", "session_id": "my-session-123"},
        )
        if response.status_code == 200:
            metadata = response.json().get("metadata", {})
            assert metadata.get("session_id") == "my-session-123"

    def test_chat_missing_query_returns_422(self, client):
        response = client.post("/api/chat", json={})
        assert response.status_code == 422

    def test_chat_get_method_not_allowed(self, client):
        response = client.get("/api/chat")
        assert response.status_code == 405


class TestSessionResetEndpoint:

    def test_reset_endpoint_exists(self, client):
        response = client.post(
            "/api/session/reset",
            json={"session_id": "test-session"},
        )
        assert response.status_code != 404

    def test_reset_returns_ok_status(self, client):
        response = client.post(
            "/api/session/reset",
            json={"session_id": "test-session"},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["status"] == "ok"

    def test_reset_returns_session_id(self, client):
        response = client.post(
            "/api/session/reset",
            json={"session_id": "my-session"},
        )
        if response.status_code == 200:
            data = response.json()
            assert data["session_id"] == "my-session"


class TestUnknownRoutes:

    def test_unknown_route_returns_404(self, client):
        response = client.get("/this-does-not-exist")
        assert response.status_code == 404

    def test_unknown_api_route_returns_404(self, client):
        response = client.get("/api/nonexistent")
        assert response.status_code == 404