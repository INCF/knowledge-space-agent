"""Tests for health check endpoints."""


class TestRootEndpoint:

    def test_root_returns_200(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_root_contains_message(self, client):
        response = client.get("/")
        data = response.json()
        assert "message" in data
        assert "running" in data["message"].lower()

    def test_root_contains_version(self, client):
        response = client.get("/")
        data = response.json()
        assert "version" in data
        assert data["version"] == "2.0.0"


class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_status_is_healthy(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_health_contains_timestamp(self, client):
        response = client.get("/health")
        data = response.json()
        assert "timestamp" in data

    def test_health_contains_service_name(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["service"] == "knowledge-space-agent-backend"

    def test_health_contains_version(self, client):
        response = client.get("/health")
        data = response.json()
        assert data["version"] == "2.0.0"


class TestApiHealthEndpoint:

    def test_api_health_returns_200(self, client):
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_api_health_status_is_healthy(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert data["status"] == "healthy"

    def test_api_health_contains_components(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "components" in data

    def test_api_health_components_have_expected_keys(self, client):
        response = client.get("/api/health")
        components = response.json()["components"]
        assert "vector_search" in components
        assert "llm" in components
        assert "keyword_search" in components

    def test_api_health_keyword_search_always_enabled(self, client):
        response = client.get("/api/health")
        components = response.json()["components"]
        assert components["keyword_search"] == "enabled"

    def test_api_health_contains_timestamp(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert "timestamp" in data

    def test_api_health_contains_version(self, client):
        response = client.get("/api/health")
        data = response.json()
        assert data["version"] == "2.0.0"