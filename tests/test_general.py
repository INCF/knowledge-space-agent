def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()


def test_api_health_endpoint(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    assert "status" in response.json()
