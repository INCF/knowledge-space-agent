# from unittest.mock import patch


# def test_chat_endpoint_success(client):
#     with patch("backend.main.assistant") as mock_assistant:
#         mock_assistant.chat.return_value = "Mocked response"

#         response = client.post(
#             "/api/chat",
#             json={"query": "Hello"}
#         )

#         assert response.status_code == 200
#         assert response.json()["response"] == "Mocked response"


# def test_chat_endpoint_validation_error(client):
#     response = client.post(
#         "/api/chat",
#         json={}
#     )

#     assert response.status_code == 422


from unittest.mock import AsyncMock, patch


def test_chat_endpoint_success(client):
    with patch("backend.main.assistant") as mock_assistant:
        mock_assistant.handle_chat = AsyncMock(return_value="Mocked response")

        response = client.post("/api/chat", json={"query": "Hello"})

        assert response.status_code == 200
        assert response.json()["response"] == "Mocked response"


def test_chat_endpoint_validation_error(client):
    response = client.post("/api/chat", json={})

    assert response.status_code == 422
