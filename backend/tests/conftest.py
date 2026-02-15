import os
import sys
import pytest
from unittest.mock import MagicMock

# Set test environment variables before any imports
os.environ["GOOGLE_API_KEY"] = "test-key-for-testing"
os.environ["RATE_LIMIT"] = "100/minute"
os.environ["CORS_ALLOW_ORIGINS"] = "*"

# Mock heavy dependencies that are not needed for API tests
sys.modules["torch"] = MagicMock()
sys.modules["google.cloud"] = MagicMock()
sys.modules["google.cloud.aiplatform"] = MagicMock()
sys.modules["google.cloud.bigquery"] = MagicMock()
sys.modules["vertexai"] = MagicMock()
sys.modules["vertexai.generative_models"] = MagicMock()
sys.modules["google.generativeai"] = MagicMock()
sys.modules["sentence_transformers"] = MagicMock()

# Mock the retrieval module entirely
mock_retriever = MagicMock()
mock_retriever.Retriever = MagicMock()
sys.modules["retrieval"] = mock_retriever

# Mock the agents module with a fake assistant
mock_assistant = MagicMock()
mock_assistant.NeuroscienceAssistant = MagicMock


class FakeAssistant:
    """Fake assistant that returns predictable responses for testing."""

    async def handle_chat(self, session_id="default", query="", reset=False):
        return f"Test response for: {query}"

    def reset_session(self, session_id):
        pass


mock_agents = MagicMock()
mock_agents.NeuroscienceAssistant = FakeAssistant
sys.modules["agents"] = mock_agents

# Now we can safely import the app
# Add backend to path so main.py can be found
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from main import app
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    with TestClient(app) as c:
        yield c