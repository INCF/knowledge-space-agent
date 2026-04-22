import os
import sys
import pytest
from unittest.mock import patch, MagicMock

# Add the backend directory to sys.path so tests can import from it
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Mock heavy dependencies that are not installed in the global env
# to allow unit tests to run without them.
mocked_modules = [
    'langgraph',
    'langgraph.graph',
    'torch',
    'google',
    'google.cloud',
    'google.cloud.aiplatform',
    'google.cloud.bigquery',
    'google.genai',
    'google.genai.types',
    'transformers',
    'ks_search_tool'
]

for mod in mocked_modules:
    sys.modules[mod] = MagicMock()

# agents.py imports END from langgraph.graph
sys.modules['langgraph.graph'].END = "END"

@pytest.fixture(autouse=True)
def mock_env_vars():
    env_patcher = patch.dict(os.environ, {
        "GOOGLE_API_KEY": "test-key-123",
        "GCP_PROJECT_ID": "",
        "GEMINI_USE_VERTEX": "false",
        "CORS_ALLOW_ORIGINS": "*",
        "ENVIRONMENT": "test"
    }, clear=False)
    
    env_patcher.start()
    yield
    env_patcher.stop()

@pytest.fixture
def test_client():
    from main import app
    from fastapi.testclient import TestClient
    return TestClient(app)
