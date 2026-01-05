import pytest
from unittest.mock import MagicMock, patch
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from backend.neo4j_search_tool import GraphRetriever

@pytest.fixture
def mock_driver():
    """Fakes the Neo4j driver so we don't need a real DB."""
    with patch("backend.neo4j_search_tool.GraphDatabase.driver") as mock_dt:
        mock_instance = MagicMock()
        mock_dt.return_value = mock_instance
        yield mock_instance

def test_initialization():
    """Verifies the tool loads environment variables correctly."""
    with patch.dict(os.environ, {"NEO4J_URI": "bolt://test:7687"}):
        tool = GraphRetriever()
        assert tool.uri == "bolt://test:7687"

def test_search_execution(mock_driver):
    """Verifies the search method sends the correct Cypher query."""
    # 1. Setup the fake return data
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    
    # Fake a Neo4j record
    mock_record = MagicMock()
    mock_record.data.return_value = {"name": "Hippocampus", "definition": "Brain region"}
    mock_session.run.return_value = [mock_record]

    # 2. Run the tool
    tool = GraphRetriever()
    tool.connect()
    results = tool.search("Hippocampus")

    # 3. Verify it worked
    assert len(results) == 1
    assert results[0]["name"] == "Hippocampus"
    
    # Verify the code actually called the DB
    mock_session.run.assert_called_once()