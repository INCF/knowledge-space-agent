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

def test_search_ontology(mock_driver):
    """Verifies the search method returns GraphItem objects."""
    mock_session = MagicMock()
    mock_driver.session.return_value.__enter__.return_value = mock_session
    mock_record = MagicMock()
    
    data = {"name": "Hippocampus", "def": "Brain region", "rels": []}
    mock_record.__getitem__.side_effect = data.__getitem__
    mock_record.get.side_effect = data.get
    mock_session.run.return_value = [mock_record]

    tool = GraphRetriever()
    results = tool.search_ontology("Hippocampus")
    assert results[0].name == "Hippocampus"
    assert results[0].definition == "Brain region"
    mock_session.run.assert_called_once()