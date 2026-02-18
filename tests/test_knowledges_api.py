import sys
import os
import pytest

# Ensure Python can find the backend folder
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.knowledgespace_api import format_datasets_list, list_datasources

def test_format_datasets_list_with_data():
    """Test that the formatter correctly handles valid JSON data"""
    mock_data = {
        "total_count": 1,
        "current_page": 0,
        "results": [{
            "id": "123",
            "dc": {"title": "Test Dataset", "description": "A test description"}
        }]
    }
    result = format_datasets_list(mock_data)
    
    assert "Total Datasets Found: 1" in result
    assert "Test Dataset" in result
    assert "A test description" in result

def test_format_datasets_list_empty():
    """Test that the formatter handles empty results gracefully"""
    empty_data = {"results": [], "total_count": 0}
    result = format_datasets_list(empty_data)
    # The code returns the header even if results are empty
    assert "Total Datasets Found: 0" in result
    
def test_list_datasources_mocked(mocker):
    """Test API call without actually hitting the internet"""
    mock_get = mocker.patch('requests.get')
    mock_get.return_value.json.return_value = [{"name": "Source 1", "id": "S1"}]
    mock_get.return_value.status_code = 200

    result = list_datasources()
    
    assert len(result) == 1
    assert result[0]["name"] == "Source 1"
    mock_get.assert_called_once()