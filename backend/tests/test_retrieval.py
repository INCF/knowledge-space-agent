from unittest.mock import patch, MagicMock

from retrieval import VertexRetriever, get_retriever
from local_retriever import LocalRetriever

def test_local_retriever():
    retriever = LocalRetriever()
    assert retriever.is_enabled is True
    assert retriever.search("test query") == []

@patch("retrieval.os.getenv")
@patch("retrieval.aiplatform")
@patch("retrieval.bigquery")
@patch("retrieval.AutoTokenizer")
@patch("retrieval.AutoModel")
@patch("retrieval.torch.cuda.is_available", return_value=False)
def test_vertex_retriever_initialization(mock_cuda, mock_model, mock_tokenizer, mock_bq, mock_ai, mock_getenv):
    def mock_env(key, default=""):
        mapping = {
            "GCP_PROJECT_ID": "test-project",
            "GCP_REGION": "us-central1",
            "INDEX_ENDPOINT_ID_FULL": "projects/123/locations/us-central1/indexEndpoints/456",
            "DEPLOYED_INDEX_ID": "test_index",
        }
        return mapping.get(key, default)
    
    mock_getenv.side_effect = mock_env
    
    # prevent actual model downloads during test
    mock_model.from_pretrained.return_value.eval.return_value.to.return_value = MagicMock()
    
    retriever = VertexRetriever()
    
    assert retriever.is_enabled is True
    mock_ai.init.assert_called_once_with(project="test-project", location="us-central1")
    mock_bq.Client.assert_called_once_with(project="test-project")
    mock_tokenizer.from_pretrained.assert_called_once()
    mock_model.from_pretrained.assert_called_once()

def test_get_retriever_fallback():
    # fallback happens because we stripped GCP env vars in conftest
    retriever = get_retriever()
    assert isinstance(retriever, LocalRetriever)
