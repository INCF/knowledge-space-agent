from unittest.mock import patch, AsyncMock, MagicMock

from agents import (
    _is_more_query,
    QueryIntent,
    fuse_results,
    AgentState,
    NeuroscienceAssistant
)

def test_is_more_query():
    assert _is_more_query("next 10") == 10
    assert _is_more_query("show 5") == 5
    assert _is_more_query("more 20") == 20
    
    assert _is_more_query("more") is None
    assert _is_more_query("continue") is None
    
    assert _is_more_query("find rat electrophysiology") is None
    assert _is_more_query("") is None

def test_fuse_results():
    state: AgentState = {
        "session_id": "test_session",
        "query": "rat data",
        "history": [],
        "keywords": [],
        "effective_query": "rat data",
        "intents": [],
        "ks_results": [{"_id": "doc_common", "_score": 10.0}, {"_id": "doc_ks_only", "_score": 5.0}],
        "vector_results": [{"id": "doc_common", "similarity": 0.8}, {"id": "doc_vec_only", "similarity": 0.9}],
        "final_results": [],
        "all_results": [],
        "start_number": 1,
        "previous_text": "",
        "final_response": "",
    }
    
    new_state = fuse_results(state)
    all_res = new_state["all_results"]
    
    assert len(all_res) == 3
    
    # doc_common score: vector (0.8 * 0.6 = 0.48) + ks (10.0 * 0.4 = 4.0) = 4.48
    # doc_vec_only score: vector (0.9 * 0.6 = 0.54) + ks (0) = 0.54
    # doc_ks_only score: vector (0) + ks (5.0 * 0.4 = 2.0) = 2.0
    doc_ids = [res.get("id") or res.get("_id") for res in all_res]
    assert doc_ids == ["doc_common", "doc_ks_only", "doc_vec_only"]

def test_neuroscience_assistant_reset():
    assistant = NeuroscienceAssistant()
    
    assistant.chat_history["session_123"] = ["User: Hello", "Assistant: Hi"]
    assistant.session_memory["session_123"] = {"page": 1}
    
    assistant.reset_session("session_123")
    
    assert "session_123" not in assistant.chat_history
    assert "session_123" not in assistant.session_memory
