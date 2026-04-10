import pytest
from rrf import reciprocal_rank_fusion, extract_doc_id

def test_extract_doc_id():
    assert extract_doc_id({"id": "123"}) == "123"
    assert extract_doc_id({"_id": "456"}) == "456"
    assert extract_doc_id({"id": "123", "_id": "456"}) == "123"  # Prefers 'id'
    assert extract_doc_id({}) == ""

def test_rrf_single_list():
    list1 = [{"id": "A"}, {"id": "B"}, {"id": "C"}]
    fused = reciprocal_rank_fusion([list1], k=60, top_k=10)
    
    assert len(fused) == 3
    assert fused[0]["id"] == "A"
    assert fused[1]["id"] == "B"
    assert fused[2]["id"] == "C"
    
    # Check score math: A=1/61, B=1/62, C=1/63
    assert fused[0]["rrf_score"] == 1 / 61
    assert fused[1]["rrf_score"] == 1 / 62
    assert fused[2]["rrf_score"] == 1 / 63

def test_rrf_two_lists_same_order():
    list1 = [{"id": "A"}, {"id": "B"}]
    list2 = [{"_id": "A"}, {"_id": "B"}] # Note list2 uses _id
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=10)
    
    assert len(fused) == 2
    assert fused[0]["id"] == "A" # Source dict comes from list1 first
    assert fused[1]["id"] == "B"
    
    # A is rank 1 in both: 1/61 + 1/61
    assert fused[0]["rrf_score"] == (1/61) + (1/61)

def test_rrf_boosts_overlap():
    # A is in both lists but ranked lower. B is rank 1 in list1 only. C is rank 1 in list2 only.
    list1 = [{"id": "B"}, {"id": "A"}, {"id": "X"}]
    list2 = [{"id": "C"}, {"id": "A"}, {"id": "Y"}]
    
    fused = reciprocal_rank_fusion([list1, list2], k=60, top_k=10)
    
    weights = {doc["id"]: doc["rrf_score"] for doc in fused}
    
    # A: rank 2 + rank 2 = 1/62 + 1/62 = 0.032258
    # B: rank 1 + none   = 1/61 + 0    = 0.016393
    # C: rank 1 + none   = 1/61 + 0    = 0.016393
    
    assert fused[0]["id"] == "A"
    assert weights["A"] > weights["B"]
    assert weights["A"] > weights["C"]

def test_rrf_empty_lists():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []
    
    list1 = [{"id": "A"}]
    # Fuses one empty list and one populated list
    fused = reciprocal_rank_fusion([list1, []], k=60)
    assert len(fused) == 1
    assert fused[0]["id"] == "A"

def test_rrf_top_k_truncates():
    list1 = [{"id": str(i)} for i in range(100)]
    fused = reciprocal_rank_fusion([list1], k=60, top_k=5)
    assert len(fused) == 5
    assert fused[-1]["id"] == "4" # Indices 0, 1, 2, 3, 4

def test_rrf_id_fallback():
    # If a document doesn't have id or _id, the function uses a hash fallback.
    # While relying on title_guess is weak, this ensures no crash.
    list1 = [{"title_guess": "Unique Title"}, {"title_guess": "Another Title"}]
    fused = reciprocal_rank_fusion([list1])
    assert len(fused) == 2
    assert fused[0].get("rrf_score") is not None
