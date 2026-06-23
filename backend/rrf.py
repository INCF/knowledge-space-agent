import logging
from typing import List, Dict, Any, Set

logger = logging.getLogger("rrf")

def extract_doc_id(result: Dict[str, Any]) -> str:
    """
    Safely extract a unique document ID from a search result dictionary.
    Handles differences between Keyword Search (KS) and Vector Search formats.
    """
    return str(result.get("id") or result.get("_id") or "")

def reciprocal_rank_fusion(
    ranked_lists: List[List[Dict[str, Any]]],
    k: int = 60,
    top_k: int = 15
) -> List[Dict[str, Any]]:
    """
    Combines multiple ranked lists of documents into a single ranked list using
    Reciprocal Rank Fusion (RRF).

    Formula: RRF_score(d) = sum(1 / (k + rank_i(d)))
    where `rank_i(d)` is the 1-based index (rank) of document `d` in list `i`.

    Args:
        ranked_lists: A list of lists, where each inner list contains document dicts
                      ordered by their original search score (highest first).
        k: The smoothing constant (default: 60, standard from literature).
        top_k: The number of top fused results to return.

    Returns:
        A single fused list of document dictionaries, ordered by RRF score descending.
        Each dictionary will have an added 'rrf_score' field and an updated 'final_score'
        field for compatibility with the rest of the application.
    """
    # 1. Initialize RRF scores for all unique document IDs
    rrf_scores: Dict[str, float] = {}
    
    # We also keep a mapping of ID -> original document dict
    # so we can reconstruct the final list (we use the first occurrence we find)
    doc_map: Dict[str, Dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for idx, doc in enumerate(ranked_list):
            doc_id = extract_doc_id(doc)
            
            # Skip if we couldn't resolve an ID (should theoretically not happen, but safe)
            if not doc_id:
                # Generate a weak fallback ID based on content hash or title context if needed,
                # but for KnowledgeSpace, id or _id should always exist.
                doc_id = str(hash(doc.get("title_guess", "unknown")))
                
            rank = idx + 1  # RRF uses 1-based ranks
            
            # Add the reciprocal rank score for this document
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            
            # Store the underlying doc if we haven't seen it yet
            if doc_id not in doc_map:
                # Make a shallow copy to avoid mutating the original deeply
                doc_map[doc_id] = dict(doc)

    # 2. Sort documents by their accumulated RRF score descending
    sorted_keys = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    sorted_doc_ids: List[str] = list(sorted_keys)

    # 3. Construct the final fused list
    fused_results: List[Dict[str, Any]] = []
    
    for doc_id in sorted_doc_ids[:top_k]:
        doc = doc_map[doc_id]
        score = rrf_scores[doc_id]
        
        # Add tracking fields to the document
        doc["rrf_score"] = score
        # Maintain backward compatibility with agents.py expectations
        doc["final_score"] = score
        
        fused_results.append(doc)

    logger.debug(f"Combined {len(ranked_lists)} lists into {len(fused_results)} results.")
    return fused_results
