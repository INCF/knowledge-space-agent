import os
import json
import re
from dataclasses import dataclass
from typing import List, Optional

# 1. DYNAMIC PATH LOADING (Finds json in the same folder)
def load_local_data():
    """
    Loads local_knowledge.json from the SAME directory as this script.
    """
    # Get the directory where THIS script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build the full path to the json file
    json_path = os.path.join(current_dir, "local_knowledge.json")
    
    print(f" Loading local data from: {json_path}") 
    
    if not os.path.exists(json_path):
        print(" Error: File not found.")
        return []

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f" Error reading JSON: {e}")
        return []

# 2. OFFLINE SEARCH LOGIC (BM25)
@dataclass
class RetrievedItem:
    id: str
    title_guess: str
    content: str
    similarity: float

def local_offline_search(query: str):
    data = load_local_data()
    if not data:
        return []

    print(f" Searching for: '{query}'...")
    
    query_words = set(re.findall(r'\w+', query.lower()))
    results = []

    for item in data:
        score = 0.0
        title_text = item.get("title", "")
        content_text = item.get("content", "")
        
        # Simple scoring logic
        for word in query_words:
            if word in title_text.lower():
                score += 1.5
            if word in content_text.lower():
                score += 1.0
        
        if score > 0:
            results.append(RetrievedItem(
                id=str(item.get("id")),
                title_guess=title_text,
                content=content_text,
                similarity=float(score)
            ))

    # Sort results
    results.sort(key=lambda x: x.similarity, reverse=True)
    return results[:5]

# 3.TEST RUNNER 
if __name__ == "__main__":
    print("  STARTING OFFLINE AGENT TEST  ")
    
    # Test Query
    test_query = "Alzheimer" 
    hits = local_offline_search(test_query)
    
    if not hits:
        print(" No results found. Check your json file content.")
    
    for hit in hits:
        print(f"\n Found: {hit.title_guess} (Score: {hit.similarity})")
        print(f"   Context: {hit.content[:100]}...")