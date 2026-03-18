from ks_search_tool import smart_knowledge_search
import json

def test():
    print("\n" + "="*80)
    print("QUERY: 'hippocampus' (Expanded: 'hippocampus OR CA1 OR CA3 OR dentate gyrus')")
    print("="*80 + "\n")
    
    # Run the search
    res = smart_knowledge_search("hippocampus", top_k=4)
    results = res.get("combined_results", [])
    
    for i, r in enumerate(results):
        title = str(r.get("title", "Unknown Title"))
        if len(title) > 60: title = title[:57] + "..."
        
        score = float(r.get("_score", 0.0))
        mult = float(r.get("_rerank_multiplier", 1.0))
        meta = r.get("metadata", {}) or {}
        
        year = meta.get("publication_year") or meta.get("year") or "N/A"
        cits = meta.get("citations") or meta.get("citation_count") or 0
        src = r.get("datasource_name") or meta.get("source") or "Unknown"
        
        print(f"[{i+1}] {title}")
        print(f"    * Final Score : {score:.4f}  (Multiplier Applied: {mult:.4f}x)")
        print(f"    + Metadata    : Year={year} | Citations={cits} | Source={src}")
        print("-" * 80)
        
if __name__ == "__main__":
    test()
