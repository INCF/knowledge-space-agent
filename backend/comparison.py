from graph_retrieval import GraphRetriever

def mock_vector_search(query):
    """
    Simulates what their current system returns.
    Vector search looks for keywords. If I search 'Ataxia', 
    it finds the disease but misses the brain region connection if not explicitly stated.
    """
    if "ataxia" in query.lower():
        return [
            "Ataxia is a degenerative disease of the nervous system.",
            "Symptoms include lack of voluntary coordination of muscle movements."
        ]
    return []

def main():
    
    query = "What brain region does Ataxia affect?"
    print(f"User Query: '{query}'\n")

    # (Vector Only)
    print(f"Vector Search Results:")
    vector_results = mock_vector_search(query)
    for res in vector_results:
        print(f" - {res}")

    # (Graph Retrieval)
    print(f"Graph Context Retrieval:")
    gr = GraphRetriever()
    results = gr.search_ontology("Ataxia") 
    
    if not results:
        print("No graph results found. (Did you run seed_graph.py?)")
    else:
        for item in results:
            print(f" - Entity: {item.name}")
            print(f" - Definition: {item.definition}")
            print(f" - Relationships: {item.relationships}")
            
    gr.close()

if __name__ == "__main__":
    main()