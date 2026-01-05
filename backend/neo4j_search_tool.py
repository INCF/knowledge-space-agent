import os
import logging
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from neo4j import GraphDatabase

logger = logging.getLogger("graph_retrieval")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)

@dataclass
class GraphItem:
    """Standardized output similar to RetrievedItem in retrieval.py"""
    name: str
    definition: str
    relationships: List[str]
    source: str

class GraphRetriever:
    """
    Neo4j-backed retriever for NIFSTD ontology terms.
    
    Environment variables:
      - NEO4J_URI
      - NEO4J_USER
      - NEO4J_PASSWORD
    """
    def __init__(self):
        self.uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = os.getenv("NEO4J_USER", "neo4j")
        self.password = os.getenv("NEO4J_PASSWORD", "password")
        
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            self.driver.verify_connectivity()
            logger.info(f"Connected to Neo4j at {self.uri}")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            self.driver = None

    def close(self):
        if self.driver:
            self.driver.close()

    def search_ontology(self, term: str) -> List[GraphItem]:
        """
        Finds a term in the graph and returns its definition + parent/child links.
        """
        if not self.driver:
            return []

        query = """
        MATCH (n)
        WHERE toLower(n.label) CONTAINS toLower($term)
        OPTIONAL MATCH (n)-[r]->(related)
        RETURN n.label as name, n.definition as def, collect(type(r) + " -> " + related.label) as rels
        LIMIT 5
        """
        
        results = []
        try:
            with self.driver.session() as session:
                records = session.run(query, term=term)
                for record in records:
                    results.append(GraphItem(
                        name=record["name"],
                        definition=record.get("def", "No definition found"),
                        relationships=record["rels"],
                        source="NIFSTD Ontology"
                    ))
        except Exception as e:
            logger.error(f"Graph search failed: {e}")
            
        return results

if __name__ == "__main__":
    gr = GraphRetriever()
    items = gr.search_ontology("hippocampus")
    for item in items:
        print(f"Found: {item.name} | Rels: {item.relationships}")
    gr.close()