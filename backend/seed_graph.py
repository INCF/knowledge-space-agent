import os
from neo4j import GraphDatabase

URI = "bolt://localhost:7687"
AUTH = ("neo4j", "password")  
def seed_data():
    driver = GraphDatabase.driver(URI, auth=AUTH)
    
    with driver.session() as session:
        print("1. Clearing old data...")
        session.run("MATCH (n) DETACH DELETE n")
        
        print("2. Seeding new NIFSTD ontology data...")
        query = """
        CREATE (brain:NIFSTD_Term {label: "Brain", id: "UBERON:0000955", definition: "The central organ of the nervous system."})
        CREATE (hind:NIFSTD_Term {label: "Hindbrain", id: "UBERON:0002028", definition: "The posterior part of the brain."})
        CREATE (cere:NIFSTD_Term {label: "Cerebellum", id: "UBERON:0002037", definition: "Region of the brain that plays an important role in motor control."})
        
        // Create Relationships
        CREATE (cere)-[:PART_OF]->(hind)
        CREATE (hind)-[:PART_OF]->(brain)

        // Create a Disease that links to the specific region
        CREATE (atak:Disease {label: "Ataxia", definition: "A degenerative disease of the nervous system."})
        CREATE (atak)-[:AFFECTS]->(cere)
        
        RETURN count(brain) as nodes_created
        """
        
        result = session.run(query)
        record = result.single()
        print(f"Database seeded! Nodes created: {record['nodes_created'] if record else 0}")
    
    driver.close()

if __name__ == "__main__":
    seed_data()