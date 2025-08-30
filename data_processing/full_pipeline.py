import os
import json
import subprocess
import sys
from pathlib import Path
from typing import Set, List, Dict, Any

from dotenv import load_dotenv
from google.cloud import bigquery
from tqdm import tqdm

def generate_embeddings_in_memory(new_chunks_data):
    """Generate embeddings for new chunks using SentenceTransformers"""
    import os
    import torch
    from sentence_transformers import SentenceTransformer
    
    # Configure CUDA and select device
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = os.getenv('EMBED_MODEL_NAME', "nomic-ai/nomic-embed-text-v1.5")
    
    if device == "cuda":
        gpu_count = torch.cuda.device_count()
        print(f"Found {gpu_count} CUDA-enabled GPU(s). Using GPU acceleration.")
    else:
        print("No CUDA-enabled GPU found. Using CPU for embedding generation.")
    
    print(f"Loading SentenceTransformer model: '{model_name}'")
    model = SentenceTransformer(model_name, trust_remote_code=True, device=device)
    
    texts_to_embed = [chunk["chunk"] for chunk in new_chunks_data]
    chunk_ids = [chunk["datapoint_id"] for chunk in new_chunks_data]
    
    print(f"Generating embeddings for {len(texts_to_embed):,} chunks...")
    
    # Use multi-GPU processing if available
    if device == "cuda" and gpu_count > 1:
        print("Starting multi-GPU processing pool")
        pool = model.start_multi_process_pool()
        
        print(f"Encoding embeddings across {gpu_count} GPUs")
        embeddings = model.encode_multi_process(
            texts_to_embed,
            pool=pool,
            batch_size=32,
        )
        
        model.stop_multi_process_pool(pool)
    else:
        print(f"Encoding embeddings on {device.upper()}")
        embeddings = model.encode(
            texts_to_embed,
            batch_size=32,
            show_progress_bar=True,
            convert_to_tensor=False
        )
    
    print("Embedding generation complete.")
    return [{"id": chunk_ids[i], "embedding": embedding_vector.tolist()} 
            for i, embedding_vector in enumerate(embeddings)]


def upsert_vectorstore_in_memory(embedding_data):
    """Upload embeddings to Vertex AI Matching Engine"""
    from dotenv import load_dotenv
    from google.cloud import aiplatform
    from google.cloud.aiplatform_v1.types import IndexDatapoint
    from tqdm import tqdm
    from vectorstore import get_existing_index, verify_index_deployment
    
    load_dotenv()
    PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    REGION = os.getenv('GCP_REGION')
    INDEX_ENDPOINT_ID = os.getenv('INDEX_ENDPOINT_ID')
    
    aiplatform.init(project=PROJECT_ID, location=REGION)
    idx = get_existing_index()
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{INDEX_ENDPOINT_ID}")
    verify_index_deployment(endpoint, idx)
    
    # Batch upsert embeddings in groups of 1000
    batch = []
    for record in tqdm(embedding_data, desc="Upserting"):
        batch.append(IndexDatapoint(datapoint_id=record["id"], feature_vector=record["embedding"]))
        if len(batch) >= 1000:
            idx.upsert_datapoints(datapoints=batch)
            batch.clear()
    
    if batch:
        idx.upsert_datapoints(datapoints=batch)
    
    return True


def upsert_bigquery_in_memory(chunks_data):
    """Upload chunk metadata to BigQuery"""
    from dotenv import load_dotenv
    from google.cloud import bigquery
    from push_to_bq import ensure_table, merge_rows
    
    load_dotenv()
    PROJECT_ID = os.getenv('GCP_PROJECT_ID')
    LOCATION = os.getenv('BQ_LOCATION', "EU")
    
    bq = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    ensure_table(bq)
    
    # Batch insert chunks in groups of 100k
    buf = []
    for record in chunks_data:
        buf.append(record)
        if len(buf) >= 100000:
            merge_rows(bq, buf)
            buf.clear()
    
    if buf:
        merge_rows(bq, buf)
    
    return True



load_dotenv()

PROJECT_ID = os.getenv('GCP_PROJECT_ID')
DATASET_ID = os.getenv('BQ_DATASET_ID')
TABLE_ID = os.getenv('BQ_TABLE_ID')
LOCATION = os.getenv('BQ_LOCATION')

SCRIPT_DIR = Path(__file__).parent
ROOT_DIR = SCRIPT_DIR.parent
PREPROCESS_DIR = SCRIPT_DIR / "preprocess"
ALL_CHUNKS_FILE = ROOT_DIR / "all_chunks.jsonl"


def run_preprocessing_scripts():
    """Run all preprocessing scripts in the preprocess directory"""
    print("Checking for preprocessing directory")
    if not PREPROCESS_DIR.exists():
        print(f"No preprocessing directory found at {PREPROCESS_DIR}")
        return True
    
    scripts = list(PREPROCESS_DIR.glob("*.py"))
    if not scripts:
        print("No preprocessing scripts found")
        return True
    
    print(f"Found {len(scripts)} preprocessing scripts")
    for script in scripts:
        try:
            print(f"Running {script.name}")
            result = subprocess.run([sys.executable, str(script)], cwd=ROOT_DIR, 
                          capture_output=True, timeout=3600)
            print(f"Script {script.name} completed with return code {result.returncode}")
        except Exception as e:
            print(f"Error running {script.name}: {e}")
    
    return True


def process_chunks():
    """Run process_all_chunks.py to generate all_chunks.jsonl"""
    print("Starting chunk processing")
    try:
        print(f"Running {SCRIPT_DIR / 'process_all_chunks.py'}")
        result = subprocess.run(
            [sys.executable, str(SCRIPT_DIR / "process_all_chunks.py")],
            cwd=ROOT_DIR, capture_output=True, timeout=1800, text=True)
        if result.returncode == 0:
            print("Chunk processing completed successfully")
            return True
        else:
            print(f"Chunk processing failed with return code {result.returncode}")
            print(f"stderr: {result.stderr}")
            return False
    except Exception as e:
        print(f"Error during chunk processing: {e}")
        return False


def get_existing_ids_from_bigquery() -> Set[str]:
    """Get set of existing datapoint IDs from BigQuery"""
    print("Connecting to BigQuery to get existing IDs")
    try:
        bq = bigquery.Client(project=PROJECT_ID, location=LOCATION)
        query = f"SELECT DISTINCT datapoint_id FROM `{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}`"
        print(f"Running query: {query}")
        results = bq.query(query).result()
        existing_ids = {row.datapoint_id for row in results}
        print(f"Found {len(existing_ids)} existing datapoint IDs")
        return existing_ids
    except Exception as e:
        print(f"Error getting existing IDs from BigQuery: {e}")
        print("Assuming no existing data")
        return set()



def identify_new_chunks(existing_ids: Set[str]) -> List[Dict[str, Any]]:
    """Filter out chunks that already exist in BigQuery"""
    print(f"Checking for new chunks in {ALL_CHUNKS_FILE}...")
    if not ALL_CHUNKS_FILE.exists():
        print(f"File {ALL_CHUNKS_FILE} does not exist")
        return []
    
    new_chunks = []
    total_chunks = 0
    with ALL_CHUNKS_FILE.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Checking chunks"):
            if not line.strip():
                continue
            total_chunks += 1
            record = json.loads(line)
            chunk_id = record.get("datapoint_id")
            if chunk_id and chunk_id not in existing_ids:
                new_chunks.append(record)
    
    print(f"Processed {total_chunks} total chunks, found {len(new_chunks)} new chunks")
    
    return new_chunks


def generate_embeddings_for_new_chunks(new_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not new_chunks:
        return []
    try:
        return generate_embeddings_in_memory(new_chunks)
    except:
        return []


def upsert_to_vectorstore(embedding_data: List[Dict[str, Any]]) -> bool:
    if not embedding_data:
        return True
    try:
        return upsert_vectorstore_in_memory(embedding_data)
    except:
        return False


def upsert_to_bigquery(new_chunks: List[Dict[str, Any]]) -> bool:
    if not new_chunks:
        return True
    try:
        return upsert_bigquery_in_memory(new_chunks)
    except:
        return False


def cleanup_local_files():
    """Remove temporary all_chunks.jsonl file"""
    print(f"Removing temporary file {ALL_CHUNKS_FILE}")
    if ALL_CHUNKS_FILE.exists():
        try:
            ALL_CHUNKS_FILE.unlink()
        except:
            pass


def main():
    """Main pipeline: preprocess -> process -> embed -> upsert"""
    print("Starting full pipeline...")
    
    # Step 1: Run preprocessing and chunk processing
    print("Step 1: Running preprocessing scripts")
    preprocess_success = run_preprocessing_scripts()
    print(f"Preprocessing result: {preprocess_success}")
    
    print("Step 1: Processing chunks")
    chunks_success = process_chunks()
    print(f"Chunk processing result: {chunks_success}")
    
    success = preprocess_success and chunks_success
    
    if success:
        print("Step 2: Finding new chunks not already in BigQuery")
        existing_ids = get_existing_ids_from_bigquery()
        new_chunks = identify_new_chunks(existing_ids)
        
        # Skip if no new chunks found
        if not new_chunks:
            print("No new chunks found, pipeline complete")
            return True
        
        print("Step 3: Generating embeddings")
        processed_chunks = generate_embeddings_for_new_chunks(new_chunks)
        
        if not processed_chunks:
            print("Failed to generate embeddings")
            return False
        
        print("Step 4: Upserting to vector store")
        vector_success = upsert_to_vectorstore(processed_chunks)
        print(f"Vector store upsert result: {vector_success}")
        
        print("Step 5: Upserting to BigQuery")
        bq_success = upsert_to_bigquery(new_chunks)
        print(f"BigQuery upsert result: {bq_success}")
        
        success = processed_chunks and vector_success and bq_success
    
    # Step 6: Clean up temporary files on success
    if success:
        print("Step 6: Cleaning up temporary files")
        cleanup_local_files()
        print("Pipeline completed successfully!")
    else:
        print("Pipeline failed")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)