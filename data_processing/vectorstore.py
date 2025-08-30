import json
import time
import random
import os
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from tqdm import tqdm
from google.api_core import exceptions
from google.cloud import aiplatform, aiplatform_v1
from google.cloud.aiplatform_v1.types import IndexDatapoint

# Load environment variables
load_dotenv()

# CONFIG

PROJECT_ID         = os.getenv('GCP_PROJECT_ID')
PROJECT_NUMBER     = os.getenv('GCP_PROJECT_NUMBER')
REGION             = os.getenv('GCP_REGION')

# These values should be set when index and endpoint already exist
INDEX_DISPLAY_NAME = "kschunks-index-nomic-768"
INDEX_ENDPOINT_ID  = os.getenv('INDEX_ENDPOINT_ID')
DEPLOYED_INDEX_ID  = os.getenv('DEPLOYED_INDEX_ID')

LOCAL_EMBEDDINGS_PATH = Path("embeddings1.jsonl")
UPSERT_BATCH_SIZE     = 1000

CHECKPOINT_FILE = Path(".upsert_ckpt.txt")
MAX_RETRIES     = 8
BACKOFF_BASE    = 2.0  

#  INDEX RETRIEVAL
def get_existing_index():
    """Get the existing index. Index and endpoint must already be created and deployed."""
    print(f"Looking for existing index named '{INDEX_DISPLAY_NAME}'...")
    indexes = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{INDEX_DISPLAY_NAME}"'
    )
    if not indexes:
        raise RuntimeError(f"Index '{INDEX_DISPLAY_NAME}' not found. Please create and deploy the index first.")
    
    idx = indexes[0]
    print(f"Found existing index: {idx.resource_name}")
    return idx


def verify_index_deployment(endpoint_obj, index_obj):
    """Verify that the index is properly deployed to the endpoint."""
    deployed_list = getattr(endpoint_obj, "deployed_indexes", None)
    if deployed_list is None:
        deployed_list = endpoint_obj.gca_resource.deployed_indexes

    for di in deployed_list:
        if di.index == index_obj.resource_name or di.id == DEPLOYED_INDEX_ID:
            print(f"Index is deployed (id={di.id}).")
            return True
    
    raise RuntimeError(f"Index is not deployed to endpoint. Please deploy the index first.")


#  UPSERT HELPER
def _safe_upsert(index_obj, batch, batch_num):
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            index_obj.upsert_datapoints(datapoints=batch)
            return
        except (exceptions.ServiceUnavailable,
                exceptions.DeadlineExceeded,
                exceptions.InternalServerError) as e:
            wait = BACKOFF_BASE ** attempt + random.uniform(0, 1.5)
            print(f"\nWARNING: Batch {batch_num} failed ({e.__class__.__name__}). "
                  f"Retry {attempt}/{MAX_RETRIES} in {wait:.1f}s")
            time.sleep(wait)
    raise RuntimeError(f"Upsert batch {batch_num} failed after {MAX_RETRIES} retries.")


def _load_checkpoint() -> int:
    if CHECKPOINT_FILE.exists():
        try:
            return int(CHECKPOINT_FILE.read_text().strip())
        except Exception:
            pass
    return 0


def _save_checkpoint(n_processed: int):
    CHECKPOINT_FILE.write_text(str(n_processed))


def stream_upload_vectors(index_obj):
    with LOCAL_EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        total_lines = sum(1 for ln in f if ln.strip())

    start_line = _load_checkpoint()
    print(f"Total lines: {total_lines:,}. Resuming from line {start_line:,}.")

    batch: List[IndexDatapoint] = []
    processed = start_line
    batch_num = 0

    with LOCAL_EMBEDDINGS_PATH.open("r", encoding="utf-8") as f, \
         tqdm(total=total_lines, initial=start_line, desc="Upserting", unit="dp") as pbar:

        for i, line in enumerate(f):
            if not line.strip():
                continue
            if i < start_line:
                continue

            rec = json.loads(line)
            dp = IndexDatapoint(
                datapoint_id=rec["id"],
                feature_vector=rec["embedding"],
            )
            batch.append(dp)
            processed += 1

            if len(batch) == UPSERT_BATCH_SIZE:
                batch_num += 1
                _safe_upsert(index_obj, batch, batch_num)
                batch.clear()
                _save_checkpoint(processed)
                pbar.update(UPSERT_BATCH_SIZE)

        if batch:
            batch_num += 1
            _safe_upsert(index_obj, batch, batch_num)
            _save_checkpoint(processed)
            pbar.update(len(batch))

    print("All vectors upserted.")
    try:
        CHECKPOINT_FILE.unlink()
    except OSError:
        pass


#  MAIN
    """Upload vectors to existing deployed index."""
    aiplatform.init(project=PROJECT_ID, location=REGION)
    
    # Get existing index and endpoint
    idx = get_existing_index()
    
    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=endpoint_name
    )
    
    # Verify index is deployed
    verify_index_deployment(endpoint, idx)
    
    # Upload vectors
    stream_upload_vectors(idx)


if __name__ == "__main__":
    main()
