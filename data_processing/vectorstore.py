import json
from pathlib import Path
from typing import List, Sequence
import os
from dotenv import load_dotenv

from tqdm import tqdm
from google.cloud import aiplatform, aiplatform_v1
from google.cloud.aiplatform_v1.types import IndexDatapoint

# Load environment variables from .env
load_dotenv()

# CONFIG (from .env or defaults)
PROJECT_ID         = os.environ.get("GCP_PROJECT_ID", "knowledgespace-217609").replace('"','')
REGION             = os.environ.get("GCP_REGION", "europe-north1").replace('"','')
PROJECT_NUMBER     = os.environ.get("GCP_PROJECT_NUMBER", "452527985942").replace('"','')
LOCAL_EMBEDDINGS_PATH = Path(os.environ.get("LOCAL_EMBEDDINGS_PATH", "data_processing/embeddings.jsonl").replace('"',''))
INDEX_DISPLAY_NAME = os.environ.get("INDEX_DISPLAY_NAME", "ks-chunks-index-nomic-768").replace('"','')
INDEX_ENDPOINT_ID  = os.environ.get("INDEX_ENDPOINT_ID", "6943442317684506624").replace('"','')
DEPLOYED_INDEX_ID  = os.environ.get("DEPLOYED_INDEX_ID", "deployed_ks_chunks_index_nomic_768").replace('"','')
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", 768))
DISTANCE_MEASURE     = os.environ.get("DISTANCE_MEASURE", "DOT_PRODUCT_DISTANCE").replace('"','')
UPSERT_BATCH_SIZE    = int(os.environ.get("UPSERT_BATCH_SIZE", 100))
API_ENDPOINT = f"{PROJECT_NUMBER}.{REGION}-{PROJECT_ID}.vdb.vertexai.goog"


#INDEX CREATE / DEPLOY 
def get_or_create_streaming_index():
    print(f"Checking for existing index named '{INDEX_DISPLAY_NAME}'...")
    indexes = aiplatform.MatchingEngineIndex.list(
        filter=f'display_name="{INDEX_DISPLAY_NAME}"'
    )
    if indexes:
        idx = indexes[0]
        print(f"✅ Found existing index: {idx.resource_name}")
        return idx

    print("No existing index found. Creating a new streaming index...")
    idx = aiplatform.MatchingEngineIndex.create_brute_force_index(
        display_name=INDEX_DISPLAY_NAME,
        dimensions=EMBEDDING_DIMENSIONS,
        distance_measure_type=DISTANCE_MEASURE,
        index_update_method="STREAM_UPDATE",
    )
    print(f"Index created: {idx.resource_name}")
    return idx


def _get_deployed_list(endpoint_obj):
    dl = getattr(endpoint_obj, "deployed_indexes", None)
    if dl is None:
        dl = endpoint_obj.gca_resource.deployed_indexes
    return dl


def deploy_index_if_needed(endpoint_obj, index_obj):
    """Deploy index to endpoint if not already deployed."""
    deployed_list = _get_deployed_list(endpoint_obj)

    for di in deployed_list:
        if di.index == index_obj.resource_name or di.id == DEPLOYED_INDEX_ID:
            print(f"Index already deployed (id={di.id}).")
            return

    print("Deploying index to endpoint…")
    op = endpoint_obj.deploy_index(index=index_obj, deployed_index_id=DEPLOYED_INDEX_ID)
    op.result()  # wait
    print(" Deployment completed.")


# UPSERT
def stream_upload_vectors(index_obj):
    print(f"Reading vectors from '{LOCAL_EMBEDDINGS_PATH}'...")
    with LOCAL_EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        total = sum(1 for _ in f)

    batch: List[IndexDatapoint] = []
    with LOCAL_EMBEDDINGS_PATH.open("r", encoding="utf-8") as f:
        for line in tqdm(f, total=total, desc="Upserting"):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            dp = IndexDatapoint(
                datapoint_id=rec["id"],
                feature_vector=rec["embedding"],
            )
            batch.append(dp)
            if len(batch) == UPSERT_BATCH_SIZE:
                index_obj.upsert_datapoints(datapoints=batch)
                batch = []

    if batch:
        index_obj.upsert_datapoints(datapoints=batch)

    print("All vectors upserted.")


def build_match_client() -> aiplatform_v1.MatchServiceClient:
    return aiplatform_v1.MatchServiceClient(
        client_options={"api_endpoint": API_ENDPOINT}
    )


def find_neighbors(
    feature_vector: Sequence[float],
    neighbor_count: int = 10,
    return_full_datapoint: bool = False,
):
    client = build_match_client()

    query_dp = IndexDatapoint(feature_vector=feature_vector)
    query = aiplatform_v1.FindNeighborsRequest.Query(
        datapoint=query_dp,
        neighbor_count=neighbor_count,
    )
    request = aiplatform_v1.FindNeighborsRequest(
        index_endpoint=f"projects/{PROJECT_NUMBER}/locations/{REGION}/indexEndpoints/{INDEX_ENDPOINT_ID}",
        deployed_index_id=DEPLOYED_INDEX_ID,
        queries=[query],
        return_full_datapoint=return_full_datapoint,
    )
    response = client.find_neighbors(request)
    return response.nearest_neighbors



def main():
    aiplatform.init(project=PROJECT_ID, location=REGION)

    idx = get_or_create_streaming_index()

    endpoint_name = (
        f"projects/{PROJECT_ID}/locations/{REGION}/indexEndpoints/{INDEX_ENDPOINT_ID}"
    )
    endpoint = aiplatform.MatchingEngineIndexEndpoint(
        index_endpoint_name=endpoint_name
    )

    deploy_index_if_needed(endpoint, idx)
    # Comment the next line if you already upserted once.
    stream_upload_vectors(idx)

    # Example search 
    demo_vec = [0.0] * EMBEDDING_DIMENSIONS
    print(find_neighbors(demo_vec, 5))


if __name__ == "__main__":
    main()
