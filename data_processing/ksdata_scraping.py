import json
import os
import sys
import time
import requests
from datetime import datetime, timezone
from google.cloud import storage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Config from environment
PUBLIC_API_URL   = os.getenv("PUBLIC_API_URL")
ELASTIC_BASE_URL = os.getenv("ELASTIC_BASE_URL")
ELASTIC_AUTH     = (os.getenv("ELASTIC_USERNAME"), os.getenv("ELASTIC_PASSWORD"))
PAGE_SIZE        = int(os.getenv("PAGE_SIZE", 1000))
GCS_BUCKET       = os.getenv("GCS_BUCKET")
GCS_PREFIX       = os.getenv("GCS_PREFIX")

HEADERS = {
    "User-Agent": "Elastic-Harvester/1.0",
    "Accept-Encoding": "gzip",
    "Content-Type": "application/json"
}

def make_request(session, method, url, **kwargs):
    try:
        response = session.request(method, url, timeout=90, **kwargs)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"\nRequest failed: {e}", file=sys.stderr)
        raise

def stream_to_gcs(bucket, blob_name, batches):
    print(f"Starting stream to gs://{GCS_BUCKET}/{blob_name}")
    blob = bucket.blob(blob_name)
    with blob.open("w", content_type="application/json") as gcs_stream:
        gcs_stream.write("[")
        is_first_item = True
        total_saved = 0
        for batch in batches:
            for item in batch:
                if not is_first_item:
                    gcs_stream.write(",")
                gcs_stream.write(json.dumps(item.get("_source", {}), separators=(",", ":")))
                is_first_item = False
            total_saved += len(batch)
            print(f"  Streamed {total_saved} datasets...", end='\r', flush=True)
        gcs_stream.write("]")
    print(f"\n  Finished stream. Total saved: {total_saved}", flush=True)
    return total_saved

def pit_search_generator(session, datasource_id):
    pit_id = None
    try:
        pit_response = make_request(
            session, 'POST',
            f"{ELASTIC_BASE_URL}/{datasource_id}/_pit?keep_alive=2m"
        )
        pit_id = pit_response["id"]

        body = {
            "size": PAGE_SIZE,
            "pit": {"id": pit_id},
            "sort": [{"_doc": "asc"}]
        }
        
        while True:
            response = make_request(session, 'POST', f"{ELASTIC_BASE_URL}/_search", json=body)
            hits = response.get("hits", {}).get("hits", [])
            if not hits:
                break
            yield hits 
            body["search_after"] = hits[-1]["sort"]

    finally:
        if pit_id:
            try:
                make_request(session, 'DELETE', f"{ELASTIC_BASE_URL}/_pit", json={"id": pit_id})
                print("\n  -> Cleaned up PIT context.", flush=True)
            except Exception as e:
                print(f"\n  -> Note: Failed to clean up PIT context: {e}", file=sys.stderr)

def harvest_datasource(session, bucket, datasource_id):
    print(f"\nProcessing datasource: {datasource_id}", flush=True)
    blob_name = f"{GCS_PREFIX}/{datasource_id}.json"
    batches_generator = pit_search_generator(session, datasource_id)
    total_saved = stream_to_gcs(bucket, blob_name, batches_generator)
    return total_saved

def main():
    storage_client = storage.Client()
    bucket = storage_client.bucket(GCS_BUCKET)

    with requests.Session() as public_session:
        public_session.headers.update(HEADERS)
        print("Fetching list of all datasources from public API")
        datasource_ids = make_request(public_session, 'GET', f"{PUBLIC_API_URL}/datasources")
        if not isinstance(datasource_ids, list):
             sys.exit("Could not fetch list of datasources!")

    with requests.Session() as elastic_session:
        elastic_session.headers.update(HEADERS)
        elastic_session.auth = ELASTIC_AUTH

        start_time = datetime.now(timezone.utc)
        grand_total = 0
        print(f"\nFound {len(datasource_ids)} datasources to process via direct database access.")

        for ds_id in datasource_ids:
            try:
                grand_total += harvest_datasource(elastic_session, bucket, ds_id)
                time.sleep(1)
            except Exception as exc:
                print(f"ERROR: Datasource {ds_id} failed unexpectedly: {exc}", file=sys.stderr)

        duration = (datetime.now(timezone.utc) - start_time).total_seconds()
        print(f"\nAll done. Scraped {grand_total:,} datasets in {duration:,.0f} seconds.")

if __name__ == "__main__":
    main()
