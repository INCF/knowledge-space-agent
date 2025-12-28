import json
import hashlib
from pathlib import Path
from typing import Iterable, Tuple

from google.cloud import storage

PROJECT_ID = "knowledgespace-217609"
BUCKET_NAME = "ks_datasets"
PREPROC_PREFIX = "preprocessed_data/"
OUT_JSONL = Path("all_chunks.jsonl")


def is_valid_chunk(text: str) -> bool:
    if not text:
        return False

    t = text.strip()
    if not t:
        return False

    if len(t) < 50:
        return False

    if len(t) > 6000:
        return False

    noise_terms = [
        "all rights reserved",
        "terms of use",
        "cookie policy",
        "click here",
        "login to view",
        "javascript:void",
    ]

    tl = t.lower()
    if any(n in tl for n in noise_terms):
        return False

    return True


def make_hash_id(chunk: str, used: set) -> str:
    chunk_text = (chunk or "").strip()
    if not chunk_text:
        chunk_text = "empty"

    h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]

    if h in used:
        suffix = 1
        base_h = h
        while h in used:
            h = f"{base_h}-{suffix}"
            suffix += 1

    used.add(h)
    return h


def iter_json_blobs(client: storage.Client) -> Iterable[Tuple[str, storage.Blob]]:
    bucket = client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=PREPROC_PREFIX):
        if blob.name.endswith(".json"):
            yield blob.name, blob


def process_blob(blob_name: str, blob: storage.Blob, used_ids: set):
    try:
        print(f"Downloading {blob_name} ({blob.size / (1024*1024):.1f} MB)...")
        records = json.loads(blob.download_as_text())
    except KeyboardInterrupt:
        print(f"Interrupted while downloading {blob_name}")
        raise
    except Exception as e:
        print(f"Error downloading {blob_name}: {e}")
        return []

    if not isinstance(records, list):
        print(f"Skipping {blob_name}: not a list")
        return []

    output_records = []
    skipped = 0

    for i, rec in enumerate(records):
        if i > 0 and i % 10000 == 0:
            print(f"  Processing record {i:,}...")

        if not isinstance(rec, dict):
            continue

        chunk = rec.get("chunk", "")

        if not is_valid_chunk(chunk):
            skipped += 1
            continue

        hash_id = make_hash_id(chunk, used_ids)

        output_records.append({
            "datapoint_id": hash_id,
            "chunk": chunk,
            "metadata_filters": rec.get("metadata_filters", {}),
            "source_file": blob_name,
        })

    if skipped > 0:
        print(f"  Skipped {skipped} low-quality chunks in {blob_name}")

    return output_records


def main():
    storage_client = storage.Client(project=PROJECT_ID)
    used_ids = set()
    total = 0

    try:
        with OUT_JSONL.open("w", encoding="utf-8") as fout:
            for blob_name, blob in iter_json_blobs(storage_client):
                output_records = process_blob(blob_name, blob, used_ids)

                for record in output_records:
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")

                total += len(output_records)
                print(f"SUCCESS: Wrote {len(output_records):,} records from {blob_name}")

        print(f"\nSUCCESS: Completed: {total:,} total records")
        print(f"Output: {OUT_JSONL.resolve()}")

    except KeyboardInterrupt:
        print(f"\nWARNING: Interrupted after processing {total:,} records")
        print(f"Partial output saved: {OUT_JSONL.resolve()}")
    except Exception as e:
        print(f"\nERROR: {e}")
        print(f"Partial output may be saved: {OUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()
