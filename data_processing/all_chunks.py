import json
import csv
import hashlib
import re
from pathlib import Path
from typing import Dict, Any, Iterable, Tuple

from google.cloud import storage


PROJECT_ID     = "knowledgespace-217609"
BUCKET_NAME    = "ks_datasets"
PREPROC_PREFIX = "preprocessed_data/"
OUT_JSONL      = Path("all_chunks.jsonl")
OUT_MANIFEST   = Path("all_chunks_manifest.csv")

_slug_re = re.compile(r"[^a-z0-9]+")

def slugify(s: str) -> str:
    s = s.lower()
    s = _slug_re.sub("-", s).strip("-")
    return s or "src"

def make_vector_id(meta: Dict[str, Any], chunk: str, source_file: str,
                   used: set) -> str:
    dsname = (meta.get("datasource_name") or "").strip()
    ds_slug = slugify(dsname) if dsname else "unk"
    rec_id = meta.get("id")
    if isinstance(rec_id, str) and rec_id.strip():
        cand = f"{ds_slug}__{rec_id.strip()}"
    else:
        base = f"{ds_slug}::{source_file}::{chunk}"
        cand = f"{ds_slug}__{hashlib.sha1(base.encode('utf-8')).hexdigest()[:16]}"
    if cand in used:
        stem = cand
        i = 1
        while cand in used:
            cand = f"{stem}-{i}"
            i += 1
    used.add(cand)
    return cand

def iter_json_blobs(client: storage.Client) -> Iterable[Tuple[str, storage.Blob]]:
    bucket = client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=PREPROC_PREFIX):
        if blob.name.endswith(".json"):
            yield blob.name, blob

def main():
    storage_client = storage.Client(project=PROJECT_ID)
    used_ids = set()
    total = 0
    manifest_rows = []

    with OUT_JSONL.open("w", encoding="utf-8") as fout:
        for blob_name, blob in iter_json_blobs(storage_client):
            txt = blob.download_as_text()
            try:
                records = json.loads(txt)
            except Exception as e:
                print(f"Skipping {blob_name}: JSON parse error {e}")
                continue
            if not isinstance(records, list):
                print(f"Skipping {blob_name}: root not list")
                continue

            written = 0
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                chunk = rec.get("chunk", "")

                meta = rec.get("metadata_filters")
                if not isinstance(meta, dict):
                    meta = {}
                meta = dict(meta)  # copy

                # prefer preexisting vector_id
                vid = rec.get("vector_id") or meta.get("vector_id")
                if vid and vid in used_ids:
                    # global collision
                    vid = make_vector_id(meta, chunk, blob_name, used_ids)
                elif not vid:
                    vid = make_vector_id(meta, chunk, blob_name, used_ids)
                else:
                    used_ids.add(vid)

                meta["vector_id"] = vid
                if not meta.get("id"):
                    meta["id"] = vid  # safe fallback

                out_rec = {
                    "datapoint_id": vid,
                    "chunk": chunk,
                    "metadata_filters": meta,
                    "source_file": blob_name,
                }
                fout.write(json.dumps(out_rec, ensure_ascii=False) + "\n")
                written += 1
                total += 1

            manifest_rows.append((blob_name, written))
            print(f"Wrote {written:,} from {blob_name}")

    with OUT_MANIFEST.open("w", newline="", encoding="utf-8") as csvout:
        w = csv.writer(csvout)
        w.writerow(["source_file", "records_written"])
        w.writerows(manifest_rows)

    print(f"\nDone. Total records written: {total:,}")
    print(f"JSONL: {OUT_JSONL.resolve()}")
    print(f"Manifest: {OUT_MANIFEST.resolve()}")

if __name__ == "__main__":
    main()
