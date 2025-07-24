import json
import uuid
import hashlib
import re
from typing import Dict, Any, Iterable, Tuple

from google.cloud import storage

PROJECT_ID     = "knowledgespace-217609"
BUCKET_NAME    = "ks_datasets"
PREPROC_PREFIX = "preprocessed_data/"   # input + overwrite

# slugify helper
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
        # deterministic fallback hash
        base = f"{ds_slug}::{source_file}::{chunk}"
        h = hashlib.sha1(base.encode("utf-8")).hexdigest()[:16]
        cand = f"{ds_slug}__{h}"

    # collision guard
    if cand in used:
        # append short counter or uuid
        suffix = 1
        base_cand = cand
        while cand in used:
            cand = f"{base_cand}-{suffix}"
            suffix += 1

    used.add(cand)
    return cand

def iter_json_blobs(client: storage.Client) -> Iterable[Tuple[str, storage.Blob]]:
    bucket = client.bucket(BUCKET_NAME)
    for blob in bucket.list_blobs(prefix=PREPROC_PREFIX):
        if blob.name.endswith(".json"):
            yield blob.name, blob

def process_blob(blob_name: str, blob: storage.Blob,
                 used_ids: set) -> int:
    txt = blob.download_as_text()
    try:
        records = json.loads(txt)
    except Exception as e:
        print(f"Skipping {blob_name}: failed to parse JSON ({e})")
        return 0

    if not isinstance(records, list):
        print(f"Skipping {blob_name}: JSON root not a list.")
        return 0

    updated = False
    for rec in records:
        if not isinstance(rec, dict):
            continue

        # ensure metadata_filters dict
        meta = rec.get("metadata_filters")
        if not isinstance(meta, dict):
            meta = {}
            rec["metadata_filters"] = meta

        # existing?
        vid = rec.get("vector_id") or meta.get("vector_id")
        if isinstance(vid, str) and vid.strip():
            if vid in used_ids:
                # collision across files; make new safe id
                new_vid = make_vector_id(meta, rec.get("chunk", ""), blob_name, used_ids)
                rec["vector_id"] = new_vid
                meta["vector_id"] = new_vid
                updated = True
            else:
                used_ids.add(vid)
                # ensure both places
                rec["vector_id"] = vid
                meta["vector_id"] = vid
            continue

        # create new
        new_vid = make_vector_id(meta, rec.get("chunk", ""), blob_name, used_ids)
        rec["vector_id"] = new_vid
        meta["vector_id"] = new_vid
        updated = True

    if updated:
        blob.upload_from_string(json.dumps(records, ensure_ascii=False), content_type="application/json")
        print(f"Updated + wrote {blob_name}")
    else:
        print(f"No changes needed {blob_name}")
    return len(records)

def main():
    storage_client = storage.Client(project=PROJECT_ID)
    used_ids = set()

    total = 0
    for name, blob in iter_json_blobs(storage_client):
        total += process_blob(name, blob, used_ids)

    print(f"\nProcessed {total:,} records across all files.")
    print(f"Total unique vector_ids: {len(used_ids):,}")

if __name__ == "__main__":
    main()
