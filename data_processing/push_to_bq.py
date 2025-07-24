
import json
import uuid
from pathlib import Path
from typing import Dict, Any, List

from google.cloud import bigquery


PROJECT_ID  = "knowledgespace-217609"
DATASET_ID  = "ks_metadata"
TABLE_ID    = "docstore"
LOCATION    = "EU"

INPUT_JSONL = Path("all_chunks.jsonl")  

ROWS_PER_STAGE = 100_000

def ensure_table(bq: bigquery.Client):
    ds_ref = bigquery.DatasetReference(PROJECT_ID, DATASET_ID)
    try:
        bq.get_dataset(ds_ref)
    except Exception:
        ds = bigquery.Dataset(ds_ref)
        ds.location = LOCATION
        bq.create_dataset(ds)
        print(f"Created dataset {PROJECT_ID}:{DATASET_ID}")

    tbl_ref = ds_ref.table(TABLE_ID)
    try:
        bq.get_table(tbl_ref)
        return tbl_ref
    except Exception:
        schema = [
            bigquery.SchemaField("datapoint_id", "STRING", mode="REQUIRED"),
            bigquery.SchemaField("chunk", "STRING"),
            bigquery.SchemaField("metadata_filters", "JSON"),
            bigquery.SchemaField("source_file", "STRING"),
        ]
        tbl = bigquery.Table(tbl_ref, schema=schema)
        bq.create_table(tbl)
        print(f"Created table {tbl.full_table_id}")
        return tbl_ref

def merge_rows(bq: bigquery.Client, rows: List[Dict[str, Any]]):
    if not rows:
        return
    stage_name = f"_stage_{uuid.uuid4().hex[:8]}"
    stage_ref  = f"{PROJECT_ID}.{DATASET_ID}.{stage_name}"
    schema = [
        bigquery.SchemaField("datapoint_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("chunk", "STRING"),
        bigquery.SchemaField("metadata_filters", "JSON"),
        bigquery.SchemaField("source_file", "STRING"),
    ]
    bq.create_table(bigquery.Table(stage_ref, schema=schema))
    job = bq.load_table_from_json(
        rows,
        stage_ref,
        job_config=bigquery.LoadJobConfig(
            schema=schema,
            write_disposition="WRITE_TRUNCATE",
        ),
    )
    job.result()
    dest = f"{PROJECT_ID}.{DATASET_ID}.{TABLE_ID}"
    sql = f"""
    MERGE `{dest}` T
    USING `{stage_ref}` S
    ON T.datapoint_id = S.datapoint_id
    WHEN MATCHED THEN UPDATE SET
      chunk = S.chunk,
      metadata_filters = S.metadata_filters,
      source_file = S.source_file
    WHEN NOT MATCHED THEN
      INSERT (datapoint_id, chunk, metadata_filters, source_file)
      VALUES (S.datapoint_id, S.chunk, S.metadata_filters, S.source_file)
    """
    bq.query(sql).result()
    bq.delete_table(stage_ref, not_found_ok=True)

def stream_load():
    bq = bigquery.Client(project=PROJECT_ID, location=LOCATION)
    ensure_table(bq)

    buf: List[Dict[str, Any]] = []
    total = 0
    with INPUT_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            buf.append(rec)
            if len(buf) >= ROWS_PER_STAGE:
                merge_rows(bq, buf)
                total += len(buf)
                buf.clear()
                print(f"Merged {total:,} rows ...")
    if buf:
        merge_rows(bq, buf)
        total += len(buf)
    print(f"Done. {total:,} rows merged into {PROJECT_ID}.{DATASET_ID}.{TABLE_ID}")

if __name__ == "__main__":
    stream_load()
