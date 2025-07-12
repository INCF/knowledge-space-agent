import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH   = "ks_datasets/raw_dataset/data_sources/scr_017041_sparc_old.json"
OUTPUT_GCS_PATH  = "ks_datasets/preprocessed_data/scr_017041_sparc_old.json"
DATASOURCE_ID    = "scr_017041_sparc_old"
DATASOURCE_NAME  = "SPARC Old"
DATASOURCE_TYPE  = "dataset_archive"

def clean_html(html_str: str) -> str:
    return BeautifulSoup(html_str or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join([str(x).strip() for x in lst if isinstance(x, str) and x.strip()])

def preprocess_record(rec: dict) -> dict:
    item       = rec.get("item", {}) or {}
    protocols  = rec.get("protocols", []) or []
    if isinstance(protocols, str):
        protocols = [protocols]
    dc         = rec.get("dc", {}) or {}

    name        = item.get("name", "") or ""
    description = item.get("description", "") or ""
    folder_name = item.get("folder_name", "") or ""
    keywords    = item.get("keywords", []) or []
    modalities  = item.get("modalities", []) or []
    title       = dc.get("title", "") or ""
    main_id     = dc.get("identifier", "")

    parts = [
        name,
        clean_html(description),
        folder_name,
        safe_join(keywords),
        safe_join(modalities),
        title
    ]
    chunk = "\n".join([p for p in parts if p])

    meta = {
        "name": name,
        "keywords": [k for k in keywords if isinstance(k, str)],
        "modalities": [m for m in modalities if isinstance(m, str)],
        "protocols": [p for p in protocols if isinstance(p, str)],
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_type": DATASOURCE_TYPE,
    }
    if main_id:
        meta["identifier"] = main_id

    desc_urls = extract_urls(description)
    for idx, url in enumerate(desc_urls, start=1):
        meta[f"identifier{idx}"] = url

    return {
        "chunk": chunk,
        "metadata_filters": meta
    }

client = storage.Client()

in_bucket, in_blob = INPUT_GCS_PATH.split("/", 1)
raw = client.bucket(in_bucket).blob(in_blob).download_as_text()
records = json.loads(raw)

processed = [preprocess_record(r) for r in records]

# Printing a sample
print("Sample preprocessed record:\n", json.dumps(processed[0], indent=2, ensure_ascii=False))

# Upload preprocessed JSON
out_bucket, out_blob = OUTPUT_GCS_PATH.split("/", 1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(
    json.dumps(processed, indent=2, ensure_ascii=False),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")

"""
 {
  "chunk": "Brainstem neuron recordings - Morris Lab USF\nSimultaneously-recorded cat brainstem neuron extracellular potentials with nerves and sorted spike trains\nFeline brainstem neuron extracellular potential recordings\nswallow; superior laryngeal nerve; electrical stimulation; respiration; neural circuits; phrenic; hypoglossal; blood pressure; tracheal pressure; expired CO2\nelectrophysiology\nFeline brainstem neuron extracellular potential recordings",
  "metadata_filters": {
    "name": "Brainstem neuron recordings - Morris Lab USF",
    "keywords": [
      "swallow",
      "superior laryngeal nerve",
      "electrical stimulation",
      "respiration",
      "neural circuits",
      "phrenic",
      "hypoglossal",
      "blood pressure",
      "tracheal pressure",
      "expired CO2"
    ],
    "modalities": [
      "electrophysiology"
    ],
    "protocols": [
      "Morris USF Lab protocol"
    ],
    "datasource_id": "scr_017041_sparc_old",
    "datasource_name": "SPARC Old",
    "datasource_type": "dataset_archive",
    "identifier": "https://doi.org/10.26275/1upo-xvkt"
  }
}
Uploaded 54 records to gs://ks_datasets/preprocessed_data/scr_017041_sparc_old.json
    """