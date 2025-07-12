import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH   = "ks_datasets/raw_dataset/data_sources/scr_007271_modeldb_models.json"
OUTPUT_GCS_PATH  = "ks_datasets/preprocessed_data/scr_007271_modeldb_models.json"
DATASOURCE_ID    = "scr_007271_modeldb_models"
DATASOURCE_NAME  = "ModelDB"
DATASOURCE_TYPE  = "models"

def clean_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join(str(x).strip() for x in lst if isinstance(x, str) and x.strip())

def preprocess_record(rec: dict) -> dict:
    model_type            = rec.get("model_type", "") or ""
    model_neurotransmitters= rec.get("model_neurotransmitters", []) or []
    model_neurons         = rec.get("model_neurons", []) or []
    dc                    = rec.get("dc", {}) or {}
    subject               = dc.get("subject", []) or []
    title                 = dc.get("title", "") or ""
    description           = dc.get("description", "") or ""
    model_receptors       = rec.get("model_receptors", []) or []
    simulator_software    = rec.get("simulator_software", "") or ""
    name                  = rec.get("name", "") or ""
    implemented_by        = rec.get("implemented_by", "") or ""
    notes                 = rec.get("notes", "") or ""
    model_concepts        = rec.get("model_concepts", []) or []
    model_currents        = rec.get("model_currents", []) or []
    dataTypes             = rec.get("dataItem", {}).get("dataTypes", []) or []
    model_url             = rec.get("model_url", "") or ""
    rec_id                = rec.get("id", "") or ""

    parts = [
        model_type,
        safe_join(model_neurotransmitters),
        safe_join(model_neurons),
        safe_join(subject),
        title,
        clean_html(description),
        safe_join(model_receptors),
        simulator_software,
        name,
        clean_html(implemented_by),
        clean_html(notes),
        safe_join(model_concepts),
        safe_join(model_currents),
    ]
    chunk = "\n".join(p for p in parts if p)

    meta = {
        "model_type": model_type,
        "id": rec_id,
        "name": name,
        "model_concepts": [c for c in model_concepts if isinstance(c, str)],
        "simulator_software": simulator_software,
        "model_currents": [c for c in model_currents if isinstance(c, str)],
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_type": DATASOURCE_TYPE,
    }

    primary_id = dc.get("identifier")
    if primary_id:
        meta["identifier"] = primary_id

    if model_url:
        meta["model_url"] = model_url

    urls = extract_urls(description) + extract_urls(notes)
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen and u != primary_id and u != model_url:
            seen.add(u)
            unique_urls.append(u)
    for idx, u in enumerate(unique_urls, start=1):
        meta[f"identifier{idx}"] = u

    return {"chunk": chunk, "metadata_filters": meta}

client = storage.Client()

in_bucket, in_blob = INPUT_GCS_PATH.split("/", 1)
raw = client.bucket(in_bucket).blob(in_blob).download_as_text()
records = json.loads(raw)

processed = [preprocess_record(r) for r in records]

print("Sample preprocessed record:\n", json.dumps(processed[0], indent=2, ensure_ascii=False))

out_bucket, out_blob = OUTPUT_GCS_PATH.split("/", 1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(
    json.dumps(processed, indent=2, ensure_ascii=False),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


""" 
  {
  "chunk": "Realistic Network\nRegulation of a slow STG rhythm (Nadim et al 1998)\nFrequency regulation of a slow rhythm by a fast periodic input. Nadim, F., Manor, Y., Nusbaum, M. P., Marder, E. (1998) J. Neurosci. 18: 5053-5067\nNEURON\nRegulation of a slow STG rhythm (Nadim et al 1998)\nNadim, Farzan [Farzan at andromeda.Rutgers.edu]\nFrequency regulation of a slow rhythm by a fast periodic input. Nadim, F., Manor, Y., Nusbaum, M. P., Marder, E. (1998) J. Neurosci. 18: 5053-5067\nTemporal Pattern Generation; Invertebrate\nI Na,t; I K",
  "metadata_filters": {
    "model_type": "Realistic Network",
    "id": "o3511",
    "name": "Regulation of a slow STG rhythm (Nadim et al 1998)",
    "model_concepts": [
      "Temporal Pattern Generation",
      "Invertebrate"
    ],
    "simulator_software": "NEURON",
    "model_currents": [
      "I Na,t",
      "I K"
    ],
    "datasource_id": "scr_007271_modeldb_models",
    "datasource_name": "ModelDB",
    "datasource_type": "models",
    "identifier": "http://senselab.med.yale.edu/ModelDB/showModel?model=3511",
    "model_url": "http://senselab.med.yale.edu/ModelDB/showModel?model=3511"
  }
}
Uploaded 1362 records to gs://ks_datasets/preprocessed_data/scr_007271_modeldb_models.json 
"""