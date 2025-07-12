import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH          = "ks_datasets/raw_dataset/data_sources/scr_017571_dandi.json"
OUTPUT_GCS_PATH         = "ks_datasets/preprocessed_data/scr_017571_dandi.json"
DATASOURCE_ID           = "scr_017571_dandi"
DATASOURCE_NAME         = "Dandi Archive"
DATASOURCE_DESCRIPTION  = (
    "Contains data sets and metadata for neuroscience data stored in the DANDI Archive."
)
DATASOURCE_TYPE         = "archive"

def clean_html(html_str: str) -> str:
    return BeautifulSoup(html_str or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join([str(x).strip() for x in lst if isinstance(x, str) and x.strip()])

def preprocess_record(rec: dict) -> dict:
    rec_id   = rec.get("id", "")
    dc       = rec.get("dc", {}) or {}
    title    = dc.get("title", "") or ""
    desc     = dc.get("description", "") or ""
    about    = rec.get("about", []) or []
    contrib  = rec.get("contributors", []) or []
    species  = rec.get("species", []) or []
    ds_std   = rec.get("dataStandard", []) or []
    approach = rec.get("approach", []) or []
    meas     = rec.get("measurementTechnique", []) or []
    license  = rec.get("license", []) or []
    keywords = rec.get("keywords", []) or []
    related  = rec.get("relatedResource", []) or []

    parts = [
        title,
        clean_html(desc),
        safe_join(about),
        safe_join(species),
        safe_join(ds_std),
        safe_join(approach),
        safe_join(meas),
        safe_join(license),
        safe_join(keywords),
        safe_join([r.get("name", "") for r in related]),
    ]
    chunk = "\n".join([p for p in parts if p])

    meta = {
        "id": rec_id,
        "about": [a for a in about if isinstance(a, str)],
        "contributors": [c for c in contrib if isinstance(c, str)],
        "species": [s for s in species if isinstance(s, str)],
        "dataStandard": [d for d in ds_std if isinstance(d, str)],
        "approach": [a for a in approach if isinstance(a, str)],
        "measurementTechnique": [m for m in meas if isinstance(m, str)],
        "license": [l for l in license if isinstance(l, str)],
        "keywords": [k for k in keywords if isinstance(k, str)],
        "relatedResource": related,
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE,
    }

    main_id = dc.get("identifier")
    if main_id:
        meta["identifier"] = main_id

    # URLs in description
    desc_urls = extract_urls(desc)
    for idx, url in enumerate(desc_urls, start=1):
        meta[f"identifier{idx}"] = url

    # URLs in relatedResource
    for r in related:
        url = r.get("url") or r.get("identifier")
        if url:
            existing = [k for k in meta if k.startswith("identifier")]
            next_idx = len(existing) + 1
            meta[f"identifier{next_idx}"] = url

    return {"chunk": chunk, "metadata_filters": meta}

client = storage.Client()

in_bucket_name, in_blob_path = INPUT_GCS_PATH.split("/", 1)
bucket = client.bucket(in_bucket_name)
blob   = bucket.blob(in_blob_path)
raw    = blob.download_as_text()
records = json.loads(raw)

processed = [preprocess_record(rec) for rec in records]

# Printing a sample
print("Sample preprocessed record:\n", json.dumps(processed[0], indent=2, ensure_ascii=False))

# Upload preprocessed JSON
out_bucket_name, out_blob_path = OUTPUT_GCS_PATH.split("/", 1)
out_bucket = client.bucket(out_bucket_name)
out_blob   = out_bucket.blob(out_blob_path)
out_blob.upload_from_string(
    json.dumps(processed, indent=2, ensure_ascii=False),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


"""
{
  "chunk": "Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells\nData from \"Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells\" Senzai, Buzsaki, Neuron 2017. Electrophysiology recordings of hippocampus during theta maze exploration.\nhippocampus\nHouse mouse\nNeurodata Without Borders (NWB)\nelectrophysiological approach; behavioral approach\nsignal filtering technique; fourier analysis technique; spike sorting technique; behavioral technique; multi electrode extracellular electrophysiology recording technique\nspdx:CC-BY-4.0\ncell types; current source density; laminar recordings; oscillations; mossy cells; granule cells; optogenetics\nPhysiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells",
  "metadata_filters": {
    "id": "000003",
    "about": [
      "hippocampus"
    ],
    "contributors": [
      "Senzai, Yuta",
      "Fernandez-Ruiz, Antonio",
      "Buzsáki, György",
      "National Institutes of Health",
      "Nakajima Foundation",
      "National Science Foundation",
      "Simons Foundation"
    ],
    "species": [
      "House mouse"
    ],
    "dataStandard": [
      "Neurodata Without Borders (NWB)"
    ],
    "approach": [
      "electrophysiological approach",
      "behavioral approach"
    ],
    "measurementTechnique": [
      "signal filtering technique",
      "fourier analysis technique",
      "spike sorting technique",
      "behavioral technique",
      "multi electrode extracellular electrophysiology recording technique"
    ],
    "license": [
      "spdx:CC-BY-4.0"
    ],
    "keywords": [
      "cell types",
      "current source density",
      "laminar recordings",
      "oscillations",
      "mossy cells",
      "granule cells",
      "optogenetics"
    ],
    "relatedResource": [
      {
        "identifier": "doi:10.1016/j.neuron.2016.12.011",
        "name": "Physiological Properties and Behavioral Correlates of Hippocampal Granule Cells and Mossy Cells",
        "url": "https://doi.org/10.1016/j.neuron.2016.12.011",
        "relation": "dcite:IsDescribedBy",
        "schemaKey": "Resource",
        "resourceType": "dcite:JournalArticle"
      }
    ],
    "datasource_id": "scr_017571_dandi",
    "datasource_name": "Dandi Archive",
    "datasource_description": "Contains data sets and metadata for neuroscience data stored in the DANDI Archive.",
    "datasource_type": "archive",
    "identifier": "https://dandiarchive.org/dandiset/000003/draft",
    "identifier2": "https://doi.org/10.1016/j.neuron.2016.12.011"
  }
}
Uploaded 835 records to gs://ks_datasets/preprocessed_data/scr_017571_dandi.json
    """