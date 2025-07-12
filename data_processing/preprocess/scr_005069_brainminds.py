import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH   = "ks_datasets/raw_dataset/data_sources/scr_005069_brainminds.json"
OUTPUT_GCS_PATH  = "ks_datasets/preprocessed_data/scr_005069_brainminds.json"
DATASOURCE_ID    = "scr_005069_brainminds"
DATASOURCE_NAME  = "Brain/MINDS"
DATASOURCE_TYPE  = "dataset_archive"

def clean_html(html_str: str) -> str:
    return BeautifulSoup(html_str or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join([str(x).strip() for x in lst if isinstance(x, str) and x.strip()])

def preprocess_record(rec: dict) -> dict:
    name        = rec.get("name", "") or ""
    description = rec.get("description", "") or ""
    keywords    = rec.get("keywords", []) or []
    citation    = rec.get("citation", "") or ""
    license_obj = rec.get("license", {}) or {}
    dc          = rec.get("dc", {}) or {}
    url_field   = rec.get("url", "") or ""

    parts = [
        name,
        clean_html(description),
        safe_join(keywords),
        license_obj.get("@type", ""),
        citation
    ]
    chunk = "\n".join([p for p in parts if p])

    meta = {
        "name": name,
        "keywords": [k for k in keywords if isinstance(k, str)],
        "citation": citation,
        "license": license_obj.get("@type", ""),
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_type": DATASOURCE_TYPE,
    }

    main_id = dc.get("identifier") or url_field
    if main_id:
        meta["identifier"] = main_id

    urls = []
    urls += extract_urls(description)
    urls += extract_urls(citation)
    if license_obj.get("url"):
        urls.append(license_obj["url"])
    urls = list(dict.fromkeys(urls))
    for idx, url in enumerate(urls, start=1):
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
  "chunk": "Brain/MINDS Marmoset Tracer Injection image-sets\nThe Brain/MINDS Marmoset Tracer Injection image-sets.\nmarmoset; Callithrix jacchus; prefrontal cortex; PFC; Connectome Matrix; AAV; serial two-photon tomography; STPT; nonnegative matrix factorization; non-human primate; cortical column; anterograde tracer; association cortex; topographic connectivity; tractography\nCreativeWork\nWatakabe A., Skibbe H., Nakae K., Abe H., Ichinohe N., Rachmadi M.F., Wang J., Takaji M., Mizukami H., Woodward A., Gong R., Hata J., Van Essen D.C., Okano H., Ishii S., Yamamori T. Local and long-distance organization of prefrontal cortex circuits in the marmoset brain. Neuron. 2023 May 9, S0896-6273(23)00338-0. https://doi.org/10.1016/j.neuron.2023.04.028",
  "metadata_filters": {
    "name": "Brain/MINDS Marmoset Tracer Injection image-sets",
    "keywords": [
      "marmoset",
      "Callithrix jacchus",
      "prefrontal cortex",
      "PFC",
      "Connectome Matrix",
      "AAV",
      "serial two-photon tomography",
      "STPT",
      "nonnegative matrix factorization",
      "non-human primate",
      "cortical column",
      "anterograde tracer",
      "association cortex",
      "topographic connectivity",
      "tractography"
    ],
    "citation": "Watakabe A., Skibbe H., Nakae K., Abe H., Ichinohe N., Rachmadi M.F., Wang J., Takaji M., Mizukami H., Woodward A., Gong R., Hata J., Van Essen D.C., Okano H., Ishii S., Yamamori T. Local and long-distance organization of prefrontal cortex circuits in the marmoset brain. Neuron. 2023 May 9, S0896-6273(23)00338-0. https://doi.org/10.1016/j.neuron.2023.04.028",
    "license": "CreativeWork",
    "datasource_id": "scr_005069_brainminds",
    "datasource_name": "Brain/MINDS",
    "datasource_type": "dataset_archive",
    "identifier": "https://doi.org/10.24475/bminds.mti.6358",
    "identifier1": "https://doi.org/10.1016/j.neuron.2023.04.028",
    "identifier2": "http://creativecommons.org/licenses/by/4.0/"
  }
}
Uploaded 10 records to gs://ks_datasets/preprocessed_data/scr_005069_brainminds.json """