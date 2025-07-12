import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH   = "ks_datasets/raw_dataset/data_sources/scr_017041_sparc.json"
OUTPUT_GCS_PATH  = "ks_datasets/preprocessed_data/scr_017041_sparc.json"
DATASOURCE_ID    = "scr_017041_sparc"
DATASOURCE_NAME  = "SPARC"
DATASOURCE_TYPE  = "dataset_archive"

def clean_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join(str(x).strip() for x in lst if isinstance(x, str) and x.strip())

def preprocess_record(rec: dict) -> dict:
    rec_id    = rec.get("id")
    contribs  = rec.get("contributors", [])
    org_name  = rec.get("organizationName", "")
    item      = rec.get("item", {}) or {}
    dc        = rec.get("dc", {}) or {}

    item_name    = item.get("name", "")
    keywords     = item.get("keywords", []) or []
    summary      = item.get("summary", "")
    title        = dc.get("title", "")
    description  = dc.get("description", "")

    chunk_parts = [
        org_name,
        item_name,
        safe_join(keywords),
        clean_html(summary),
        title,
        clean_html(description),
    ]
    chunk = "\n".join(p for p in chunk_parts if p)

    meta = {
        "id": rec_id,
        "contributors": contribs,
        "organizationName": org_name,
        "item": {
            "name": item_name,
            "keywords": keywords,
            
        },
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_type": DATASOURCE_TYPE,
    }

    main_id = dc.get("identifier")
    if main_id:
        meta["identifier"] = main_id

    # then any URLs in summary, description
    urls = extract_urls(summary) + extract_urls(description)
    # dedupe
    seen = set()
    urls_unique = []
    for u in urls:
        if u not in seen:
            seen.add(u)
            urls_unique.append(u)
    for idx, u in enumerate(urls_unique, start=1):
        meta[f"identifier{idx}"] = u

    return {"chunk": chunk, "metadata_filters": meta}

client = storage.Client()

in_bucket, in_blob = INPUT_GCS_PATH.split("/", 1)
raw = client.bucket(in_bucket).blob(in_blob).download_as_text()
records = json.loads(raw)

processed = [preprocess_record(r) for r in records]

# printing sample
print("Sample:", json.dumps(processed[0], indent=2, ensure_ascii=False))

# upload
out_bucket, out_blob = OUTPUT_GCS_PATH.split("/", 1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(
    json.dumps(processed, indent=2, ensure_ascii=False),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


"""  {
  "chunk": "Mayo\nIntracranial EEG Epilepsy - Study 3\nepilepsy; eeg; intracranial; grid electrodes; strip electrodes; depth electrodes; seizure\nThe patient is a right-handed, 21-year old male who was admitted to the epilepsy monitoring unit for **intracranial monitoring**. The age at onset was 13 years old.\n\nThese data are from a 6 x 8 grid that was placed over the right frontal region, two 4 x 6 grids placed over the right temporal lobe and right parietal region, respectively, a 4-contact strip wrapped beneath the right anterior temporal lobe, a 4-contact strip wrapped beneath the right posterior temporal lobe, and a 4-contact depth electrode inserted in the anterior temporal region.\n \nThis is an abnormal computer-assisted prolonged intracranial EEG monitoring session due to the presence of frequent independent right frontal, right parietal and right temporal epileptiform discharges. During the monitoring session, the patient had a total of **eleven seizures, including nine clinical seizures and two seizures which were subclinical**. The first six seizures showed a diffuse ictal onset over the right frontal and right temporal grids. The last five seizures showed either a focal ictal onset simultaneously from the right anterior temporal strip and the posterior superior region of the right frontal grid, or focal onset from these two foci independently. These findings could be consistent with a localization-related epilepsy with seizure onset in the right frontotemporal neocortex.\n \nThe patient underwent surgery of reopening the right forntotemporoparietal craniotomy and removing the subdural grid electrodes, then a **right anterior-superior frontocortical resection, a right temporal lobectomy, and a right amygdalohippocampectomy**. Pathology samples of leptomeningeal and focal parenchymal lymphohistiocytic infiltrate consistent with grid/electrode placement. Mild subpial and subcortical gliosis. The patient also underwent surgery for a right mesial temporal lobe resection. Pathology samples of mesial temporal structures with focal parenchymal macrophage aggregate consistent with electrode placement. **Mild subpial and subcortical gliosis**.\nIntracranial EEG Epilepsy - Study 3\nData for a patient with epilepsy obtained for clinical treatment and collected for research, consisting of a large intracranial EEG dataset (grid, strip, and depth electrodes) with multiple seizures.",
  "metadata_filters": {
    "id": 14,
    "contributors": [
      {
        "full_name": "Brian Litt",
        "orcid_id": null
      },
      {
        "full_name": "Gregory Worrell",
        "orcid_id": null
      }
    ],
    "organizationName": "Mayo",
    "item": {
      "name": "Intracranial EEG Epilepsy - Study 3",
      "keywords": [
        "epilepsy",
        "eeg",
        "intracranial",
        "grid electrodes",
        "strip electrodes",
        "depth electrodes",
        "seizure"
      ]
    },
    "datasource_id": "scr_017041_sparc",
    "datasource_name": "SPARC",
    "datasource_type": "dataset_archive",
    "identifier": "https://doi.org/10.26275/psj7-wggf"
  }
}
Uploaded 344 records to gs://ks_datasets/preprocessed_data/scr_017041_sparc.json"""