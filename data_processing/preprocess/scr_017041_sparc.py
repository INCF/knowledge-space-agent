import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", ". ", " ", ""]
)

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

    science_tags = [
        f"Organization:{org_name}" if org_name else "",
        f"Item:{item_name}" if item_name else "",
        f"Keywords:{safe_join(keywords)}" if keywords else "",
        f"Title:{title}" if title else ""
    ]
    science_context_header = "\n".join(t for t in science_tags if t)
    
    desc_text = "\n".join(p for p in [clean_html(summary), clean_html(description)] if p)
    desc_chunks = text_splitter.split_text(desc_text)

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

    out = []
    if not desc_chunks:
        out.append({"chunk": science_context_header, "metadata_filters": meta})
    else:
        for index, text_block in enumerate(desc_chunks):
            focused_semantic_chunk = f"{science_context_header}\n--\nExcerpt part{index+1}:\n{text_block}"
            out.append({"chunk": focused_semantic_chunk, "metadata_filters": meta})
            
    return out

client = storage.Client()

in_bucket, in_blob = INPUT_GCS_PATH.split("/", 1)
raw = client.bucket(in_bucket).blob(in_blob).download_as_text()
records = json.loads(raw)

processed = []
for r in records:
    processed.extend(preprocess_record(r))

# printing sample
print("Sample:", json.dumps(processed[0], indent=2, ensure_ascii=False))

# upload
out_bucket, out_blob = OUTPUT_GCS_PATH.split("/", 1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(
    json.dumps(processed, indent=2, ensure_ascii=False),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


"""
{
  "chunk": "Organization:Mayo\nItem:Intracranial EEG Epilepsy - Study 3\nKeywords:epilepsy; eeg; intracranial; grid electrodes; strip electrodes; depth electrodes; seizure\nTitle:Intracranial EEG Epilepsy - Study 3\n--\nExcerpt part1:\nThe patient is a right-handed, 21-year old male who was admitted to the epilepsy monitoring unit for **intracranial monitoring**. The age at onset was 13 years old.",
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
Uploaded 344 records to gs://ks_datasets/preprocessed_data/scr_017041_sparc.json
"""