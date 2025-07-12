import json, re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_017612_ebrains.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_017612_ebrains.json"
DATASOURCE_ID="scr_017612_ebrains"
DATASOURCE_NAME="EBRAINS"
DATASOURCE_DESCRIPTION="A curated repository of EBRAINS datasets."
DATASOURCE_TYPE="morphology"

def clean_html(html):
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text):
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(x,sep="; "):
    if isinstance(x,list):
        return sep.join(v.strip() for v in x if isinstance(v,str) and v.strip())
    return str(x).strip()

client = storage.Client()
b,bl = INPUT_GCS_PATH.split("/",1)
records = json.loads(client.bucket(b).blob(bl).download_as_text())
out = []
for r in records:
    dc = r.get("dc",{}) or {}
    ds = r.get("dataset",{}) or {}
    title = dc.get("title") or ""
    desc = dc.get("description") or ""
    main_id = dc.get("identifier") or ""
    dataset_id = ds.get("id") or ""
    owners = ds.get("owner",[]) or []
    owner_names = [(o.get("givenName") or "") + " " + (o.get("familyName") or "") for o in owners]
    owner_ids = [o.get("id") or "" for o in owners]
    authors = ds.get("author",[]) or []
    author_names = [(a.get("givenName") or "") + " " + (a.get("familyName") or "") for a in authors]
    versions = ds.get("versions",[]) or []
    version_ids = [v.get("versionIdentifier") or "" for v in versions]
    isabout = r.get("isAbout",[]) or []
    exp_app = r.get("experimental_approach",[]) or []
    prep = r.get("preparation",[]) or []
    sex_list = r.get("sex",[]) or []
    sex_vals = sex_list[:2]
    techniques = r.get("techniques",[]) or []
    rec_id = r.get("id") or ""
    urls = extract_urls(desc)
    chunk = "\n".join(filter(None, [
        title,
        clean_html(desc),
        safe_join(owner_names),
        safe_join(isabout),
        safe_join(exp_app),
        safe_join(prep),
        safe_join(sex_vals),
        safe_join(techniques)
    ]))
    meta = {
        "doi": r.get("doi") or "",
        "dataset_id": dataset_id,
        "owner_name": owner_names,
        "owner_id": owner_ids,
        "author": author_names,
        "versions": version_ids,
        "isAbout": isabout,
        "experimental_approach": exp_app,
        "preparation": prep,
        "sex": sex_vals,
        "techniques": techniques,
        "id": rec_id,
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE,
        "identifier": main_id
    }
    for i, u in enumerate(urls, start=1):
        meta[f"identifier{i}"] = u
    out.append({"chunk": chunk, "metadata_filters": meta})

print(json.dumps(out[0], ensure_ascii=False, indent=2))

ob,obl = OUTPUT_GCS_PATH.split("/",1)
client.bucket(ob).blob(obl).upload_from_string(
    json.dumps(out, ensure_ascii=False, indent=2),
    content_type="application/json"
)
print(f"Uploaded {len(out)} records to gs://{OUTPUT_GCS_PATH}")


""" {
  "chunk": "Overriding the impact of attentional probability learning by top-down control\nHere we tested the unique and combined effects of different attentional control (AC) mechanisms on attentional deployment. Experiments 1 and 4 assessed the impact of top-down control by using an endogenous cueing protocol with location-specific (Experiment 1) or region-specific (Experiment 4) valid (vs. neutral) cues. Experiment 2 investigated the influence of statistical learning (SL) by introducing an imbalance in target frequency across locations (high/intermediate/low). Experiments 3 and 5 evaluated the interaction between these two AC signals, by using location-specific (see Exp. 1) and region-specific (see Exp. 4) cueing, respectively. Results showed better performance following a valid (vs. neutral) cue and in the high (vs. low) frequency location, confirming the influence of both mechanisms on attentional guidance. However, when active together, top-down control seems to prevail over the biasing impact of SL, with the latter emerging only in neutral cue trials. By introducing a salient distractor, we also assessed the interfering effects of bottom-up AC signals which diverted attention from the target, regardless of the presence or absence of other AC mechanisms.\nElisa Santandrea\nvisual attention; attention\nbehavior\nin vivo\nmale; female\nintra-subject analysis",
  "metadata_filters": {
    "doi": "https://doi.org/10.25493/HR6S-103",
    "dataset_id": "https://kg.ebrains.eu/api/instances/840ab5b0-34f6-4af4-9b74-6445a5d78865",
    "owner_name": [
      "Elisa Santandrea"
    ],
    "owner_id": [
      "https://kg.ebrains.eu/api/instances/91c50366-e152-45da-89ff-ef40654742cc"
    ],
    "author": [
      "Carola Dolci",
      "Eleonora Baldini",
      "Suliann Ben Hamed",
      "C. Nico Boehler",
      "Emiliano Macaluso",
      "Leonardo Chelazzi",
      "Elisa Santandrea"
    ],
    "versions": [
      "v1.1",
      "v1"
    ],
    "isAbout": [
      "visual attention",
      "attention"
    ],
    "experimental_approach": [
      "behavior"
    ],
    "preparation": [
      "in vivo"
    ],
    "sex": [
      "male",
      "female"
    ],
    "techniques": [
      "intra-subject analysis"
    ],
    "id": "03248f0c-e4a9-453f-9589-372c9c7b24f9",
    "datasource_id": "scr_017612_ebrains",
    "datasource_name": "EBRAINS",
    "datasource_description": "A curated repository of EBRAINS datasets.",
    "datasource_type": "morphology",
    "identifier": "https://search.kg.ebrains.eu/instances/Dataset/03248f0c-e4a9-453f-9589-372c9c7b24f9"
  }
}
Uploaded 1541 records to gs://ks_datasets/preprocessed_data/scr_017612_ebrains.json """