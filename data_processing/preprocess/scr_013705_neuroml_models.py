import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH          = "ks_datasets/raw_dataset/data_sources/scr_013705_neuroml_models.json"
OUTPUT_GCS_PATH         = "ks_datasets/preprocessed_data/scr_013705_neuroml_models.json"
DATASOURCE_ID           = "scr_013705_neuroml_models"
DATASOURCE_NAME         = "NeuroML Database"
DATASOURCE_DESCRIPTION  = (
    "Curated relational database that provides for the storage and retrieval of computational neuroscience model."
)
DATASOURCE_TYPE         = "models"

def clean_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join(str(x).strip() for x in lst if isinstance(x, str) and x.strip())

def preprocess_record(rec: dict) -> dict:
    model_id             = rec.get("model_id", "")
    model_name           = rec.get("model_name", "")
    model_type           = rec.get("model_type", "")
    pubmed_title         = rec.get("pubmed_title", "")
    authors              = rec.get("authors", []) or []
    neurolex_terms       = rec.get("neurolex_terms", []) or []
    keywords             = rec.get("keywords", []) or []
    children_model_names = rec.get("children_model_name", []) or []

    dc                   = rec.get("dc", {}) or {}
    title                = dc.get("title", "")
    description          = dc.get("description", "")
    main_identifier      = dc.get("identifier", "")

    parts = [
        model_name,
        model_type,
        pubmed_title,
        safe_join(neurolex_terms),
        title,
        clean_html(description),
        safe_join(keywords),
        safe_join(children_model_names),
    ]
    chunk = "\n".join(p for p in parts if p)

    meta = {
        "model_id": model_id,
        "model_name": model_name,
        "model_type": model_type,
        "pubmed_title": pubmed_title,
        "authors": authors,
        "neurolex_terms": neurolex_terms,
        "children_model_name": children_model_names,
        "keywords": keywords,
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE,
    }

    if main_identifier:
        meta["identifier"] = main_identifier

    urls = extract_urls(description) + extract_urls(title)
    seen = set()
    unique_urls = []
    for u in urls:
        if u and u != main_identifier and u not in seen:
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
  "chunk": "Layer 5 Burst Accommodating Martinotti Cell (2)\nCell\nReconstruction and Simulation of Neocortical Microcircuitry.\nMartinotti Cell; Neocortex Layer 5\nNeuroML Database: NMLCL000158- Layer 5 Burst Accommodating Martinotti Cell (2)\nID:NMLCL000158. Type: Cell. Keywords: burst accommodating,Layer 5,GABAergic, inhibitory, neocortex, rat, somatosensory cortex, Martinotti, biophysical, burst accommodating martinotti cell, Realistic Network, Neuron or other electrically excitable cell, Synapse, Spatio-temporal Activity Patterns, Detailed Neuronal Models Publication: Reconstruction and Simulation of Neocortical Microcircuitry.\nburst accommodating; Layer 5; GABAergic; inhibitory; neocortex; rat; somatosensory cortex; Martinotti; biophysical; burst accommodating martinotti cell; Realistic Network; Neuron or other electrically excitable cell; Synapse; Spatio-temporal Activity Patterns; Detailed Neuronal Models\nCaHVA High Voltage Activated Calcium; CaLVA Low Voltage Activated Calcium; Ih Hyperpolarization Activated Cation; IM M Type Potassium; KPst Slow Inactivating Potassium; KTst Fast Inactivating Potassium; NaP Persistent Sodium; NaTa Fast Inactivating Sodium; NaTs Fast Inactivating Sodium; KCa SK Type Calcium Dependent Potassium; K Fast Noninactivating Potassium; Passive Leak; Calcium Pool Dynamics",
  "metadata_filters": {
    "model_id": "NMLCL000158",
    "model_name": "Layer 5 Burst Accommodating Martinotti Cell (2)",
    "model_type": "Cell",
    "pubmed_title": "Reconstruction and Simulation of Neocortical Microcircuitry.",
    "authors": [
      "H Markram",
      "Eilif Muller",
      "Srikanth Ramaswamy",
      "Michael Reimann",
      "Marwan Abdellah",
      "Carlos Sanchez",
      "Anastasia Ailamaki",
      "Lidia Alonso-Nanclares",
      "Nicolas Antille",
      "Selim Arsever",
      "Guy Kahou",
      "Thomas Berger",
      "Ahmet Bilgili",
      "Nenad Buncic",
      "Athanassia Chalimourda",
      "Giuseppe Chindemi",
      "Jean-Denis Courcol",
      "Fabien Delalondre",
      "Vincent Delattre",
      "Shaul Druckmann",
      "Raphael Dumusc",
      "James Dynes",
      "Stefan Eilemann",
      "Eyal Gal",
      "Michael Gevaert",
      "Jean-Pierre Ghobril",
      "Albert Gidon",
      "Joe Graham",
      "Anirudh Gupta",
      "Valentin Haenel",
      "E Hay",
      "Thomas Heinis",
      "Juan Hernando",
      "Michael Hines",
      "Lida Kanari",
      "Daniel Keller",
      "John Kenyon",
      "Georges Khazen",
      "Yihwa Kim",
      "James King",
      "Zoltan Kisvarday",
      "Pramod Kumbhar",
      "Jean-Vincent Le Bé",
      "Bruno Magalhães",
      "Angel Merchán-Pérez",
      "Julie Meystre",
      "Benjamin Morrice",
      "Jeffrey Muller",
      "Alberto Muñoz-Céspedes",
      "Shruti Muralidhar",
      "Keerthan Muthurasa",
      "Daniel Nachbaur",
      "Taylor Newton",
      "Max Nolte",
      "Aleksandr Ovcharenko",
      "Juan Palacios",
      "Luis Pastor",
      "Rodrigo Perin",
      "Rajnish Ranjan",
      "Imad Riachi",
      "Juan Riquelme",
      "Christian Rössert",
      "Konstantinos Sfyrakis",
      "Ying Shi",
      "Julian Shillcock",
      "Gilad Silberberg",
      "Ricardo Silva",
      "Farhan Tauheed",
      "Martin Telefont",
      "Maria Toledo-Rodriguez",
      "Thomas Tränkler",
      "Werner Van Geit",
      "Jafet Diaz",
      "Richard Walker",
      "Yun Wang",
      "Stefano Zaninetta",
      "Javier DeFelipe",
      "S Hill",
      "Idan Segev",
      "F Schürmann",
      "Padraig Gleeson"
    ],
    "neurolex_terms": [
      "Martinotti Cell",
      "Neocortex Layer 5"
    ],
    "children_model_name": [
      "CaHVA High Voltage Activated Calcium",
      "CaLVA Low Voltage Activated Calcium",
      "Ih Hyperpolarization Activated Cation",
      "IM M Type Potassium",
      "KPst Slow Inactivating Potassium",
      "KTst Fast Inactivating Potassium",
      "NaP Persistent Sodium",
      "NaTa Fast Inactivating Sodium",
      "NaTs Fast Inactivating Sodium",
      "KCa SK Type Calcium Dependent Potassium",
      "K Fast Noninactivating Potassium",
      "Passive Leak",
      "Calcium Pool Dynamics"
    ],
    "keywords": [
      "burst accommodating",
      "Layer 5",
      "GABAergic",
      " inhibitory",
      " neocortex",
      " rat",
      " somatosensory cortex",
      " Martinotti",
      " biophysical",
      " burst accommodating martinotti cell",
      " Realistic Network",
      " Neuron or other electrically excitable cell",
      " Synapse",
      " Spatio-temporal Activity Patterns",
      " Detailed Neuronal Models"
    ],
    "datasource_id": "scr_013705_neuroml_models",
    "datasource_name": "NeuroML Database",
    "datasource_description": "Curated relational database that provides for the storage and retrieval of computational neuroscience model.",
    "datasource_type": "models",
    "identifier": "https://neuroml-db.org/model_info?model_id=NMLCL000158"
  }
}
Uploaded 1586 records to gs://ks_datasets/preprocessed_data/scr_013705_neuroml_models.json"""