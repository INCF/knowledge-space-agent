from google.cloud import storage
import json
from bs4 import BeautifulSoup

# Initialize GCS client and define paths
BUCKET_NAME = "ks_datasets"
INPUT_BLOB_PATH = "raw_dataset/data_sources/scr_003105_neurondb_currents.json"
OUTPUT_BLOB_PATH = "preprocessed_data/scr_003105_neurondb_currents.json"

client = storage.Client()
bucket = client.bucket(BUCKET_NAME)


def clean_html(html_str):
    return BeautifulSoup(html_str, "html.parser").get_text()

def extract_urls(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    return [a["href"] for a in soup.find_all("a")]

blob = bucket.blob(INPUT_BLOB_PATH)
raw_text = blob.download_as_text()
records = json.loads(raw_text)

processed = []

datasource_id = "scr_003105_neurondb_currents"
datasource_name = "NeuronDB"
datasource_description = "Provides data about neurotransmitter properties for submitted neurons"
datasource_type = "physiology"

for rec in records:
    raw_desc = rec.get("dc", {}).get("description", "")
    raw_ref  = rec.get("reference_note", "")

    subject        = "; ".join(rec.get("dc", {}).get("subject", []))
    description    = clean_html(raw_desc)
    neuron         = rec.get("neuron", "")
    current        = rec.get("current", "")
    compartment    = rec.get("compartment", "")
    connect_note   = rec.get("connect_note", "")
    reference_note = clean_html(raw_ref)
    rec_id         = rec.get("id", "")
    identifier     = rec.get("dc", {}).get("identifier", "")

    chunk = "\n".join([
        f"Subject: {subject}",
        f"Description: {description}",
        f"Neuron: {neuron}",
        f"Current: {current}",
        f"Compartment: {compartment}",
        f"Connection Note: {connect_note}",
        f"Reference: {reference_note}"
    ])

    # Extract and dedupe URLs
    urls = extract_urls(raw_desc) + extract_urls(raw_ref)
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)

    # Build metadata filters including identifier fields
    metadata_filters = {
        "neuron": neuron,
        "compartment": compartment,
        "current": current,
        "id": rec_id,
        "identifier": identifier,
        "datasource_id": datasource_id,
        "datasource_name": datasource_name,
        "datasource_description": datasource_description,
        "datasource_type": datasource_type
    }
    # Add extracted URLs as identifier1, identifier2, ...
    for idx, url in enumerate(unique_urls, start=1):
        metadata_filters[f"identifier{idx}"] = url

    processed.append({"chunk": chunk, "metadata_filters": metadata_filters})

# Printing a sample
print(json.dumps(processed[0], indent=2))

# Saving all preprocessed records back to GCS
out_blob = bucket.blob(OUTPUT_BLOB_PATH)
out_blob.upload_from_string(
    data=json.dumps(processed, ensure_ascii=False),
    content_type="application/json"
)
print(f"Preprocessed data saved to gs://{BUCKET_NAME}/{OUTPUT_BLOB_PATH}")


""" 
{
  "chunk": "Subject: Neocortex V1 pyramidal corticothalamic L6 cell\nDescription: Many authors have described the activation of dendritic voltage activated Ca channels (58).\nNeuron: Neocortex V1 pyramidal corticothalamic L6 cell\nCurrent: I p,q\nCompartment: Proximal apical dendrite\nConnection Note: \nReference: Many authors have described the activation of dendritic voltage activated Ca channels (58).",
  "metadata_filters": {
    "neuron": "Neocortex V1 pyramidal corticothalamic L6 cell",
    "compartment": "Proximal apical dendrite",
    "current": "I p,q",
    "id": "1161",
    "identifier": "https://senselab.med.yale.edu/NeuronDB/NeuronProp.aspx?mo=4&pr=C&id=265",
    "datasource_id": "scr_003105_neurondb_currents",
    "datasource_name": "NeuronDB",
    "datasource_description": "Provides data about neurotransmitter properties for submitted neurons",
    "datasource_type": "physiology",
    "identifier1": "https://senselab.med.yale.edu/NeuronDB/references/refData.aspx?r=58"
  }
}
Preprocessed data saved to gs://ks_datasets/preprocessed_data/scr_003105_neurondb_currents.json
"""