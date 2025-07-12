from google.cloud import storage
import json

# GCS paths and datasource metadata
INPUT_BLOB = "raw_dataset/data_sources/scr_014194_icg_ionchannels.json"
OUTPUT_BLOB = "preprocessed_data/scr_014194_icg_ionchannels.json"
DATASOURCE_ID = "scr_014194_icg_ionchannels"
DATASOURCE_NAME = "IonChannelGenealogy"
DATASOURCE_DESCRIPTION = "Provides a quantitative assay of publicly available ion channel models."
DATASOURCE_TYPE = "models"

# Initialize GCS client
client = storage.Client()
bucket = client.bucket("ks_datasets")

blob = bucket.blob(INPUT_BLOB)
records = json.loads(blob.download_as_text())

processed = []

for rec in records:
    dc           = rec.get("dc", {})
    title        = dc.get("title", "")
    description  = dc.get("description", "")
    name         = rec.get("name", "")
    comments     = rec.get("comments", "")
    subtype      = rec.get("subtype", "")
    temperature  = rec.get("temperature", "")
    def join_list(field): return "; ".join(rec.get(field, [])) if isinstance(rec.get(field, []), list) else rec.get(field, "")
    brain_area    = join_list("brain_area")
    neuron_region = join_list("neuron_region")
    neuron_type   = join_list("neuron_type")
    age           = join_list("age")

    parts = [
        f"Title: {title}",
        f"Description: {description}",
        f"Name: {name}",
        f"Comments: {comments}",
        f"Subtype: {subtype}",
        f"Temperature: {temperature}",
        f"Brain Area: {brain_area}",
        f"Neuron Region: {neuron_region}",
        f"Neuron Type: {neuron_type}",
        f"Age: {age}"
    ]
    chunk = "\n".join(parts)

    # Collect all URL-like fields
    urls = []
    for key in [dc.get("identifier"), rec.get("channel_url"), rec.get("pmid_link")]:
        if key and isinstance(key, str) and key.startswith("http"):
            urls.append(key)
    # Remove duplicates preserving order
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)

    rec_id = rec.get("id", "")
    metadata_filters = {
        "subtype": subtype,
        "brain_area": brain_area,
        "animal_model": join_list("animal_model"),
        "neuron_region": neuron_region,
        "neuron_type": neuron_type,
        "age": age,
        "id": rec_id,
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE
    }
    if dc.get("identifier"):
        metadata_filters["identifier"] = dc.get("identifier")
    for idx, url in enumerate(unique_urls, start=1):
        metadata_filters[f"identifier{idx}"] = url

    processed.append({"chunk": chunk, "metadata_filters": metadata_filters})

# Printing a sample
if processed:
    print(json.dumps(processed[0], indent=2))

# Saving all preprocessed records back to GCS
out_blob = bucket.blob(OUTPUT_BLOB)
out_blob.upload_from_string(
    data=json.dumps(processed, ensure_ascii=False),
    content_type="application/json"
)
print(f"Preprocessed {len(processed)} records and saved to gs://ks_datasets/{OUTPUT_BLOB}")

""" 
{
  "chunk": "Title: Channel 137259-cal2. CA3\nDescription: ICG id: 984L-type calcium channel. Ancestor model from hemond et al. (2008), model no. 101629, also for ca3 pyramidal neuron. This model is identical to ancestor model. Modeling study that uses experimental results from hemond et al. (2008, 2009) and perez-rosello et al. (2011). Animal model here reflects the data in these experimental studies. This channel was adapted from model in hemond et al. (2008), model no. 101629.. Keywords: [soma,  dendrites], [pyramidal cell], [Sprague-Dawley,  rat]. \nName: cal2\nComments: L-type calcium channel. Ancestor model from hemond et al. (2008), model no. 101629, also for ca3 pyramidal neuron. This model is identical to ancestor model. Modeling study that uses experimental results from hemond et al. (2008, 2009) and perez-rosello et al. (2011). Animal model here reflects the data in these experimental studies. This channel was adapted from model in hemond et al. (2008), model no. 101629.\nSubtype: L-type\nTemperature: Model fitting was done using experimental characterizations at near physiological temperatures (34-35 deg C)\nBrain Area: CA3;  hippocampus\nNeuron Region: soma;  dendrites\nNeuron Type: pyramidal cell\nAge: 4-6 weeks;  23+/-5 days",
  "metadata_filters": {
    "subtype": "L-type",
    "brain_area": "CA3;  hippocampus",
    "animal_model": "Sprague-Dawley;  rat",
    "neuron_region": "soma;  dendrites",
    "neuron_type": "pyramidal cell",
    "age": "4-6 weeks;  23+/-5 days",
    "id": "984",
    "datasource_id": "scr_014194_icg_ionchannels",
    "datasource_name": "IonChannelGenealogy",
    "datasource_description": "Provides a quantitative assay of publicly available ion channel models.",
    "datasource_type": "models",
    "identifier": "http://icg.neurotheory.ox.ac.uk/channels/1973/984",
    "identifier1": "http://icg.neurotheory.ox.ac.uk/channels/1973/984",
    "identifier2": "http://www.ncbi.nlm.nih.gov/pubmed/21191641"
  }
}
Preprocessed 2827 records and saved to gs://ks_datasets/preprocessed_data/scr_014194_icg_ionchannels.json 
"""