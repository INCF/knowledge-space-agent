from google.cloud import storage
import json

INPUT_BLOB = "raw_dataset/data_sources/scr_006131_hba_atlas.json"
OUTPUT_BLOB = "preprocessed_data/scr_006131_hba_atlas.json"

# Initialize GCS client and bucket
client = storage.Client()
bucket = client.bucket("ks_datasets")

blob = bucket.blob(INPUT_BLOB)
records = json.loads(blob.download_as_text())

processed = []
for rec in records:
    subject      = rec.get("subject", "")
    brain_region = rec.get("brain_region", "")
    brain_view   = rec.get("brain_view", "")
    feature      = rec.get("feature", "")
    species      = rec.get("species", "")
    rec_id       = rec.get("id", "")

    chunk = "\n".join([
        f"Subject: {subject}",
        f"Brain Region: {brain_region}",
        f"Brain View: {brain_view}",
        f"Feature: {feature}",
        f"Species: {species}",
        f"ID: {rec_id}"
    ])

    # Collect URLs: dc.identifier and image_url only
    urls = []
    dc_id = rec.get("dc", {}).get("identifier", "")
    if dc_id.startswith("http"):
        urls.append(dc_id)
    img_url = rec.get("image_url", "")
    if img_url.startswith("http"):
        urls.append(img_url)

    metadata_filters = {
        "brain_region": brain_region,
        "id": rec_id,
        "brain_view": brain_view,
        "species": species,
        "feature": feature
    }
    # Add URL identifiers
    if urls:
        metadata_filters["identifier"]  = urls[0]
    if len(urls) > 1:
        metadata_filters["identifier1"] = urls[1]

    processed.append({
        "chunk": chunk,
        "metadata_filters": metadata_filters
    })

# Printing sample
if processed:
    print(json.dumps(processed[0], indent=2))

# Upload preprocessed data to GCS
out_blob = bucket.blob(OUTPUT_BLOB)
out_blob.upload_from_string(
    json.dumps(processed, ensure_ascii=False),
    content_type="application/json"
)
print(f"Preprocessed {len(processed)} records and saved to gs://ks_datasets/{OUTPUT_BLOB}")


"""
    {
  "chunk": "Subject: human\nBrain Region: middle cerebellar peduncle\nBrain View: sagittal\nFeature: mri\nSpecies: human\nID: 0572middle cerebellar pedunclemrisagittal",
  "metadata_filters": {
    "brain_region": "middle cerebellar peduncle",
    "id": "0572middle cerebellar pedunclemrisagittal",
    "brain_view": "sagittal",
    "species": "human",
    "feature": "mri",
    "identifier": "http://kobiljak.msu.edu/CAI/NOP552/Med_Neuroscience/humanatlas/sagittal/0572_mri_labelled.html",
    "identifier1": "http://www.msu.edu/~brains/brains/human/sagittal/0572_mri.jpg"
  }
}
Preprocessed 3982 records and saved to gs://ks_datasets/preprocessed_data/scr_006131_hba_atlas.json
    """