from google.cloud import storage
import json
from bs4 import BeautifulSoup

# Configuration for the GENSAT datasource
INPUT_BLOB = "raw_dataset/data_sources/scr_002721_gensat_geneexpression.json"
OUTPUT_BLOB = "preprocessed_data/scr_002721_gensat_geneexpression.json"
DATASOURCE_ID = "scr_002721_gensat_geneexpression"
DATASOURCE_NAME = "GENSAT"
DATASOURCE_DESCRIPTION = "Contains gene expression data and maps of the mouse brain and spinal cord."
DATASOURCE_TYPE = "expression"

# Initialize GCS client and bucket
client = storage.Client()
bucket = client.bucket("ks_datasets")

def clean_html(html_str):
    return BeautifulSoup(html_str, "html.parser").get_text()

def extract_urls(html_str):
    soup = BeautifulSoup(html_str, "html.parser")
    return [a["href"] for a in soup.find_all("a")]

blob = bucket.blob(INPUT_BLOB)
raw_text = blob.download_as_text()
records = json.loads(raw_text)

processed = []

for rec in records:
    raw_desc = rec.get("dc", {}).get("description", "")
    description = clean_html(raw_desc)

    title              = rec.get("dc", {}).get("title", "")
    age                = rec.get("age", "")
    expr_pattern       = rec.get("expression_pattern", "")
    additional_subtype = rec.get("subtype_expanded", "")
    additional_info    = rec.get("additional_information", "")
    expression_level   = rec.get("expression_level", "")
    gene_name          = rec.get("gene_name", "")
    acq_technique      = rec.get("acquisition_technique", "")
    image_orient       = rec.get("image_orientation", "")
    cell_subtype       = rec.get("cell_subtype", "")
    structure_name     = rec.get("structure_name", "")
    stain              = rec.get("stain", "")
    section_proc       = rec.get("section_procedure", "")
    gene_symbol        = rec.get("gene_symbol", "")
    rec_id             = rec.get("id", "")
    identifier         = rec.get("dc", {}).get("identifier", "")

    parts = [
        f"Title: {title}",
        f"Description: {description}",
        f"Age: {age}",
        f"Expression Pattern: {expr_pattern}"
    ]
    if additional_subtype:
        parts.append(f"Additional Subtype: {additional_subtype}")
    parts += [
        f"Additional Info: {additional_info}",
        f"Expression Level: {expression_level}",
        f"Gene Name: {gene_name}",
        f"Acquisition Technique: {acq_technique}",
        f"Image Orientation: {image_orient}",
        f"Cell Subtype: {cell_subtype}",
        f"Structure Name: {structure_name}",
        f"Stain: {stain}",
        f"Section Procedure: {section_proc}",
        f"Gene Symbol: {gene_symbol}"
    ]
    chunk = "\n".join(parts)

    # Extract and dedupe URLs
    urls = extract_urls(raw_desc)
    unique_urls = []
    for url in urls:
        if url not in unique_urls:
            unique_urls.append(url)

    # Build metadata filters
    metadata_filters = {
        "structure": structure_name,
        "stain": stain,
        "age_group": age,
        "acquisition_technique": acq_technique,
        "geneid": rec.get("gene_id", ""),
        "gene_name": gene_name,
        "gene_symbol": gene_symbol,
        "expression_level": expression_level,
        "identifier": identifier,
        "id": rec_id,
        "cell_subtype": cell_subtype,
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE
    }
    for i, url in enumerate(unique_urls, start=1):
        metadata_filters[f"identifier{i}"] = url

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
  "chunk": "Title: adenylate cyclase activating polypeptide 1\nDescription: The BAC data is consistent with both literature and BGEM in situ data in general. However, the in situ data shows Adcyap1 expression in subiculum and postsubiculum area, which is not very obvious in the BAC data. At adult, the BAC data shows much less expression in pontine nuclei. In hypothalamus, the expression is primarily confined to the mammillary nuclei.\nAge: adult\nExpression Pattern: region-specific\nAdditional Info: The BAC data is consistent with both literature and BGEM in situ data in general. However, the in situ data shows Adcyap1 expression in subiculum and postsubiculum area, which is not very obvious in the BAC data. At adult, the BAC data shows much less expression in pontine nuclei. In hypothalamus, the expression is primarily confined to the mammillary nuclei.\nExpression Level: moderate to strong signal\nGene Name: adenylate cyclase activating polypeptide 1\nAcquisition Technique: brightfield\nImage Orientation: Sagittal\nCell Subtype: neuron\nStructure Name: Midbrain\nStain: DAB\nSection Procedure: cryostat\nGene Symbol: Adcyap1",
  "metadata_filters": {
    "structure": "Midbrain",
    "stain": "DAB",
    "age_group": "adult",
    "acquisition_technique": "brightfield",
    "geneid": "387",
    "gene_name": "adenylate cyclase activating polypeptide 1",
    "gene_symbol": "Adcyap1",
    "expression_level": "moderate to strong signal",
    "identifier": "http://www.gensat.org/GeneProgressTracker.jsp?gensatGeneID=387",
    "id": "389670",
    "cell_subtype": "neuron",
    "datasource_id": "scr_002721_gensat_geneexpression",
    "datasource_name": "GENSAT",
    "datasource_description": "Contains gene expression data and maps of the mouse brain and spinal cord.",
    "datasource_type": "expression"
  }
}
Preprocessed 198668 records and saved to gs://ks_datasets/preprocessed_data/scr_002721_gensat_geneexpression.json 
"""