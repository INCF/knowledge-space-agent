import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_002978_aba_expression.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_002978_aba_expression.json"
DATASOURCE_ID="scr_002978_aba_expression"
DATASOURCE_NAME="Brain Atlas Mouse Brain - Expression"
DATASOURCE_DESCRIPTION="A genome_wide database of gene expression in the mouse brain."
DATASOURCE_TYPE="expression"

def clean_html(html):
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text):
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

client=storage.Client()
bucket,blob=INPUT_GCS_PATH.split("/",1)
records=json.loads(client.bucket(bucket).blob(blob).download_as_text())
processed=[]

for rec in records:
    dc=rec.get("dc",{}) or {}
    title=dc.get("title","") or ""
    desc=dc.get("description","") or ""
    datasource=rec.get("datasource","") or ""
    structure_name=rec.get("structure_name","") or ""
    gene_symbol=rec.get("gene_symbol","") or ""
    expression_level=rec.get("expression_level","") or ""
    structure_label=rec.get("structure_label","") or ""
    species=rec.get("species","") or ""
    expression_density=rec.get("expression_density","") or ""
    gene_name=rec.get("gene_name","") or ""
    allen_id=rec.get("allen_id","") or ""
    ident=dc.get("identifier","") or ""
    urls=extract_urls(desc)
    chunk="\n".join([title,clean_html(desc),datasource,structure_name,gene_symbol,expression_level,structure_label,species,expression_density,gene_name])
    meta={"structure_name":structure_name,"gene_symbol":gene_symbol,"gene_id":rec.get("gene_id",""),"structure_label":structure_label,"expression_level":expression_level,"species":species,"expression_density":expression_density,"allen_id":allen_id,"gene_name":gene_name,"datasource_id":DATASOURCE_ID,"datasource_name":DATASOURCE_NAME,"datasource_description":DATASOURCE_DESCRIPTION,"datasource_type":DATASOURCE_TYPE,"identifier":ident}
    for i,u in enumerate(urls, start=1):
        meta[f"identifier{i}"]=u
        
    processed.append({"chunk":chunk,"metadata_filters":meta})

print(json.dumps(processed[0], ensure_ascii=False, indent=2))

out_bucket,out_blob=OUTPUT_GCS_PATH.split("/",1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(json.dumps(processed,ensure_ascii=False,indent=2),"application/json")

print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")

""" 
{
  "chunk": "expressed sequence AI987712\nBrain region:Cerebral cortex, Gene Symbol:AI987712, Organism:Mouse, Species:Mouse\nAllen Brain Atlas Mouse Brain\nCerebral cortex\nAI987712\n3\nCTX\nMouse\n2\nexpressed sequence AI987712",
  "metadata_filters": {
    "structure_name": "Cerebral cortex",
    "gene_symbol": "AI987712",
    "gene_id": "NCBIGene:105935",
    "structure_label": "CTX",
    "expression_level": "3",
    "species": "Mouse",
    "expression_density": "2",
    "allen_id": "70100",
    "gene_name": "expressed sequence AI987712",
    "datasource_id": "scr_002978_aba_expression",
    "datasource_name": "Brain Atlas Mouse Brain - Expression",
    "datasource_description": "A genome_wide database of gene expression in the mouse brain.",
    "datasource_type": "expression",
    "identifier": "http://mouse.brain-map.org/gene/show/70100"
  }
}
Uploaded 330562 records to gs://ks_datasets/preprocessed_data/scr_002978_aba_expression.json"""