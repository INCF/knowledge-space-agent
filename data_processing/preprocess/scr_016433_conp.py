import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_016433_conp.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_016433_conp.json"
DATASOURCE_ID="scr_016433_conp"
DATASOURCE_NAME="CONP"
DATASOURCE_TYPE="dataset"

def clean_html(html):
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text):
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst, sep="; "):
    return sep.join(str(x).strip() for x in (lst if isinstance(lst, list) else [lst]) if isinstance(x, str) and x.strip())

client=storage.Client()
bucket,blob=INPUT_GCS_PATH.split("/",1)
records=json.loads(client.bucket(bucket).blob(blob).download_as_text())
processed=[]
for rec in records:
    dc=rec.get("dc",{}) or {}
    title=dc.get("title","") or ""
    desc=dc.get("description","") or ""
    ident=dc.get("identifier","") or ""
    depr=rec.get("depricated",False)
    ver=rec.get("version","") or ""
    src=rec.get("source_git_url","") or ""
    kws=rec.get("keywords",[]) or []
    lic=rec.get("license","") or ""
    creators=rec.get("creators",[]) or []
    urls=extract_urls(desc)
    chunk="\n".join([title,clean_html(desc),str(depr),safe_join(kws),lic,safe_join(creators)])
    meta={"creators":creators,"license":lic,"keywords":kws,"source_git_url":src,"depricated":depr,"version":ver,"identifier":ident,"datasource_id":DATASOURCE_ID,"datasource_name":DATASOURCE_NAME,"datasource_type":DATASOURCE_TYPE}
    for i,u in enumerate(urls, start=1):
        meta[f"identifier{i}"]=u
    processed.append({"chunk":chunk,"metadata_filters":meta})

print(json.dumps(processed[0], ensure_ascii=False, indent=2))
out_bucket,out_blob=OUTPUT_GCS_PATH.split("/",1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(json.dumps(processed,ensure_ascii=False,indent=2),"application/json")
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


""" 
{
  "chunk": "Comparing Perturbation Modes for Evaluating Instabilities in Neuroimaging: Processed NKI-RS Subset (08/2019)\nThe processed subset of the NKI-RS dataset for evaluation of various perturbation modes when studying instabilities. Linked to the pre-print found here, can be visualized using the plotting code here, and generated with various scripts and launch configurations found here.\nFalse\nconnectomes; canadian-open-neuroscience-platform; dataset; neuroimaging\nCC-BY-4.0\nKiar, Gregory",
  "metadata_filters": {
    "creators": [
      "Kiar, Gregory"
    ],
    "license": "CC-BY-4.0",
    "keywords": [
      "connectomes",
      "canadian-open-neuroscience-platform",
      "dataset",
      "neuroimaging"
    ],
    "source_git_url": "https://api.github.com/repos/conp-bot/conp-dataset-Comparing-Perturbation-Modes-for-Evaluating-Instabilities-in-Neuroimaging-Processed-NK/git/blobs/dcf9eee604b681ace36137f819027d6ee9c127e9",
    "depricated": false,
    "version": "None",
    "identifier": "http://github.com/conp-bot/conp-dataset-Comparing-Perturbation-Modes-for-Evaluating-Instabilities-in-Neuroimaging-Processed-NK",
    "datasource_id": "scr_016433_conp",
    "datasource_name": "CONP",
    "datasource_type": "dataset",
    "identifier1": "https://github.com/gkiar/stability-mca/blob/master/code/dipy_exploratory/mca_dipy_exploratory_analysis.ipynb",   
    "identifier2": "https://github.com/gkiar/stability/tree/master/code/experiments/paper0_comparing_perturbation_modes",
    "identifier3": "https://arxiv.org/abs/1908.10922"
  }
}
Uploaded 55 records to gs://ks_datasets/preprocessed_data/scr_016433_conp_preprocessed.json """