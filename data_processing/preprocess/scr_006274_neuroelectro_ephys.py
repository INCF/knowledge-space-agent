import json,re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_006274_neuroelectro_ephys.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_006274_neuroelectro_ephys.json"
DATASOURCE_ID="scr_006274_neuroelectro_ephys"
DATASOURCE_NAME="NeuroElectro"
DATASOURCE_DESCRIPTION="A database of electrophysiological properties text-mined from the biomedical literature as a function of neuron type."
DATASOURCE_TYPE="physiology"

def clean_html(h):return BeautifulSoup(h or "","html.parser").get_text()
def extract_urls(t):return list(set(re.findall(r"https?://[^\s\"<>]+",t or "")))

client=storage.Client()

b,bl=INPUT_GCS_PATH.split("/",1)
recs=json.loads(client.bucket(b).blob(bl).download_as_text())
out=[]

for r in recs:
    dc=r.get("dc",{}) or {}
    title=dc.get("title","") or ""
    desc=dc.get("description","") or ""
    ident=dc.get("identifier","") or ""
    value_sd=r.get("value_sd","") or ""
    nelx_id=r.get("nelx_id","") or ""
    e_definition=r.get("e_definition","") or ""
    rec_id=r.get("id","") or ""
    n_name=r.get("n_name","") or ""
    e_name=r.get("e_name","") or ""
    property_name=r.get("property_name","") or ""
    urls=extract_urls(desc)
    chunk="\n".join(filter(None,[title,clean_html(desc),n_name,e_name,property_name,e_definition]))
    meta={
        "value_sd":value_sd,
        "nelx_id":nelx_id,
        "e_definition":e_definition,
        "id":rec_id,
        "n_name":n_name,
        "e_name":e_name,
        "property_name":property_name,
        "datasource_id":DATASOURCE_ID,
        "datasource_name":DATASOURCE_NAME,
        "datasource_description":DATASOURCE_DESCRIPTION,
        "datasource_type":DATASOURCE_TYPE,
        "identifier":ident
    }
    for i,u in enumerate(urls, start=1):meta[f"identifier{i}"]=u
    out.append({"chunk":chunk,"metadata_filters":meta})
    
print(json.dumps(out[0],ensure_ascii=False,indent=2))

ob,obl=OUTPUT_GCS_PATH.split("/",1)
client.bucket(ob).blob(obl).upload_from_string(json.dumps(out,ensure_ascii=False,indent=2),"application/json")

print(f"Uploaded {len(out)} records to gs://{OUTPUT_GCS_PATH}")


""" 
{
  "chunk": "Substantia nigra pars compacta dopaminergic cell\nMaximum rate of rise of membrane voltage during spike falling phase\nSubstantia nigra pars compacta dopaminergic cell\nspike max decay slope\nspike max decay slope\nMaximum rate of rise of membrane voltage during spike falling phase",
  "metadata_filters": {
    "value_sd": "6.0",
    "nelx_id": "nifext_145",
    "e_definition": "Maximum rate of rise of membrane voltage during spike falling phase",
    "id": "33",
    "n_name": "Substantia nigra pars compacta dopaminergic cell",
    "e_name": "spike max decay slope",
    "property_name": "spike max decay slope",
    "datasource_id": "scr_006274_neuroelectro_ephys",
    "datasource_name": "NeuroElectro",
    "datasource_description": "A database of electrophysiological properties text-mined from the biomedical literature as a function of neuron type.",    
    "datasource_type": "physiology",
    "identifier": "https://neuroelectro.org/neuron/183"
  }
}
Uploaded 48 records to gs://ks_datasets/preprocessed_data/scr_006274_neuroelectro_ephys.json
"""