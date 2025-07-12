import json,re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_014306_bbp_cellmorphology.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_014306_bbp_cellmorphology.json"
DATASOURCE_ID="scr_014306_bbp_cellmorphology"
DATASOURCE_NAME="Blue Brain Project Cell Morphology"
DATASOURCE_DESCRIPTION="3D Models of rat neuronal morphologies"
DATASOURCE_TYPE="morphology"

def clean_html(h):
    return BeautifulSoup(h or "","html.parser").get_text()

def extract_urls(t):
    return list(set(re.findall(r"https?://[^\s\"<>]+",t or "")))

def safe_join(l,sep="; "):
    return sep.join(str(x).strip() for x in (l if isinstance(l,list) else [l]) if isinstance(x,str) and x.strip())

client=storage.Client()

b,bl=INPUT_GCS_PATH.split("/",1)
recs=json.loads(client.bucket(b).blob(bl).download_as_text())
out=[]

for r in recs:
    dc=r.get("dc",{}) or {}
    title=dc.get("title","") or ""
    subjects=dc.get("subject",[]) or []
    fn=r.get("file_name","") or ""
    fn_link=r.get("file_name_link","") or ""
    cell_term=r.get("cell_term","") or ""
    region=r.get("region_term","") or ""
    url=r.get("url","") or ""
    dts=r.get("dataItem",{}).get("dataTypes",[]) or []
    ident=dc.get("identifier","") or ""
    chunk="\n".join(filter(None,[title,safe_join(subjects),cell_term,region]))
    meta={
        "filename":fn,
        "cell_term":cell_term,
        "file_name_link":fn_link,
        "region_term":region,
        "url":url,
        "datasource_id":DATASOURCE_ID,
        "datasource_name":DATASOURCE_NAME,
        "datasource_description":DATASOURCE_DESCRIPTION,
        "datasource_type":DATASOURCE_TYPE,
        "identifier":ident
    }
    urls=extract_urls(fn_link)+([url] if url else [])
    for i,u in enumerate(urls, start=1):meta[f"identifier{i}"]=u
    out.append({"chunk":chunk,"metadata_filters":meta})
    
print(json.dumps(out[0],ensure_ascii=False,indent=2))

ob,obl=OUTPUT_GCS_PATH.split("/",1)
client.bucket(ob).blob(obl).upload_from_string(json.dumps(out,ensure_ascii=False,indent=2),"application/json")
print(f"Uploaded {len(out)} records to gs://{OUTPUT_GCS_PATH}")

""" 
{
  "chunk": "C230501A4\nNeocortex Layer IV Martinotti Cell\nNeocortex Layer IV Martinotti Cell",
  "metadata_filters": {
    "filename": "C230501A4",
    "cell_term": "Neocortex Layer IV Martinotti Cell",
    "file_name_link": "<a class=\"external\" target=\"_blank\" href=\"http://microcircuits.epfl.ch/#/animal/e511ac26-b806-11e4-bce1-6003088da632\">C230501A4</a>",
    "region_term": "",
    "url": "http://microcircuits.epfl.ch/#/animal/e511ac26-b806-11e4-bce1-6003088da632",
    "datasource_id": "scr_014306_bbp_cellmorphology",
    "datasource_name": "Blue Brain Project Cell Morphology",
    "datasource_description": "3D Models of rat neuronal morphologies",
    "datasource_type": "morphology",
    "identifier": "http://microcircuits.epfl.ch/#/animal/e511ac26-b806-11e4-bce1-6003088da632",
    "identifier1": "http://microcircuits.epfl.ch/#/animal/e511ac26-b806-11e4-bce1-6003088da632",
    "identifier2": "http://microcircuits.epfl.ch/#/animal/e511ac26-b806-11e4-bce1-6003088da632"
  }
}
Uploaded 1241 records to gs://ks_datasets/preprocessed_data/scr_014306_bbp_cellmorphology.json """