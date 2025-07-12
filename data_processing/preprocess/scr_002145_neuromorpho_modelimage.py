import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH="ks_datasets/raw_dataset/data_sources/scr_002145_neuromorpho_modelimage.json"
OUTPUT_GCS_PATH="ks_datasets/preprocessed_data/scr_002145_neuromorpho_modelimage.json"
DATASOURCE_ID="scr_002145_neuromorpho_modelimage"
DATASOURCE_NAME="NeuroMorpho"
DATASOURCE_DESCRIPTION="A curated repository of digitally reconstructed neurons."
DATASOURCE_TYPE="morphology"

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
    surface=rec.get("surface","") or ""
    volume=rec.get("volume","") or ""
    rec_id=rec.get("id","") or ""
    brain_region=rec.get("brain_region","") or ""
    age=rec.get("age","") or ""
    gender=rec.get("gender","") or ""
    staining_method=rec.get("staining_method","") or ""
    scientific_name=rec.get("scientific_name","") or ""
    soma_surface=rec.get("soma_surface","") or ""
    neuron_name=rec.get("neuron_name","") or ""
    species=rec.get("species","") or ""
    pmid_url=rec.get("pmid_url","") or ""
    png_url=rec.get("png_url","") or ""
    image_url=rec.get("image_url","") or ""
    neuron_id=rec.get("neuron_id","") or ""
    min_weight=rec.get("min_weight","") or ""
    cell_class=rec.get("cell_class","") or ""
    strain_name=rec.get("strain_name","") or ""
    title=dc.get("title","") or ""
    description=dc.get("description","") or ""
    ident=dc.get("identifier","") or ""
    note=rec.get("note","") or ""
    urls=extract_urls(description)
    chunk="\n".join([
        brain_region,
        surface,
        volume,
        title,
        clean_html(description),
        note,
        gender,
        age,
        staining_method,
        min_weight,
        scientific_name,
        soma_surface,
        neuron_name,
        rec.get("expercond","") or "",
        species,
        cell_class,
        strain_name,
        neuron_id
    ])
    meta={
        "surface":surface,
        "volume":volume,
        "id":rec_id,
        "brain_region":brain_region,
        "age":age,
        "gender":gender,
        "staining_method":staining_method,
        "scientific_name":scientific_name,
        "soma_surface":soma_surface,
        "neuron_name":neuron_name,
        "species":species,
        "pmid_url":pmid_url,
        "png_url":png_url,
        "image_url":image_url,
        "neuron_id":neuron_id,
        "min_weight":min_weight,
        "cell_class":cell_class,
        "strain_name":strain_name,
        "datasource_id":DATASOURCE_ID,
        "datasource_name":DATASOURCE_NAME,
        "datasource_description":DATASOURCE_DESCRIPTION,
        "datasource_type":DATASOURCE_TYPE,
        "identifier":ident
    }
    for i,u in enumerate(urls, start=1):
        meta[f"identifier{i}"]=u
    processed.append({"chunk":chunk,"metadata_filters":meta})

print(json.dumps(processed[0], ensure_ascii=False, indent=2))

out_bucket,out_blob=OUTPUT_GCS_PATH.split("/",1)
client.bucket(out_bucket).blob(out_blob).upload_from_string(
    json.dumps(processed, ensure_ascii=False, indent=2),
    content_type="application/json"
)
print(f"Uploaded {len(processed)} records to gs://{OUTPUT_GCS_PATH}")


""" 
{
  "chunk": "hypothalamus, anterior, ventrolateral preoptic nucleus\n4426.43\n1407.05\nDetailed Neuron Information: VLPO-29, Strain:C57BL/6J\nNeuroMorpho.Org ID: NMO_67811. Species: mouse,. Gender: Male. \n\nMale\n14-19 days, young\nbiocytin\n\n\n24556.1\nVLPO-29\nControl\nmouse\nprincipal cell, low-threshold calcium spiking, noradrenaline-inhibited\nC57BL/6J\nNMO_67811",
  "metadata_filters": {
    "surface": "4426.43",
    "volume": "1407.05",
    "id": "67811",
    "brain_region": "hypothalamus, anterior, ventrolateral preoptic nucleus",
    "age": "14-19 days, young",
    "gender": "Male",
    "staining_method": "biocytin",
    "scientific_name": "",
    "soma_surface": "24556.1",
    "neuron_name": "VLPO-29",
    "species": "mouse",
    "pmid_url": "http://www.ncbi.nlm.nih.gov/pubmed/26755200",
    "png_url": "http://neuromorpho.org/images/imageFiles/Rancillac/VLPO-29.png",
    "image_url": "http://neuromorpho.org/images/imageFiles/Rancillac/VLPO-29.png",
    "neuron_id": "NMO_67811",
    "min_weight": "",
    "cell_class": "principal cell, low-threshold calcium spiking, noradrenaline-inhibited",
    "strain_name": "C57BL/6J",
    "datasource_id": "scr_002145_neuromorpho_modelimage",
    "datasource_name": "NeuroMorpho",
    "datasource_description": "A curated repository of digitally reconstructed neurons.",
    "datasource_type": "morphology",
    "identifier": "http://neuromorpho.org/neuron_info.jsp?neuron_name=VLPO-29"
  }
}
Uploaded 147477 records to gs://ks_datasets/preprocessed_data/scr_002145_neuromorpho_modelimage.json"""