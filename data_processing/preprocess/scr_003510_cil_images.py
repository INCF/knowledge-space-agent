import json
import re
from google.cloud import storage
from bs4 import BeautifulSoup

INPUT_GCS_PATH           = "ks_datasets/raw_dataset/data_sources/scr_003510_cil_images.json"
OUTPUT_GCS_PATH          = "ks_datasets/preprocessed_data/scr_003510_cil_images.json"
DATASOURCE_ID            = "scr_003510_cil_images"
DATASOURCE_NAME          = "Cell Image Library"
DATASOURCE_DESCRIPTION   = "Provides annotated images, videos and animations of cellular processes."
DATASOURCE_TYPE          = "morphology"

def clean_html(html: str) -> str:
    return BeautifulSoup(html or "", "html.parser").get_text()

def extract_urls(text: str) -> list[str]:
    return list(set(re.findall(r"https?://[^\s\"<>]+", text or "")))

def safe_join(lst: list, sep: str = "; ") -> str:
    return sep.join(str(x).strip() for x in lst if isinstance(x, str) and x.strip())

def preprocess_record(rec: dict) -> dict:
    rec_id                      = rec.get("id", "")
    processinghistory           = rec.get("processinghistory", "") or ""
    speciestaxaspecific         = rec.get("speciestaxaspecific", "") or ""
    pathologicalprocess         = rec.get("pathologicalprocess", "") or ""
    dc                          = rec.get("dc", {}) or {}
    description                 = dc.get("description", "") or ""
    title                       = dc.get("title", "") or ""
    itemtype                    = rec.get("itemtype", "") or ""
    technicaldetails            = rec.get("technicaldetails", "") or ""
    termsandconditions          = rec.get("termsandconditions", "") or ""
    relationtointactcell        = rec.get("relationtointactcell", "") or ""
    species                     = rec.get("species", []) or []
    ncbio_id                    = rec.get("ncbiorganismalclassification_id", "") or ""
    biological_process          = rec.get("biological_process", []) or []
    cell_type                   = rec.get("cell_type", []) or []
    imaging_mode                = rec.get("imaging_mode", []) or []
    dimension_units             = rec.get("dimension_units", "") or ""
    attributions                = rec.get("attributions", []) or []
    image_url                   = rec.get("image_url", "") or ""

    parts = [
        processinghistory,
        speciestaxaspecific,
        pathologicalprocess,
        clean_html(description),
        title,
        itemtype,
        technicaldetails,
        termsandconditions,
        relationtointactcell,
        safe_join(species),
        ncbio_id,
        safe_join(biological_process),
        safe_join(cell_type),
        safe_join(imaging_mode),
        dimension_units,
        safe_join(attributions)
    ]
    chunk = "\n".join(p for p in parts if p)

    meta = {
        "id": rec_id,
        "processinghistory": processinghistory,
        "speciestaxaspecific": speciestaxaspecific,
        "pathologicalprocess": pathologicalprocess,
        "image_url": image_url,
        "itemtype": itemtype,
        "technicaldetails": technicaldetails,
        "termsandconditions": termsandconditions,
        "relationtointactcell": relationtointactcell,
        "species": [s for s in species if isinstance(s, str)],
        "ncbiorganismalclassification_id": ncbio_id,
        "biological_process": [bp for bp in biological_process if isinstance(bp, str)],
        "cell_type": [ct for ct in cell_type if isinstance(ct, str)],
        "imaging_mode": [im for im in imaging_mode if isinstance(im, str)],
        "dimension_units": dimension_units,
        "attributions": [a for a in attributions if isinstance(a, str)],
        "datasource_id": DATASOURCE_ID,
        "datasource_name": DATASOURCE_NAME,
        "datasource_description": DATASOURCE_DESCRIPTION,
        "datasource_type": DATASOURCE_TYPE,
    }

    main_id = dc.get("identifier")
    if main_id:
        meta["identifier"] = main_id

    urls = extract_urls(description) + extract_urls(technicaldetails) + extract_urls(termsandconditions)
    if image_url:
        urls.append(image_url)
    seen = set()
    unique_urls = []
    for u in urls:
        if u not in seen:
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
  "chunk": "Converted from volocity to .avi\n:laboratory mouse (C57 Black 6); Staphylococcus aureus (USA300 strain)\n:Infection of subcutaneous tissue with staphylococcus aureus bead\nVisualization of neutrophil recruitment in capillaries proximal to agarose beads. (Figure 5 - Venular data)\nCIL:47378 - Micrococcus aureus - endothelial cell\nmicrograph, recorded image\nMice were anesthetized with ketamine/xylazine, and then subcutaneous tissue on a dorsal flap was exteriorized.  GFP staphylococcus aureus beads were inserted.  Antibodies were injected into the vasculature to label neutrophils (red) and endothelial cells (blue).  Blocking antibodies against the molecule Mac1 were administered after 2 hours of infection.\npublic_domain\nintact cells-in vivo imaging\nMicrococcus aureus; Staphylococcus aureus; Staphylococus aureus; Streptococcus aureus\nNCBITaxon:1280\nimmune cell recruitment\nendothelial cell; Endotheliocyte; neutrophil; neutrophilic leukocyte\nspinning disk confocal microscopy\nmicrons\nMark Harding",
  "metadata_filters": {
    "id": "47378",
    "processinghistory": "Converted from volocity to .avi",
    "speciestaxaspecific": ":laboratory mouse (C57 Black 6); Staphylococcus aureus (USA300 strain)",
    "pathologicalprocess": ":Infection of subcutaneous tissue with staphylococcus aureus bead",
    "image_url": "http://grackle.crbs.ucsd.edu:8001/ascb_il/render_thumbnail/47378/220/",
    "itemtype": "micrograph, recorded image",
    "technicaldetails": "Mice were anesthetized with ketamine/xylazine, and then subcutaneous tissue on a dorsal flap was exteriorized.  GFP staphylococcus aureus beads were inserted.  Antibodies were injected into the vasculature to label neutrophils (red) and endothelial cells (blue).  Blocking antibodies against the molecule Mac1 were administered after 2 hours of infection.",
    "termsandconditions": "public_domain",
    "relationtointactcell": "intact cells-in vivo imaging",
    "species": [
      "Micrococcus aureus",
      "Staphylococcus aureus",
      "Staphylococus aureus",
      "Streptococcus aureus"
    ],
    "ncbiorganismalclassification_id": "NCBITaxon:1280",
    "biological_process": [
      "immune cell recruitment"
    ],
    "cell_type": [
      "endothelial cell",
      " Endotheliocyte",
      " neutrophil",
      " neutrophilic leukocyte"
    ],
    "imaging_mode": [
      "spinning disk confocal microscopy"
    ],
    "dimension_units": "microns",
    "attributions": [
      "Mark Harding"
    ],
    "datasource_id": "scr_003510_cil_images",
    "datasource_name": "Cell Image Library",
    "datasource_description": "Provides annotated images, videos and animations of cellular processes.",
    "datasource_type": "morphology",
    "identifier": "http://cellimagelibrary.org/images/47378",
    "identifier1": "http://grackle.crbs.ucsd.edu:8001/ascb_il/render_thumbnail/47378/220/"
  }
}
Uploaded 10185 records to gs://ks_datasets/preprocessed_data/scr_003510_cil_images.json"""