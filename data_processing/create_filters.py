from __future__ import annotations

import json
import time
from typing import Any, Dict, List
import requests

BASE_URL = "https://knowledge-space.org/entity/source-data-by-entity"
CONFIG_FILENAME = "datasources_config.json"



def discover_all_datasources() -> List[str]:
    
    print("Using the provided list of data sources...")
    return [
        "scr_006274_neuroelectro_ephys", "scr_017041_sparc_old", "scr_014306_bbp_cellmorphology",
        "scr_016433_conp", "scr_003510_cil_images", "scr_003105_neurondb_currents",
        "scr_017041_sparc", "scr_002721_gensat_geneexpression", "scr_013705_neuroml_models",
        "scr_005031_openneuro", "scr_002145_neuromorpho_modelimage", "scr_005069_brainminds",
        "scr_014194_icg_ionchannels", "scr_017571_dandi", "scr_017612_ebrains",
        "scr_002978_aba_expression", "scr_006131_hba_atlas", "scr_007271_modeldb_models"
    ]


def _http_get(params: Dict[str, Any], max_retries: int = 3, timeout: int = 30) -> Any:
    """GET with exponential backoff and JSON parsing (raw object; list or dict)."""
    delay = 0.8
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(BASE_URL, params=params, timeout=timeout)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2
            else:
                raise
    raise last_err  


def _normalize_resp(data: Any) -> Dict[str, Any]:
    """KS sometimes returns a list; select the first dict-like payload."""
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                return item
    return {}



def generate_full_config(data_source_ids: List[str]) -> None:
    print("\nStarting configuration generation...")
    full_config: Dict[str, Any] = {}
    
    
    # A definitive, hardcoded map for all sources based on network log analysis.
   
    DEFINITIVE_CONFIGS = {
        "scr_002978_aba_expression": { "gene_name": "gene_name.keyword", "gene_symbol": "gene_symbol.keyword", "structure_name": "structure_name.keyword", "species": "species.keyword" },
        "scr_002721_gensat_geneexpression": { "gene_name": "gene_name.keyword", "structure_name": "structure_name.keyword", "stain": "stain.keyword", "acquisition_technique": "acquisition_technique.keyword", "age": "age.keyword", "expression_level": "expression_level.keyword" },
        "scr_002145_neuromorpho_modelimage": { "species": "species.keyword", "strain_name": "strain_name.keyword", "staining_method": "staining_method.keyword", "brain_region": "brain_region.keyword", "age": "age.keyword", "gender": "gender.keyword" },
        "scr_003510_cil_images": { "species": "species.keyword", "cell_type": "cell_type.keyword", "biological_process": "biological_process.keyword", "imaging_mode": "imaging_mode.keyword", "dimension_units": "dimension_units.keyword" },
        "scr_006131_hba_atlas": { "species": "species.keyword", "brain_region": "brain_region.keyword", "brain_view": "brain_view.keyword" },
        "scr_014194_icg_ionchannels": { "neuron_type": "neuron_type.keyword", "animal_model": "animal_model.keyword", "brain_area": "brain_area.keyword", "age": "age.keyword" },
        "scr_013705_neuroml_models": { "model_type": "model_type.keyword", "neurolex_terms": "neurolex_terms.keyword", "keywords": "keywords.keyword" },
        "scr_017612_ebrains": { "sex": "sex.keyword", "species": "species.keyword", "techniques": "techniques.keyword", "preparation": "preparation.keyword", "experimental_approach": "experimental_approach.keyword" },
        "scr_007271_modeldb_models": { "model_concepts": "model_concepts.keyword", "simulator_software": "simulator_software.keyword", "model_type": "model_type.keyword" },
        "scr_014306_bbp_cellmorphology": { "cell": "dc.subject.keyword", "region": "region_term.keyword" },
        "scr_005031_openneuro": { "BIDSVersion": "BIDSVersion.keyword", "Authors": "Authors.keyword", "License": "License.keyword" },
        "scr_017571_dandi": { "dataStandard": "dataStandard.keyword", "about": "about.keyword", "measurementTechnique": "measurementTechnique.keyword", "keywords": "keywords.keyword", "species": "species.keyword", "license": "license.keyword" },
        "scr_003105_neurondb_currents": { "neuron": "neuron.keyword", "current": "current.keyword", "compartment": "compartment.keyword" },
        "scr_016433_conp": { "keywords": "keywords.keyword", "isAbout": "isAbout.keyword", "license": "license.keyword", "formats": "formats.keyword", "creators": "creators.keyword" },
        "scr_017041_sparc": { "organ": "anatomy.organ.name.keyword", "modalities": "item.modalities.keyword", "keywords": "item.keywords.keyword", "techniques": "item.techniques.keyword", "species": "organisms.primary.species.name.keyword", "protocol": "protocols.keyword" },
        "scr_006274_neuroelectro_ephys": { "property_name": "property_name.keyword" },
        "scr_005069_brainminds": { "keywords": "keywords.keyword" }
    }

    for source_id in data_source_ids:
        print(f"\nInspecting source: {source_id}")
        source_filters: Dict[str, Dict[str, Any]] = {}

        if source_id in DEFINITIVE_CONFIGS:
            print("  -> Using definitive pre-configuration for this source.")
            
            config_to_check = DEFINITIVE_CONFIGS[source_id]
            aggs_payload = {
                real_field: {"terms": {"field": real_field}}
                for user_name, real_field in config_to_check.items()
            }
            payload = {
                "query": {"bool": {"must": [{"query_string": {"query": "*"}}]}},
                "size": 0,
                "aggs": aggs_payload,
            }
            params = {"body": json.dumps(payload), "source": source_id}
            
            try:
                raw = _http_get(params)
                data = _normalize_resp(raw)
                aggregations = data.get("aggregations", {})
                
                for user_name, real_field in config_to_check.items():
                    if real_field in aggregations:
                        buckets = [b.get("key") for b in aggregations[real_field].get("buckets", []) if b.get("key")]
                        if buckets:
                            print(f"  -> Found filter: '{user_name}' via field '{real_field}' ({len(buckets)} values)")
                            source_filters[user_name] = {"field": real_field, "values": buckets}
            except requests.RequestException as e:
                print(f"  -> Could not fetch definitive config data: {e}")
        else:
            print("  -> No definitive configuration found for this source. Skipping.")

        if source_filters:
            print(f"  -> Summary: Found {len(source_filters)} total filters for {source_id}.")
            full_config[source_id] = {
                "description": f"Configuration for {source_id}",
                "available_filters": source_filters,
            }
        else:
            print(f"  -> Summary: Found 0 total filters for {source_id}.")

    with open(CONFIG_FILENAME, "w", encoding="utf-8") as f:
        json.dump(full_config, f, indent=2, ensure_ascii=False)

    print(f"\nConfiguration written to {CONFIG_FILENAME}")



if __name__ == "__main__":
    all_sources = discover_all_datasources()
    generate_full_config(all_sources)
