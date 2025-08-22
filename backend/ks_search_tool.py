import os
import json
import requests
from typing import Dict, Optional, Set, Union, List
import re
from difflib import SequenceMatcher
from urllib.parse import urlparse
from pydantic import BaseModel
from langchain.tools import tool

# Data source mapping
DATASOURCE_NAME_TO_ID = {
    "Allen Brain Atlas Mouse Brain - Expression": "scr_002978_aba_expression",
    "GENSAT": "scr_002721_gensat_geneexpression",
    "NeuroMorpho": "scr_002145_neuromorpho_modelimage",
    "Cell Image Library": "scr_003510_cil_images",
    "Human Brain Atlas": "scr_006131_hba_atlas",
    "IonChannelGenealogy": "scr_014194_icg_ionchannels",
    "NeuroML Database": "scr_013705_neuroml_models",
    "EBRAINS": "scr_017612_ebrains",
    "ModelDB": "scr_007271_modeldb_models",
    "Blue Brain Project Cell Morphology": "scr_014306_bbp_cellmorphology",
    "OpenNEURO": "scr_005031_openneuro",
    "DANDI Archive": "scr_017571_dandi",
    "NeuronDB": "scr_003105_neurondb_currents",
    "SPARC": "scr_017041_sparc",
    "CONP Portal": "scr_016433_conp",
    "NeuroElectro": "scr_006274_neuroelectro_ephys",
    "Brain/MINDS": "scr_005069_brainminds"
}

# Reverse mapping for ID to name
DATASOURCE_ID_TO_NAME = {v: k for k, v in DATASOURCE_NAME_TO_ID.items()}

# Common institution name variations
INSTITUTION_ALIASES = {
    'ebrain': 'EBRAINS',
    'ebrains': 'EBRAINS',
    'allen': 'Allen Brain Atlas Mouse Brain - Expression',
    'allen institute': 'Allen Brain Atlas Mouse Brain - Expression',
    'dandi': 'DANDI Archive',
    'openneuro': 'OpenNEURO',
    'sparc': 'SPARC',
    'modeldb': 'ModelDB',
    'neuromorpho': 'NeuroMorpho',
    'conp': 'CONP Portal',
    'brain/minds': 'Brain/MINDS',
    'brainminds': 'Brain/MINDS'
}

# Constants
DEFAULT_TIMEOUT = 10
API_BASE_URL = "https://knowledge-space.org/entity/source-data-by-entity"
GENERAL_SEARCH_URL = "https://api.knowledge-space.org/datasets/search"
CONFIG_FILE = 'datasources_config.json'


def calculate_similarity(query: str, target: str) -> float:
    """Calculate similarity between two strings."""
    if not query or not target:
        return 0.0
    return SequenceMatcher(None, query.lower(), target.lower()).ratio()


def find_best_matches(query: str, candidates: List[str], threshold: float = 0.8, max_matches: int = 5) -> List[str]:
    """Find best fuzzy matches for a query among candidates."""
    matches = []
    for candidate in candidates:
        similarity = calculate_similarity(query, candidate)
        if similarity >= threshold:
            matches.append((candidate, similarity))
    
    matches.sort(key=lambda x: x[1], reverse=True)
    return [match[0] for match in matches[:max_matches]]



def extract_datasource_info_from_link(link: str) -> tuple:
    """
    Extract datasource ID and dataset ID from a link.
    Returns (datasource_id, dataset_id) or (None, None) if not found.
    """
    if not link:
        return None, None
    
    # Pattern matching for common datasource URLs
    patterns = [
        (r'neuromorpho\.org.*neuron_id=(\d+)', 'scr_002145_neuromorpho_modelimage'),
        (r'dandiarchive\.org/dandiset/(\d+)', 'scr_017571_dandi'),
        (r'openneuro\.org/datasets/(ds\d+)', 'scr_005031_openneuro'),
        (r'modeldb\.science/(\d+)', 'scr_007271_modeldb_models'),
        (r'ebi\.ac\.uk/ebrains/.*?/([^/]+)$', 'scr_017612_ebrains'),
        (r'sparc\.science/datasets/(\d+)', 'scr_017041_sparc'),
        (r'/entity/source:([^/]+)/([^/]+)', None),
    ]
    
    for pattern, default_source in patterns:
        match = re.search(pattern, link, re.IGNORECASE)
        if match:
            if default_source:
                dataset_id = match.group(1)
                return default_source, dataset_id
            else:
                # For generic pattern, extract both source and dataset ID
                source_part = match.group(1)
                dataset_id = match.group(2) if match.lastindex > 1 else match.group(1)
                # Try to match source part to known datasources
                for ds_id in DATASOURCE_ID_TO_NAME:
                    if source_part in ds_id:
                        return ds_id, dataset_id
    
    hostname = urlparse(link).hostname
    if hostname:
        hostname_lower = hostname.lower()
        if 'neuromorpho' in hostname_lower:
            return 'scr_002145_neuromorpho_modelimage', None
        elif 'dandi' in hostname_lower:
            return 'scr_017571_dandi', None
        elif 'openneuro' in hostname_lower:
            return 'scr_005031_openneuro', None
        elif 'modeldb' in hostname_lower:
            return 'scr_007271_modeldb_models', None
        elif 'ebrains' in hostname_lower:
            return 'scr_017612_ebrains', None
        elif 'sparc' in hostname_lower:
            return 'scr_017041_sparc', None
    
    return None, None

def fetch_dataset_details(datasource_id: str, dataset_id: str) -> dict:
    """
    Fetch detailed information about a specific dataset from a datasource.
    """
    if not datasource_id or not dataset_id:
        return {}
    
    try:
        # Use the KnowledgeSpace API to get dataset details
        url = f"https://api.knowledge-space.org/datasources/{datasource_id}/datasets/{dataset_id}"
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"  -> Error fetching details for {datasource_id}/{dataset_id}: {e}")
        return {}

def enrich_with_dataset_details(results: List[dict], top_k: int = 10) -> List[dict]:
    """
    Enrich search results with detailed dataset information.
    """
    enriched_results = []
    
    for i, result in enumerate(results[:top_k]):
        if i >= top_k:
            break
            
        link = result.get('primary_link', '') or result.get('metadata', {}).get('url', '')
        datasource_id, dataset_id = extract_datasource_info_from_link(link)
        
        if not datasource_id:
            metadata = result.get('metadata', {}) or result.get('_source', {})
            source_info = metadata.get('source', '') or metadata.get('datasource', '')
            if source_info:
                for name, ds_id in DATASOURCE_NAME_TO_ID.items():
                    if name.lower() in str(source_info).lower():
                        datasource_id = ds_id
                        break
        
        if datasource_id and not dataset_id:
            metadata = result.get('metadata', {}) or result.get('_source', {})
            dataset_id = metadata.get('id', '') or metadata.get('dataset_id', '') or result.get('_id', '')
        
        if datasource_id and dataset_id:
            details = fetch_dataset_details(datasource_id, dataset_id)
            if details:
                result['detailed_info'] = details
                result['datasource_id'] = datasource_id
                result['datasource_name'] = DATASOURCE_ID_TO_NAME.get(datasource_id, datasource_id)
                
                if 'metadata' not in result:
                    result['metadata'] = {}
                result['metadata'].update(details)
        
        enriched_results.append(result)
    
    return enriched_results

def general_search(query: str, top_k: int = 10, enrich_details: bool = True) -> dict:
    """
    General search with optional dataset detail enrichment.
    When enrich_details is True, fetches detailed information for each dataset.
    """
    print("--> Executing general search...")
    base_url = "https://api.knowledge-space.org/datasets/search"
    params = {"q": query or "*", "per_page": min(top_k * 2, 50)}
    try:
        resp = requests.get(base_url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        results_list = data.get("results", [])
        normalized_results = []
        for i, item in enumerate(results_list):
            title = (item.get("title") or item.get("name") or item.get("dc.title") or "Dataset")
            description = (item.get("description") or item.get("abstract") or item.get("summary") or "")
            
            # Extract the actual dataset identifier/URL from metadata
            url = (item.get("url") or 
                   item.get("link") or 
                   item.get("access_url") or 
                   item.get("identifier") or  
                   item.get("dc", {}).get("identifier") or  # Check nested dc.identifier
                   "https://knowledge-space.org")
            
            normalized_results.append({
                "_id": item.get("id", f"ks_{i}"),
                "_source": item,
                "_score": 1.0,
                "title_guess": title,
                "content": description,
                "primary_link": url,
                "metadata": item
            })
        print(f"  -> General search returned {len(normalized_results)} results")
        
        # Enrich with detailed information if requested
        if enrich_details and normalized_results:
            print("  -> Enriching results with detailed dataset information...")
            normalized_results = enrich_with_dataset_details(normalized_results, top_k)
        
        return {"combined_results": normalized_results[:top_k]}
    except requests.RequestException as e:
        print(f"  -> Error during general search: {e}")
        return {"combined_results": []}

def _perform_search(data_source_id: str, query: str, filters: dict, all_configs: dict, timeout: int = 10) -> List[dict]:
    """Search a specific source with the exact query and flat filters."""
    print(f"--> Searching source '{data_source_id}' with query: '{(query or '*')[:50]}...'")
    base_url = "https://knowledge-space.org/entity/source-data-by-entity"
    valid_filter_map = all_configs.get(data_source_id, {}).get('available_filters', {})
    exact_match_filters = []
    for key, value in (filters or {}).items():
        if key in valid_filter_map:
            real_field = valid_filter_map[key]['field']
            # Use fuzzy matching for all fields
            field_values = valid_filter_map[key].get('values', [])
            best_matches = find_best_matches(value, field_values, threshold=0.8)
            if best_matches:
                # Use the best match for exact filter
                exact_match_filters.append({"term": {real_field: best_matches[0]}})
            else:
                # If no fuzzy match, still try exact match
                exact_match_filters.append({"term": {real_field: value}})
    query_payload = {
        "query": {
            "bool": {
                "must": {"query_string": {"query": query or "*"}},
                "filter": exact_match_filters
            }
        },
        "size": 20
    }
    params = {'body': json.dumps(query_payload), 'source': data_source_id}
    try:
        resp = requests.get(base_url, params=params, timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        hits = (data[0] if isinstance(data, list) and data else data).get('hits', {}).get('hits', [])
        print(f"  -> Retrieved {len(hits)} raw results")
        out = []
        for hit in hits:
            src = hit.get('_source', {}) or {}
            title = (src.get("title") or src.get("name") or src.get("dc.title") or "Dataset")
            desc = (src.get("description") or src.get("abstract") or src.get("summary") or "")
            
            # Extract the actual dataset identifier/URL from metadata
            link = (src.get('url') or 
                    src.get('link') or 
                    src.get('primary_link') or 
                    src.get('identifier') or 
                    src.get('dc', {}).get('identifier') or  # Check nested dc.identifier
                    "No link available")
            
            out.append({
                "_id": hit.get("_id"),
                "_source": src,
                "_score": hit.get("_score", 1.0),
                "title_guess": title,
                "content": desc,
                "primary_link": link,
                "metadata": src
            })
        return out
    except requests.RequestException as e:
        print(f"  -> Error searching {data_source_id}: {e}")
        return []

@tool(args_schema=BaseModel)
def smart_knowledge_search(query: Optional[str] = None,
                           filters: Optional[Union[Dict, Set]] = None,
                           data_source: Optional[str] = None,
                           top_k: int = 10) -> dict:
    """
    LLM-driven smart search:
    - Uses the provided query and filters exactly as given by the LLM.
    - If ANY filters exist: do NOT fall back to general search; query the sources that support them.
    - If NO filters exist: run general search with the provided query.
    """
    q = query or "*"
    f = filters or {}

    # Handle institution aliases
    if data_source:
        data_source_lower = data_source.lower()
        # Check if it's an alias
        for alias, full_name in INSTITUTION_ALIASES.items():
            if alias in data_source_lower:
                data_source = full_name
                break
    
    # If looking for specific datasource without other filters
    if data_source and not f:
        
        config_path = 'datasources_config.json'
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as fh:
                all_configs = json.load(fh)
            target_id = DATASOURCE_NAME_TO_ID.get(data_source) or (data_source if data_source in all_configs else None)
            if target_id:
                results = _perform_search(target_id, q, {}, all_configs)
                if results:
                    return {"combined_results": results[:top_k]}

    # If there are no filters, fall back to general search
    if not f:
        return general_search(q, top_k, enrich_details=True)

    # Filters exist -> query sources that support them.
    config_path = 'datasources_config.json'
    if not os.path.exists(config_path):
        print("Config not found; filters present; returning empty results.")
        return {"combined_results": []}

    with open(config_path, 'r', encoding='utf-8') as fh:
        all_configs = json.load(fh)

    # Pick sources whose available_filters contain any of our filter keys.
    if data_source:
        target_id = DATASOURCE_NAME_TO_ID.get(data_source) or (data_source if data_source in all_configs else None)
        if not target_id:
            return {"combined_results": []}
        results = _perform_search(target_id, q, f, all_configs)
    else:
        matched = [sid for sid, cfg in all_configs.items()
                   if any(k in (cfg.get('available_filters') or {}) for k in f.keys())]
        if not matched:
            print("No sources support these filters; returning empty results.")
            return {"combined_results": []}
        results = []
        for sid in matched[:3]:
            results += _perform_search(sid, q, f, all_configs)

    if not results:
        print("Sources searched but returned 0 hits; falling back to general search.")
        # Fall back to general search if no results with filters
        return general_search(q, top_k, enrich_details=True)

    # Dedupe & rank
    seen, unique = set(), []
    for r in results:
        rid = r.get('_id')
        if rid and rid not in seen:
            seen.add(rid)
            unique.append(r)
    unique.sort(key=lambda x: x.get('_score', 0), reverse=True)
    return {"combined_results": unique[:top_k]}

