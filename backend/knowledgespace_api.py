import requests
from typing import List, Dict, Optional

KS_BASE = "https://api.knowledge-space.org"


def list_datasources() -> List[Dict]:
    """List all available datasources"""
    url = f"{KS_BASE}/datasources"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_datasource_metadata(datasource_id: str) -> Dict:
    """Get metadata for a specific datasource"""
    url = f"{KS_BASE}/datasources/{datasource_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_datasource_keys(datasource_id: str) -> Dict:
    """Get keys for a specific datasource"""
    url = f"{KS_BASE}/datasources/{datasource_id}/keys"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def get_datasets(datasource_id: str, page: int = 0, per_page: int = 50) -> Dict:
    """Get datasets from a specific datasource"""
    url = f"{KS_BASE}/datasources/{datasource_id}/datasets"
    params = {"page": page, "per_page": per_page}
    response = requests.get(url, params=params, timeout=50)
    response.raise_for_status()
    return response.json()


def get_dataset_details(datasource_id: str, dataset_id: str) -> Dict:
    """Get detailed information about a specific dataset"""
    url = f"{KS_BASE}/datasources/{datasource_id}/datasets/{dataset_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def search_datasets(datasource_id: str, query: str, page: int = 0, per_page: int = 50) -> Dict:
    """Search datasets within a specific datasource"""
    url = f"{KS_BASE}/datasources/{datasource_id}/search"
    params = {"q": query, "page": page, "per_page": per_page}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def global_search_datasets(query: str, page: int = 0, per_page: int = 50) -> Dict:
    """Search datasets across all datasources"""
    url = f"{KS_BASE}/datasets/search"
    params = {"q": query, "page": page, "per_page": per_page}
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    return response.json()


def get_cde(cde_id: str) -> Dict:
    """Get Common Data Element information"""
    url = f"{KS_BASE}/cde/search?q={cde_id}"
    response = requests.get(url, timeout=10)
    response.raise_for_status()
    return response.json()


def format_datasets_list(datasets_json: Dict) -> str:
    """Format datasets JSON response into readable text"""
    total = datasets_json.get("total_count", 0)
    current = datasets_json.get("current_page", 0)
    results = datasets_json.get("results", [])

    lines = [f"Total Datasets Found: {total}  (Page {current})", ""]
    for ds in results:
        ds_id = ds.get("id", "<no_id>")
        dc = ds.get("dc", {})
        title = dc.get("title", "<no_title>")
        desc = dc.get("description", "").strip()
        identifier = dc.get("identifier", "")
        lines.append(f"- {ds_id}: {title}")
        if desc:
            lines.append(f"    Description: {desc}")
        if identifier:
            lines.append(f"    Identifier: {identifier}")
        lines.append("")  

    formatted = "\n".join(lines).strip()
    return formatted if formatted else "No datasets found."


def format_datasources_list(datasources: List[Dict]) -> str:
    """Format datasources list into readable text"""
    if not datasources:
        return "No datasources found."
    
    lines = [f"Available Datasources ({len(datasources)}):", ""]
    for ds in datasources:
        name = ds.get("name", "<no_name>")
        description = ds.get("description", "").strip()
        ds_id = ds.get("id", "<no_id>")
        lines.append(f"- {name} (ID: {ds_id})")
        if description:
            lines.append(f"    {description}")
        lines.append("")
    
    return "\n".join(lines).strip()


def format_dataset_details(dataset: Dict) -> str:
    """Format detailed dataset information into readable text"""
    dc = dataset.get("dc", {})
    title = dc.get("title", "<no_title>")
    description = dc.get("description", "").strip()
    identifier = dc.get("identifier", "")
    creator = dc.get("creator", "")
    subject = dc.get("subject", "")
    
    lines = [f"Dataset: {title}", ""]
    if description:
        lines.append(f"Description: {description}")
    if identifier:
        lines.append(f"Identifier: {identifier}")
    if creator:
        lines.append(f"Creator: {creator}")
    if subject:
        lines.append(f"Subject: {subject}")
    
    return "\n".join(lines).strip()


class KnowledgeSpaceAPI:
    """Wrapper class for KnowledgeSpace API interactions"""
    
    def __init__(self):
        self.base_url = KS_BASE
    
    def search_and_format(self, query: str, datasource_id: Optional[str] = None, limit: int = 10) -> str:
        """Search datasets and return formatted results"""
        try:
            if datasource_id:
                results = search_datasets(datasource_id, query, per_page=limit)
            else:
                results = global_search_datasets(query, per_page=limit)
            
            return format_datasets_list(results)
        except Exception as e:
            return f"Error searching datasets: {str(e)}"
    
    def get_datasources_info(self) -> str:
        """Get formatted list of available datasources"""
        try:
            response = list_datasources()
            # Handle case where response might be a dict with 'datasources' key
            if isinstance(response, dict) and 'datasources' in response:
                datasources = response.get('datasources', [])
            else:
                datasources = response
            return format_datasources_list(datasources)
        except Exception as e:
            return f"Error retrieving datasources: {str(e)}"
    
    def get_dataset_info(self, datasource_id: str, dataset_id: str) -> str:
        """Get formatted dataset details"""
        try:
            dataset = get_dataset_details(datasource_id, dataset_id)
            return format_dataset_details(dataset)
        except Exception as e:
            return f"Error retrieving dataset details: {str(e)}"
