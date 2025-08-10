import torch
from google.cloud import aiplatform, bigquery
from transformers import AutoModel, AutoTokenizer


class Retriever:
    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "")
        self.region = os.getenv("GCP_REGION", "")
        self.endpoint_id = os.getenv("INDEX_ENDPOINT_ID", "")
        self.deployed_id = os.getenv("DEPLOYED_INDEX_ID", "")

        self.embed_model = os.getenv("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
        self.bq_dataset = os.getenv("BQ_DATASET_ID", "ks_metadata")
        self.bq_table = os.getenv("BQ_TABLE_ID", "docstore")
        self.bq_location = os.getenv("BQ_LOCATION", "EU")

        aiplatform.init(project=self.project_id, location=self.region)
        full_ep = f"projects/{self.project_id}/locations/{self.region}/indexEndpoints/{self.endpoint_id}"
        self.index_ep = aiplatform.MatchingEngineIndexEndpoint(index_endpoint_name=full_ep)

        self.bq = bigquery.Client(project=self.project_id)

        self.tokenizer = AutoTokenizer.from_pretrained(self.embed_model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(self.embed_model, trust_remote_code=True).eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

    def _embed(self, text: str) -> List[float]:
        toks = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=1024).to(self.device)
        with torch.no_grad():
            out = self.model(**toks, return_dict=True)
        rep = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state.mean(dim=1)
        rep = torch.nn.functional.normalize(rep, p=2, dim=1)
        return rep[0].cpu().tolist()

    def _bq_fetch(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        table = f"{self.project_id}.{self.bq_dataset}.{self.bq_table}"
        sql = f"""
            SELECT datapoint_id, chunk, metadata_filters, source_file
            FROM {table}
            WHERE datapoint_id IN UNNEST(@ids)
        """
        cfg = bigquery.QueryJobConfig(query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)])
        rows = self.bq.query(sql, job_config=cfg, location=self.bq_location).result()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            md = r.metadata_filters
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:
                    md = {"_raw": md}
            out[r.datapoint_id] = {"chunk": r.chunk, "metadata": md, "source_file": r.source_file}
        return out



    def get_candidates(self, query: str, n: int) -> List[Dict[str, Any]]:
        vec = self._embed(query)
        resp = self.index_ep.find_neighbors(deployed_index_id=self.deployed_id, queries=[vec], num_neighbors=n)
        if not resp or not resp[0]:
            return []
        raw = [
            {"datapoint_id": nb.id, "distance": nb.distance, "similarity": -nb.distance}
            for nb in resp[0]
        ]
        meta = self._bq_fetch([r["datapoint_id"] for r in raw])
        for r in raw:
            m = meta.get(r["datapoint_id"], {})
            r.update(content=m.get("chunk", ""), metadata=m.get("metadata", {}), source_file=m.get("source_file"))
        return raw