import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch
from google.cloud import aiplatform, bigquery
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("retrieval")
logger.setLevel(logging.INFO)


@dataclass
class RetrievedItem:
    id: str
    title_guess: str
    content: str
    metadata: Dict[str, Any]
    primary_link: Optional[str]
    other_links: List[str]
    similarity: float  # Higher is better


class Retriever:
    """
    Matching Engine retriever that uses:
      - INDEX_ENDPOINT_ID_FULL (full resource path)
      - DEPLOYED_INDEX_ID
      - BigQuery docstore (dataset/table/location)
    """

    def __init__(self):
        # ---- Environment ----
        self.project_id = os.getenv("GCP_PROJECT_ID", "")
        self.region = os.getenv("GCP_REGION", "")

        # IMPORTANT: full resource path e.g.
        # projects/xxx/locations/us-central1/indexEndpoints/1234567890
        self.index_endpoint_full = os.getenv("INDEX_ENDPOINT_ID_FULL", "")
        self.deployed_id = os.getenv("DEPLOYED_INDEX_ID", "")

        self.embed_model_name = os.getenv(
            "EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5"
        )

        self.bq_dataset = os.getenv("BQ_DATASET_ID", "ks_metadata")
        self.bq_table = os.getenv("BQ_TABLE_ID", "docstore")
        self.bq_location = os.getenv("BQ_LOCATION", "US")

        self.is_enabled = all(
            [self.project_id, self.region, self.index_endpoint_full, self.deployed_id]
        )
        if not self.is_enabled:
            logger.warning(
                "GCP env incomplete. Vector search disabled "
                f"(project={bool(self.project_id)}, region={bool(self.region)}, "
                f"endpoint_full={bool(self.index_endpoint_full)}, deployed={bool(self.deployed_id)})"
            )
            return

        # Cloud clients 
        try:
            aiplatform.init(project=self.project_id, location=self.region)
            self.index_ep = aiplatform.MatchingEngineIndexEndpoint(
                index_endpoint_name=self.index_endpoint_full
            )
            self.bq = bigquery.Client(project=self.project_id)
        except Exception as e:
            logger.error(f"Failed to initialize GCP clients: {e}")
            self.is_enabled = False
            return

        # HF embedder 
        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embed_model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.embed_model_name, trust_remote_code=True
            ).eval().to(self.device)
            print(f"Vector search initialized successfully on {self.device}")
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            self.is_enabled = False

    # Embedding 
    def _embed(self, text: str) -> List[float]:
        text = " ".join((text or "").split())[:8000]
        toks = self.tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=1024
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**toks, return_dict=True)
        rep = (
            out.pooler_output
            if getattr(out, "pooler_output", None) is not None
            else out.last_hidden_state.mean(dim=1)
        )
        rep = torch.nn.functional.normalize(rep, p=2, dim=1)
        return rep[0].cpu().tolist()

    # BigQuery metadata 
    def _bq_fetch(self, ids: List[str]) -> Dict[str, Dict[str, Any]]:
        if not ids:
            return {}
        table = f"{self.project_id}.{self.bq_dataset}.{self.bq_table}"
        sql = f"""
            SELECT datapoint_id, chunk, metadata_filters, source_file
            FROM `{table}`
            WHERE datapoint_id IN UNNEST(@ids)
        """
        cfg = bigquery.QueryJobConfig(
            query_parameters=[bigquery.ArrayQueryParameter("ids", "STRING", ids)]
        )
        rows = self.bq.query(sql, job_config=cfg, location=self.bq_location).result()
        out: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            md = r.metadata_filters
            if isinstance(md, str):
                try:
                    md = json.loads(md)
                except Exception:
                    md = {"_raw": md}
            out[r.datapoint_id] = {
                "chunk": r.chunk or "",
                "metadata": md or {},
                "source_file": r.source_file,
            }
        return out

    # Public API 
    def search(
        self, query: str, top_k: int = 20, context: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedItem]:
        """
        Returns a list[RetrievedItem]. If context['raw'] is True, use the query verbatim.
        """
        if not self.is_enabled or not query:
            return []

        # Do NOT modify the query if 'raw' is set (agents pass raw=True)
        qtext = query if (context or {}).get("raw") else query

        try:
            vec = self._embed(qtext)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

        try:
            # High-level SDK returns a list of neighbors per query.
            n = max(1, min(top_k * 2, 100))
            results = self.index_ep.find_neighbors(
                deployed_index_id=self.deployed_id, queries=[vec], num_neighbors=n
            )
            neighbors = results[0] if results else []
            if not neighbors:
                return []

            ids = [nb.id for nb in neighbors]
            distances = [nb.distance for nb in neighbors]

            # Fetch metadata from BQ
            try:
                meta_map = self._bq_fetch(ids)
            except Exception as e:
                logger.error(f"BigQuery fetch error: {e}")
                meta_map = {}

            items: List[RetrievedItem] = []
            for dp_id, dist in zip(ids, distances):
                meta_info = meta_map.get(dp_id, {})
                md = meta_info.get("metadata", {}) or {}
                title = (
                    md.get("dc.title")
                    or md.get("title")
                    or md.get("name")
                    or "Untitled"
                )
                content = meta_info.get("chunk", md.get("description", "")) or ""
                # Extract the actual dataset identifier/URL from metadata
                link = (md.get("primary_link") or 
                        md.get("url") or 
                        md.get("link") or 
                        md.get("identifier") or  # This is where the actual dataset URL is stored
                        md.get("dc", {}).get("identifier") or  # Check nested dc.identifier
                        "")
                # Matching Engine distance may be negative dot-product/L2; convert to similarity
                try:
                    similarity = -float(dist) if dist is not None else 0.0
                except Exception:
                    similarity = 0.0

                items.append(
                    RetrievedItem(
                        id=dp_id,
                        title_guess=str(title),
                        content=str(content),
                        metadata=md,
                        primary_link=link,
                        other_links=[],
                        similarity=similarity,
                    )
                )

            items.sort(key=lambda x: x.similarity, reverse=True)
            return items[: top_k or 10]
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
