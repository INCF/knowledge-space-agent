# retrieval.py
import os
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import re
import math

import torch
from google.cloud import aiplatform, bigquery
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import CrossEncoder

logger = logging.getLogger("retrieval")
logger.setLevel(logging.INFO)

if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(_h)


@dataclass
class RetrievedItem:
    id: str
    title_guess: str
    content: str
    metadata: Dict[str, Any]
    primary_link: Optional[str]
    other_links: List[str]
    similarity: float
    rerank_score: Optional[float] = None  # New field for re-ranking score
    hybrid_score: Optional[float] = None  # New field for hybrid search score


class AdvancedRetriever:
    """
    Advanced RAG system with hybrid search optimization and re-ranking.

    Environment variables required to enable vector search:
      - GCP_PROJECT_ID
      - GCP_REGION
      - INDEX_ENDPOINT_ID_FULL   (full resource path, e.g. projects/.../locations/.../indexEndpoints/...)
      - DEPLOYED_INDEX_ID

    Optional:
      - EMBED_MODEL_NAME         default: nomic-ai/nomic-embed-text-v1.5
      - RERANK_MODEL_NAME        default: cross-encoder/ms-marco-MiniLM-L-6-v2
      - BQ_DATASET_ID            default: ks_metadata
      - BQ_TABLE_ID              default: docstore
      - BQ_LOCATION              default: US
      - EMBED_MAX_TOKENS         default: 1024
      - QUERY_CHAR_LIMIT         default: 8000
      - HYBRID_ALPHA             default: 0.7 (weight for vector vs keyword search)
      - RERANK_TOP_K             default: 50 (number of items to re-rank)
    """

    def __init__(self):
        self.project_id = os.getenv("GCP_PROJECT_ID", "")
        self.region = os.getenv("GCP_REGION", "")
        self.index_endpoint_full = os.getenv("INDEX_ENDPOINT_ID_FULL", "")
        self.deployed_id = os.getenv("DEPLOYED_INDEX_ID", "")

        self.embed_model_name = os.getenv("EMBED_MODEL_NAME", "nomic-ai/nomic-embed-text-v1.5")
        self.rerank_model_name = os.getenv("RERANK_MODEL_NAME", "cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.bq_dataset = os.getenv("BQ_DATASET_ID", "ks_metadata")
        self.bq_table = os.getenv("BQ_TABLE_ID", "docstore")
        self.bq_location = os.getenv("BQ_LOCATION", "EU")

        try:
            self.embed_max_tokens = int(os.getenv("EMBED_MAX_TOKENS", "1024"))
        except Exception:
            self.embed_max_tokens = 1024
        try:
            self.query_char_limit = int(os.getenv("QUERY_CHAR_LIMIT", "8000"))
        except Exception:
            self.query_char_limit = 8000
        try:
            self.hybrid_alpha = float(os.getenv("HYBRID_ALPHA", "0.7"))  # Weight for vector vs keyword
        except Exception:
            self.hybrid_alpha = 0.7
        try:
            self.rerank_top_k = int(os.getenv("RERANK_TOP_K", "50"))
        except Exception:
            self.rerank_top_k = 50

        # Enable only if everything is present
        self.is_enabled = all(
            [self.project_id, self.region, self.index_endpoint_full, self.deployed_id]
        )
        if not self.is_enabled:
            logger.warning(
                "Vector search disabled due to incomplete GCP env: "
                f"project={bool(self.project_id)}, region={bool(self.region)}, "
                f"endpoint_full={bool(self.index_endpoint_full)}, deployed={bool(self.deployed_id)}"
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
            logger.error(f"GCP client initialization failed: {e}")
            self.is_enabled = False
            return

        try:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Initialize embedding model
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.embed_model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                self.embed_model_name, trust_remote_code=True
            ).eval().to(self.device)

            # Initialize re-ranking model
            self.rerank_model = CrossEncoder(self.rerank_model_name, device=self.device)

            logger.info(f"Advanced RAG system initialized on device={self.device}")
            logger.info(f"Embedding model: {self.embed_model_name}")
            logger.info(f"Re-ranking model: {self.rerank_model_name}")
        except Exception as e:
            logger.error(f"Model initialization failed: {e}")
            self.is_enabled = False

    def _embed(self, text: str) -> List[float]:
        """
        Returns a normalized embedding vector for the given text.
        Raises on failure (caller handles).
        """
        normalized = " ".join((text or "").split())
        if self.query_char_limit > 0:
            normalized = normalized[: self.query_char_limit]
        toks = self.tokenizer(
            normalized,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.embed_max_tokens,
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

    def _keyword_score(self, query: str, content: str) -> float:
        """
        Calculate keyword-based relevance score using TF-IDF-like approach.
        """
        if not query or not content:
            return 0.0

        query_lower = query.lower()
        content_lower = content.lower()

        # Split into terms
        query_terms = set(re.findall(r'\b\w+\b', query_lower))
        content_terms = re.findall(r'\b\w+\b', content_lower)

        # Calculate term frequency and matching
        matches = 0
        total_terms = len(content_terms)

        for term in query_terms:
            term_count = content_lower.count(term)
            matches += term_count

        # Calculate a normalized score
        if total_terms == 0:
            return 0.0

        # Use a logarithmic scale to prevent very long documents from dominating
        keyword_score = matches / (math.log(1 + total_terms) + 1)

        # Boost score if exact phrase is found
        if query_lower in content_lower:
            keyword_score *= 1.5

        return min(keyword_score, 1.0)  # Cap at 1.0

    def _hybrid_score(self, vector_score: float, keyword_score: float, alpha: float = 0.7) -> float:
        """
        Combine vector similarity and keyword scores using weighted average.
        """
        return alpha * vector_score + (1 - alpha) * keyword_score

    def _rerank_results(self, query: str, items: List[RetrievedItem]) -> List[RetrievedItem]:
        """
        Re-rank results using a cross-encoder model for better relevance.
        """
        if not items or len(items) == 0:
            return items

        # Prepare pairs for cross-encoder (query, content)
        pairs = []
        valid_items = []

        for item in items:
            content = f"{item.title_guess} {item.content}".strip()
            if content:  # Only process items with content
                pairs.append([query, content])
                valid_items.append(item)

        if not pairs:
            return items

        # Get re-ranking scores
        try:
            rerank_scores = self.rerank_model.predict(pairs)

            # Convert to list if needed
            if hasattr(rerank_scores, 'tolist'):
                rerank_scores = rerank_scores.tolist()

            # Update items with re-ranking scores
            for i, score in enumerate(rerank_scores):
                valid_items[i].rerank_score = float(score)

        except Exception as e:
            logger.error(f"Re-ranking failed: {e}")
            # If re-ranking fails, use original similarity scores
            for item in valid_items:
                item.rerank_score = item.similarity

        # Sort by re-ranking score in descending order
        valid_items.sort(key=lambda x: x.rerank_score or x.similarity, reverse=True)

        return valid_items

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

    def search(
        self, query: str, top_k: int = 20, context: Optional[Dict[str, Any]] = None,
        use_hybrid: bool = True, use_rerank: bool = True
    ) -> List[RetrievedItem]:
        """
        Executes an advanced search with hybrid optimization and re-ranking.
        """
        if not self.is_enabled or not query:
            return []

        qtext = query if (context or {}).get("raw") else query

        try:
            vec = self._embed(qtext)
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            return []

        try:
            # Get more results than needed for re-ranking
            n = max(1, min((top_k or 20) * 3, 100))
            results = self.index_ep.find_neighbors(
                deployed_index_id=self.deployed_id, queries=[vec], num_neighbors=n
            )
            neighbors = results[0] if results else []
            if not neighbors:
                return []

            ids = [nb.id for nb in neighbors]
            distances = [nb.distance for nb in neighbors]

            try:
                meta_map = self._bq_fetch(ids)
            except Exception as e:
                logger.error(f"BigQuery fetch error: {e}")
                meta_map = {}

            items: List[RetrievedItem] = []
            for dp_id, dist in zip(ids, distances):
                meta_info = meta_map.get(dp_id, {})
                md = meta_info.get("metadata", {}) or {}
                title = md.get("dc.title") or md.get("title") or md.get("name") or "Untitled"
                content = meta_info.get("chunk", md.get("description", "")) or ""
                link = (
                    md.get("primary_link")
                    or md.get("url")
                    or md.get("link")
                    or md.get("identifier")
                    or (md.get("dc", {}) if isinstance(md.get("dc"), dict) else {}).get("identifier")
                    or ""
                )
                try:
                    similarity = -float(dist) if dist is not None else 0.0
                    # Normalize similarity to 0-1 range if needed
                    if similarity < 0:
                        similarity = 1.0 / (1.0 + math.exp(-similarity))  # Sigmoid normalization
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

            # Apply hybrid search if enabled
            if use_hybrid:
                for item in items:
                    keyword_score = self._keyword_score(qtext, f"{item.title_guess} {item.content}")
                    item.hybrid_score = self._hybrid_score(item.similarity, keyword_score, self.hybrid_alpha)

            # Apply re-ranking if enabled
            if use_rerank and len(items) > 0:
                # Only re-rank top candidates to save computation
                rerank_count = min(len(items), self.rerank_top_k)
                top_items = items[:rerank_count]
                remaining_items = items[rerank_count:]

                # Re-rank the top items
                reranked_items = self._rerank_results(qtext, top_items)

                # Combine with remaining items (which keep their original order/scores)
                items = reranked_items + remaining_items

            # Sort by the most appropriate score (rerank > hybrid > similarity)
            items.sort(key=lambda x: (
                x.rerank_score or
                x.hybrid_score or
                x.similarity
            ), reverse=True)

            return items[: (top_k or 20)]
        except Exception as e:
            logger.error(f"Advanced search failed: {e}")
            import traceback
            traceback.print_exc()
            return []


class Retriever:
    """
    Legacy Retriever class for backward compatibility.
    """
    def __init__(self):
        self.advanced_retriever = AdvancedRetriever()
        # Copy attributes for backward compatibility
        self.is_enabled = self.advanced_retriever.is_enabled if self.advanced_retriever else False
        self.project_id = getattr(self.advanced_retriever, 'project_id', '')
        self.region = getattr(self.advanced_retriever, 'region', '')
        self.index_endpoint_full = getattr(self.advanced_retriever, 'index_endpoint_full', '')
        self.deployed_id = getattr(self.advanced_retriever, 'deployed_id', '')

    def search(
        self, query: str, top_k: int = 20, context: Optional[Dict[str, Any]] = None
    ) -> List[RetrievedItem]:
        """
        Executes a similarity search using the advanced retriever.
        """
        if not hasattr(self, 'advanced_retriever') or not self.advanced_retriever:
            return []
        return self.advanced_retriever.search(query, top_k, context, use_hybrid=True, use_rerank=True)
