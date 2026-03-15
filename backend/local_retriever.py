from typing import List, Dict, Any, Optional
from retrieval import BaseRetriever, RetrievedItem
import logging

logger = logging.getLogger("retrieval.local")


class LocalRetriever(BaseRetriever):
    """
    Minimal local retriever for development and testing when
    Vertex/GCP is unavailable.
    """

    def __init__(self):
        self._enabled = True

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    def search(
        self,
        query: str,
        top_k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        # Dev-only fallback: return no vector results for now
        logger.info("LocalRetriever active (no GCP). Returning empty results.")
        return []
