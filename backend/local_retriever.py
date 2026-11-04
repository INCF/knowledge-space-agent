import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from retrieval import BaseRetriever, RetrievedItem

logger = logging.getLogger("retrieval.local")

_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "is", "it", "of", "on", "or", "that", "the", "this", "to",
    "with",
}


class LocalRetriever(BaseRetriever):
    """
    Minimal local retriever for development and testing when
    Vertex/GCP is unavailable.
    """

    def __init__(self):
        self._enabled = True
        fixture_path = Path(__file__).with_name("fixtures") / "sample_datasets.json"
        with fixture_path.open(encoding="utf-8") as fixture_file:
            fixtures = json.load(fixture_file)

        if not isinstance(fixtures, list):
            raise ValueError("Local retriever fixtures must be a JSON list")

        required_fields = {
            "id", "title", "description", "primary_link", "metadata"
        }
        for index, fixture in enumerate(fixtures):
            if not isinstance(fixture, dict) or not required_fields.issubset(fixture):
                raise ValueError(
                    f"Local retriever fixture {index} is missing required fields"
                )
            text_fields = ("id", "title", "description", "primary_link")
            if not all(isinstance(fixture[field], str) for field in text_fields):
                raise ValueError(
                    f"Local retriever fixture {index} text fields must be strings"
                )
            if not isinstance(fixture["metadata"], dict):
                raise ValueError(
                    f"Local retriever fixture {index} metadata must be an object"
                )

        self._fixtures = fixtures

    @property
    def is_enabled(self) -> bool:
        return self._enabled

    @staticmethod
    def _tokens(text: str) -> Set[str]:
        return {
            token
            for token in re.findall(r"\w+", text.casefold())
            if len(token) > 1 and token not in _STOP_WORDS
        }

    def search(
        self,
        query: str,
        top_k: int = 20,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[RetrievedItem]:
        query_tokens = self._tokens(query)
        if not query_tokens or top_k <= 0:
            return []

        matches = []
        for index, fixture in enumerate(self._fixtures):
            searchable_text = " ".join(
                (
                    fixture["title"],
                    fixture["description"],
                    json.dumps(fixture["metadata"], ensure_ascii=True),
                )
            )
            matched_tokens = query_tokens & self._tokens(searchable_text)
            if matched_tokens:
                similarity = len(matched_tokens) / len(query_tokens)
                metadata = dict(fixture["metadata"])
                item = RetrievedItem(
                    id=fixture["id"],
                    title_guess=fixture["title"],
                    content=fixture["description"],
                    metadata=metadata,
                    primary_link=fixture["primary_link"],
                    other_links=fixture.get("other_links", []),
                    similarity=similarity,
                )
                matches.append((similarity, len(matched_tokens), index, item))

        matches.sort(key=lambda match: (-match[0], -match[1], match[2]))
        logger.info(
            "LocalRetriever returned %d results for an offline search",
            min(len(matches), top_k),
        )
        return [match[3] for match in matches[:top_k]]
