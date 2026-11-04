import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

from local_retriever import LocalRetriever  # noqa: E402
from retrieval import get_retriever  # noqa: E402


class LocalRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.retriever = LocalRetriever()

    def test_loads_public_dataset_fixtures(self):
        self.assertTrue(self.retriever.is_enabled)
        self.assertEqual(len(self.retriever._fixtures), 15)
        modalities = {
            fixture["metadata"]["modality"] for fixture in self.retriever._fixtures
        }
        self.assertSetEqual(
            modalities, {"fMRI", "electrophysiology", "calcium imaging"}
        )

    def test_factory_uses_offline_retriever_without_gcp_settings(self):
        settings = {
            "GCP_PROJECT_ID": "",
            "GCP_REGION": "",
            "INDEX_ENDPOINT_ID_FULL": "",
            "DEPLOYED_INDEX_ID": "",
        }
        with patch.dict(os.environ, settings):
            retriever = get_retriever()

        self.assertIsInstance(retriever, LocalRetriever)
        self.assertTrue(retriever.search("calcium imaging"))

    def test_search_ranks_matching_datasets(self):
        results = self.retriever.search("NARPS fMRI mixed gambles", top_k=3)

        self.assertEqual(results[0].id, "openneuro:ds001734")
        self.assertLessEqual(len(results), 3)
        self.assertGreater(results[0].similarity, results[-1].similarity)

    def test_search_returns_modality_matches(self):
        results = self.retriever.search("two photon calcium imaging", top_k=4)

        self.assertTrue(results)
        self.assertTrue(
            all(item.metadata["modality"] == "calcium imaging" for item in results)
        )
        self.assertTrue(
            all(item.primary_link.startswith("https://") for item in results)
        )

    def test_search_returns_no_results_for_empty_or_unknown_queries(self):
        self.assertEqual(self.retriever.search(""), [])
        self.assertEqual(self.retriever.search("!!!"), [])
        self.assertEqual(self.retriever.search("unmatchedwordxyz"), [])

    def test_search_respects_non_positive_top_k(self):
        self.assertEqual(self.retriever.search("hippocampus", top_k=0), [])
        self.assertEqual(self.retriever.search("hippocampus", top_k=-1), [])


if __name__ == "__main__":
    unittest.main()
