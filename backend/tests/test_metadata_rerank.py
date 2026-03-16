import pytest
from ks_search_tool import rerank_results_using_metadata


def test_rerank_max_bounds():
    """
    Test that the maximum possible boost is exactly +30%
    (10% for Year, 15% for Citations, 5% for Trusted Source)
    """
    results = [
        # Baseline dataset
        {
            "_score": 100.0,
            "title_guess": "Old Data",
            "metadata": {"year": 1990, "citations": 0, "source": "Unknown"},
        },
        # Perfect dataset that should get the max 1.30x multiplier
        {
            "_score": 100.0,
            "title_guess": "Perfect Data",
            "metadata": {
                "year": 2024,
                "citations": 10000,
                "source": "Allen Brain Atlas",
            },
        },
    ]

    ranked = rerank_results_using_metadata(results)

    # "Perfect Data" should be first due to boost
    assert ranked[0]["title_guess"] == "Perfect Data"

    # Baseline should remain exactly 100.0 (no multiplier via min scaling)
    assert ranked[1]["_score"] == 100.0

    # Perfect Data should be exactly 130.0 (1.30x multiplier)
    assert ranked[0]["_score"] == 130.0
    assert ranked[0]["_rerank_multiplier"] == 1.30


def test_rerank_log_normalization():
    """
    Test that 10k citations doesn't astronomically outscore 10 citations
    thanks to log normalization.
    """
    results = [
        {"_score": 100.0, "title_guess": "Zero Cits", "metadata": {"citations": 0}},
        {"_score": 100.0, "title_guess": "Ten Cits", "metadata": {"citations": 10}},
        {
            "_score": 100.0,
            "title_guess": "Ten Thousand Cits",
            "metadata": {"citations": 10000},
        },
    ]

    ranked = rerank_results_using_metadata(results)

    # Highest should still be first
    assert ranked[0]["title_guess"] == "Ten Thousand Cits"

    multiplier_high = ranked[0]["_rerank_multiplier"]
    multiplier_mid = ranked[1]["_rerank_multiplier"]
    multiplier_low = ranked[2]["_rerank_multiplier"]

    # Verify the bounded maximum is respected (max +15% for citations)
    assert multiplier_high == 1.15
    assert multiplier_low == 1.00

    # 10 citations should give a meaningful logarithmic boost (log10(11) / log10(10001)) * 0.15
    # Let's just assert it is meaningfully greater than 1.0 but less than 1.15
    assert 1.0 < multiplier_mid < 1.15


def test_rerank_empty_metadata_handling():
    """
    Test that datasets missing metadata fields do not break the calculation.
    """
    results = [
        {"_score": 10.0, "title_guess": "No Meta1"},
        {"_score": 10.0, "title_guess": "No Meta2", "metadata": {}},
        {
            "_score": 10.0,
            "title_guess": "Garbage Meta",
            "metadata": {"year": "unknown", "citations": None},
        },
    ]

    ranked = rerank_results_using_metadata(results)

    # All should retain their base score of 10.0
    for r in ranked:
        assert r["_score"] == 10.0
        assert r["_rerank_multiplier"] == 1.0
