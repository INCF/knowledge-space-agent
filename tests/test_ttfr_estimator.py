import pytest
from ttfr_estimator import (
    TTFREstimate,
    TimeRange,
    ModalityComplexity,
    estimate_ttfr,
    infer_modality_from_keywords,
    detect_multimodal,
    assess_documentation_quality,
)


class TestInferModalityFromKeywords:
    def test_empty_string_returns_none(self):
        assert infer_modality_from_keywords("") is None

    def test_whitespace_only_returns_none(self):
        assert infer_modality_from_keywords("   \n\t  ") is None

    def test_very_high_multimodal(self):
        assert infer_modality_from_keywords("multimodal imaging study") == ModalityComplexity.VERY_HIGH

    def test_very_high_combined(self):
        assert infer_modality_from_keywords("combined fMRI and EEG") == ModalityComplexity.VERY_HIGH

    def test_high_mri(self):
        assert infer_modality_from_keywords("MRI dataset") == ModalityComplexity.HIGH

    def test_high_fmri(self):
        assert infer_modality_from_keywords("fMRI BOLD") == ModalityComplexity.HIGH

    def test_high_neuroimaging(self):
        assert infer_modality_from_keywords("neuroimaging pipeline") == ModalityComplexity.HIGH

    def test_medium_microscopy(self):
        assert infer_modality_from_keywords("microscopy images") == ModalityComplexity.MEDIUM

    def test_medium_ephys(self):
        assert infer_modality_from_keywords("electrophysiology recording") == ModalityComplexity.MEDIUM

    def test_low_model(self):
        assert infer_modality_from_keywords("computational model") == ModalityComplexity.LOW

    def test_low_ion_channel(self):
        assert infer_modality_from_keywords("ion channel database") == ModalityComplexity.LOW

    def test_order_very_high_wins_over_high(self):
        text = "multimodal imaging with fMRI"
        assert infer_modality_from_keywords(text) == ModalityComplexity.VERY_HIGH

    def test_no_keyword_returns_none(self):
        assert infer_modality_from_keywords("random text with no modality") is None

    def test_case_insensitive(self):
        assert infer_modality_from_keywords("MRI AND EEG") == ModalityComplexity.HIGH


class TestDetectMultimodal:
    def test_none_returns_false(self):
        assert detect_multimodal(None) is False

    def test_empty_string_returns_false(self):
        assert detect_multimodal("") is False

    def test_multimodal_returns_true(self):
        assert detect_multimodal("multimodal dataset") is True

    def test_multi_hyphen_modal_returns_true(self):
        assert detect_multimodal("multi-modal data") is True

    def test_combined_returns_true(self):
        assert detect_multimodal("combined approach") is True

    def test_integrative_returns_true(self):
        assert detect_multimodal("integrative analysis") is True

    def test_no_match_returns_false(self):
        assert detect_multimodal("single modality fMRI only") is False


class TestAssessDocumentationQuality:
    def test_none_returns_low(self):
        assert assess_documentation_quality(None) == "low"

    def test_empty_dict_returns_low(self):
        assert assess_documentation_quality({}) == "low"

    def test_long_description_with_link_returns_high(self):
        meta = {
            "description": "x" * 201,
            "documentation_url": "https://example.com/docs",
        }
        assert assess_documentation_quality(meta) == "high"

    def test_short_description_no_link_returns_low(self):
        assert assess_documentation_quality({"description": "short"}) == "low"

    def test_medium_description_returns_medium(self):
        meta = {"description": "a" * 51}
        assert assess_documentation_quality(meta) == "medium"

    def test_url_only_returns_medium(self):
        assert assess_documentation_quality({"url": "https://example.com"}) == "medium"

    def test_dc_description_used(self):
        meta = {"dc": {"description": "y" * 51}}
        assert assess_documentation_quality(meta) == "medium"


class TestTimeRangeStr:
    def test_hours_only(self):
        tr = TimeRange(0.25, 0.5)
        assert "hour" in str(tr)

    def test_singular_day(self):
        tr = TimeRange(1.0, 1.0)
        assert str(tr) == "1 day"

    def test_singular_hour(self):
        tr = TimeRange(1/24, 1/24)
        assert str(tr) == "1 hour"

    def test_days_range(self):
        tr = TimeRange(2.0, 5.0)
        assert str(tr) == "2–5 days"

    def test_mixed_units(self):
        tr = TimeRange(0.5, 2.0)
        s = str(tr)
        assert "hour" in s and "day" in s


class TestEstimateTtfr:
    def test_known_datasource_uses_config(self):
        est = estimate_ttfr(datasource_id="scr_005031_openneuro")
        assert isinstance(est, TTFREstimate)
        assert "scr_005031_openneuro" in est.assumptions[0]
        assert est.summary.min_days >= 0
        assert est.summary.max_days >= est.summary.min_days
        assert "access" in est.phases and "preprocessing" in est.phases and "first_output" in est.phases

    def test_known_datasource_ebrains(self):
        est = estimate_ttfr(datasource_id="scr_017612_ebrains")
        assert "scr_017612_ebrains" in est.assumptions[0]
        assert est.summary.min_days >= 2

    def test_unknown_datasource_infers_from_content(self):
        est = estimate_ttfr(content="fMRI BOLD neuroimaging")
        assert any("Unknown datasource" in a for a in est.assumptions)
        assert any("modality" in a.lower() for a in est.assumptions)

    def test_empty_input_uses_defaults(self):
        est = estimate_ttfr()
        assert any("Unknown datasource" in a for a in est.assumptions)
        assert len(est.phases) == 3

    def test_metadata_doc_quality_affects_estimate(self):
        est_low = estimate_ttfr(metadata={"description": "x"})
        est_high = estimate_ttfr(metadata={"description": "y" * 201, "url": "https://x.com"})
        assert est_high.summary.max_days <= est_low.summary.max_days * 1.6

    def test_multimodal_content_bumps_to_very_high(self):
        est = estimate_ttfr(content="multimodal fMRI and EEG combined")
        assert any("Multimodal" in a for a in est.assumptions)
        assert est.phases["first_output"].max_days >= 2
