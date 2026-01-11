import pytest
from ttfr_estimator import (
    estimate_ttfr,
    AccessType,
    ModalityComplexity,
    FormatType,
    infer_modality_from_keywords,
    detect_multimodal,
    assess_documentation_quality
)


class TestModalityInference:
    def test_infer_low_complexity(self):
        assert infer_modality_from_keywords("computational model simulation") == ModalityComplexity.LOW
        assert infer_modality_from_keywords("neuron morphology database") == ModalityComplexity.LOW
    
    def test_infer_medium_complexity(self):
        assert infer_modality_from_keywords("microscopy images") == ModalityComplexity.MEDIUM
        assert infer_modality_from_keywords("electrophysiology recordings") == ModalityComplexity.MEDIUM
    
    def test_infer_high_complexity(self):
        assert infer_modality_from_keywords("fMRI BOLD imaging") == ModalityComplexity.HIGH
        assert infer_modality_from_keywords("MRI brain scans") == ModalityComplexity.HIGH
    
    def test_infer_very_high_complexity(self):
        assert infer_modality_from_keywords("multimodal neuroimaging dataset") == ModalityComplexity.VERY_HIGH
    
    def test_empty_input(self):
        assert infer_modality_from_keywords("") == ModalityComplexity.MEDIUM
        assert infer_modality_from_keywords(None) == ModalityComplexity.MEDIUM


class TestMultimodalDetection:
    def test_detect_multimodal_positive(self):
        assert detect_multimodal("multimodal imaging and electrophysiology") == True
        assert detect_multimodal("combined MRI and PET") == True
        assert detect_multimodal("fMRI and behavioral data") == True
    
    def test_detect_multimodal_negative(self):
        assert detect_multimodal("fMRI imaging only") == False
        assert detect_multimodal("microscopy images") == False
    
    def test_empty_input(self):
        assert detect_multimodal("") == False
        assert detect_multimodal(None) == False


class TestDocumentationQuality:
    def test_high_quality(self):
        metadata = {
            "description": "A" * 150,
            "authors": ["Author 1"],
            "license": "MIT",
            "readme": "README.md"
        }
        assert assess_documentation_quality(metadata) == "high"
    
    def test_medium_quality(self):
        metadata = {
            "description": "A" * 150,
            "license": "MIT"
        }
        assert assess_documentation_quality(metadata) == "medium"
    
    def test_low_quality(self):
        metadata = {
            "description": "Short"
        }
        assert assess_documentation_quality(metadata) == "low"
    
    def test_empty_metadata(self):
        assert assess_documentation_quality({}) == "low"


class TestTTFREstimation:
    def test_openneuro_dataset(self):
        estimate = estimate_ttfr(
            datasource_id="scr_005031_openneuro",
            metadata={"description": "fMRI study with 100 subjects"},
            content="BIDS-formatted neuroimaging data"
        )
        
        assert estimate.total.min_days >= 0
        assert estimate.total.max_days > estimate.total.min_days
        assert "scr_005031_openneuro" in estimate.assumptions[0]
    
    def test_dandi_dataset(self):
        estimate = estimate_ttfr(
            datasource_id="scr_017571_dandi",
            metadata={"description": "Electrophysiology recordings from mouse cortex"},
            content="NWB format neural recordings"
        )
        
        assert estimate.total.min_days >= 0
        assert estimate.access_setup.min_days == 0.0
    
    def test_unknown_datasource_simple(self):
        estimate = estimate_ttfr(
            metadata={"description": "Simple morphology database"},
            content="Neuron morphology data"
        )
        
        assert estimate.total.min_days >= 0
        assert "general estimates" in estimate.assumptions[0].lower()
    
    def test_multimodal_dataset(self):
        estimate = estimate_ttfr(
            metadata={"description": "Combined fMRI and electrophysiology study"},
            content="Multimodal neuroimaging and neural recordings"
        )
        
        assert any("multimodal" in a.lower() for a in estimate.assumptions)
        assert estimate.preprocessing.max_days > 2.0
    
    def test_approval_required(self):
        estimate = estimate_ttfr(
            datasource_id="scr_005069_brainminds"
        )
        
        assert estimate.access_setup.min_days >= 1.0
    
    def test_to_dict_conversion(self):
        estimate = estimate_ttfr(datasource_id="scr_007271_modeldb_models")
        result = estimate.to_dict()
        
        assert "total" in result
        assert "breakdown" in result
        assert "assumptions" in result
        assert "display" in result["total"]
        assert isinstance(result["assumptions"], list)


class TestTimeRangeFormatting:
    def test_format_hours(self):
        estimate = estimate_ttfr(
            datasource_id="scr_002145_neuromorpho_modelimage"
        )
        
        total_str = str(estimate.total)
        assert "days" in total_str or "hours" in total_str
    
    def test_format_consistency(self):
        estimate = estimate_ttfr(
            datasource_id="scr_005031_openneuro"
        )
        
        result_dict = estimate.to_dict()
        assert result_dict["total"]["display"] == str(estimate.total)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
