from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from enum import Enum


class AccessType(Enum):
    OPEN = "open"
    LOGIN = "login"
    APPROVAL = "approval"


class ModalityComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


class FormatType(Enum):
    BIDS = "bids"
    NWB = "nwb"
    STANDARD_IMAGE = "standard_image"
    CUSTOM = "custom"


@dataclass
class TimeRange:
    min_days: float
    max_days: float
    
    def __str__(self) -> str:
        if self.min_days < 1 and self.max_days < 1:
            min_hours = int(self.min_days * 24)
            max_hours = int(self.max_days * 24)
            return f"{min_hours}–{max_hours} hours"
        elif self.min_days < 1:
            min_hours = int(self.min_days * 24)
            max_days = int(self.max_days)
            return f"{min_hours} hours–{max_days} days"
        else:
            return f"{int(self.min_days)}–{int(self.max_days)} days"


@dataclass
class TTFREstimate:
    total: TimeRange
    access_setup: TimeRange
    preprocessing: TimeRange
    first_output: TimeRange
    assumptions: list[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": {"min_days": self.total.min_days, "max_days": self.total.max_days, "display": str(self.total)},
            "breakdown": {
                "access_setup": {"min_days": self.access_setup.min_days, "max_days": self.access_setup.max_days, "display": str(self.access_setup)},
                "preprocessing": {"min_days": self.preprocessing.min_days, "max_days": self.preprocessing.max_days, "display": str(self.preprocessing)},
                "first_output": {"min_days": self.first_output.min_days, "max_days": self.first_output.max_days, "display": str(self.first_output)}
            },
            "assumptions": self.assumptions
        }


DATASOURCE_CONFIG = {
    "scr_005031_openneuro": {
        "access": AccessType.OPEN,
        "format": FormatType.BIDS,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high"
    },
    "scr_017571_dandi": {
        "access": AccessType.OPEN,
        "format": FormatType.NWB,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high"
    },
    "scr_007271_modeldb_models": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium"
    },
    "scr_017612_ebrains": {
        "access": AccessType.LOGIN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.VERY_HIGH,
        "doc_quality": "high"
    },
    "scr_003510_cil_images": {
        "access": AccessType.OPEN,
        "format": FormatType.STANDARD_IMAGE,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium"
    },
    "scr_002145_neuromorpho_modelimage": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "high"
    },
    "scr_017041_sparc": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high"
    },
    "scr_002978_aba_expression": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high"
    },
    "scr_005069_brainminds": {
        "access": AccessType.APPROVAL,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.VERY_HIGH,
        "doc_quality": "medium"
    },
    "scr_002721_gensat_geneexpression": {
        "access": AccessType.OPEN,
        "format": FormatType.STANDARD_IMAGE,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium"
    },
    "scr_003105_neurondb_currents": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "medium"
    },
    "scr_006131_hba_atlas": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high"
    },
    "scr_014194_icg_ionchannels": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "medium"
    },
    "scr_013705_neuroml_models": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high"
    },
    "scr_014306_bbp_cellmorphology": {
        "access": AccessType.LOGIN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high"
    },
    "scr_016433_conp": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "medium"
    },
    "scr_006274_neuroelectro_ephys": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high"
    }
}


MODALITY_KEYWORDS = {
    ModalityComplexity.LOW: ["simulated", "model", "morphology", "ion channel", "database"],
    ModalityComplexity.MEDIUM: ["microscopy", "image", "gene expression", "single cell", "ephys", "electrophysiology"],
    ModalityComplexity.HIGH: ["mri", "fmri", "pet", "meg", "eeg", "bold", "neuroimaging"],
    ModalityComplexity.VERY_HIGH: ["multimodal", "multi-modal", "combined", "integrative"]
}


def estimate_access_time(access_type: AccessType, doc_quality: str) -> TimeRange:
    base_times = {
        AccessType.OPEN: (0.0, 0.04),
        AccessType.LOGIN: (0.04, 0.125),
        AccessType.APPROVAL: (1.0, 7.0)
    }
    
    min_time, max_time = base_times[access_type]
    
    if access_type == AccessType.OPEN and doc_quality == "low":
        max_time = 0.25
    
    return TimeRange(min_time, max_time)


def estimate_preprocessing_time(modality: ModalityComplexity, format_type: FormatType, is_multimodal: bool) -> TimeRange:
    base_times = {
        ModalityComplexity.LOW: (0.125, 0.5),
        ModalityComplexity.MEDIUM: (0.5, 2.0),
        ModalityComplexity.HIGH: (1.0, 4.0),
        ModalityComplexity.VERY_HIGH: (2.0, 7.0)
    }
    
    min_time, max_time = base_times[modality]
    
    if format_type == FormatType.BIDS:
        min_time *= 0.5
        max_time *= 0.6
    elif format_type == FormatType.NWB:
        min_time *= 0.7
        max_time *= 0.8
    elif format_type == FormatType.STANDARD_IMAGE:
        min_time *= 0.8
        max_time *= 0.9
    
    if is_multimodal:
        min_time *= 1.5
        max_time *= 2.0
    
    return TimeRange(min_time, max_time)


def estimate_first_output_time(modality: ModalityComplexity, has_examples: bool) -> TimeRange:
    base_times = {
        ModalityComplexity.LOW: (0.125, 0.5),
        ModalityComplexity.MEDIUM: (0.25, 1.0),
        ModalityComplexity.HIGH: (0.5, 2.0),
        ModalityComplexity.VERY_HIGH: (1.0, 3.0)
    }
    
    min_time, max_time = base_times[modality]
    
    if has_examples:
        min_time *= 0.6
        max_time *= 0.7
    
    return TimeRange(min_time, max_time)


def infer_modality_from_keywords(text: str) -> ModalityComplexity:
    if not text:
        return ModalityComplexity.MEDIUM
    
    text_lower = text.lower()
    
    for complexity in [ModalityComplexity.VERY_HIGH, ModalityComplexity.HIGH, ModalityComplexity.MEDIUM, ModalityComplexity.LOW]:
        keywords = MODALITY_KEYWORDS[complexity]
        if any(kw in text_lower for kw in keywords):
            return complexity
    
    return ModalityComplexity.MEDIUM


def detect_multimodal(text: str) -> bool:
    if not text:
        return False
    
    text_lower = text.lower()
    multimodal_keywords = ["multimodal", "multi-modal", "combined", "integrative", " and "]
    
    return any(kw in text_lower for kw in multimodal_keywords)


def assess_documentation_quality(metadata: Dict[str, Any]) -> str:
    score = 0
    
    if metadata.get("description") and len(str(metadata["description"])) > 100:
        score += 1
    if metadata.get("authors") or metadata.get("contributors"):
        score += 1
    if metadata.get("license"):
        score += 1
    if metadata.get("readme") or metadata.get("documentation_url"):
        score += 1
    
    if score >= 3:
        return "high"
    elif score >= 2:
        return "medium"
    else:
        return "low"


def estimate_ttfr(
    datasource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None
) -> TTFREstimate:
    
    assumptions = []
    
    if datasource_id and datasource_id in DATASOURCE_CONFIG:
        config = DATASOURCE_CONFIG[datasource_id]
        access_type = config["access"]
        format_type = config["format"]
        modality = config["typical_modality"]
        doc_quality = config["doc_quality"]
        assumptions.append(f"Using defaults for {datasource_id}")
    else:
        access_type = AccessType.OPEN
        format_type = FormatType.CUSTOM
        modality = ModalityComplexity.MEDIUM
        doc_quality = "medium"
        assumptions.append("Using general estimates")
    
    search_text = " ".join(filter(None, [
        str(metadata.get("description", "")) if metadata else "",
        str(metadata.get("title", "")) if metadata else "",
        content or ""
    ]))
    
    if search_text:
        inferred_modality = infer_modality_from_keywords(search_text)
        if inferred_modality != ModalityComplexity.MEDIUM:
            modality = inferred_modality
            assumptions.append(f"Inferred {modality.value} complexity from content")
    
    is_multimodal = detect_multimodal(search_text) if search_text else False
    if is_multimodal:
        assumptions.append("Detected multimodal dataset")
    
    if metadata:
        inferred_doc = assess_documentation_quality(metadata)
        if datasource_id not in DATASOURCE_CONFIG:
            doc_quality = inferred_doc
            assumptions.append(f"Assessed documentation quality: {doc_quality}")
    
    has_examples = bool(metadata and (metadata.get("examples") or metadata.get("tutorials"))) if metadata else False
    
    access_time = estimate_access_time(access_type, doc_quality)
    preprocessing_time = estimate_preprocessing_time(modality, format_type, is_multimodal)
    output_time = estimate_first_output_time(modality, has_examples)
    
    total_min = access_time.min_days + preprocessing_time.min_days + output_time.min_days
    total_max = access_time.max_days + preprocessing_time.max_days + output_time.max_days
    
    assumptions.append("Assumes intermediate technical familiarity")
    
    return TTFREstimate(
        total=TimeRange(total_min, total_max),
        access_setup=access_time,
        preprocessing=preprocessing_time,
        first_output=output_time,
        assumptions=assumptions
    )
