from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class AccessType(Enum):
    OPEN = "open"
    LOGIN = "login"
    APPROVAL = "approval"


class FormatType(Enum):
    BIDS = "bids"
    NWB = "nwb"
    CUSTOM = "custom"
    STANDARD_IMAGE = "standard_image"


class ModalityComplexity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class TimeRange:
    min_days: float
    max_days: float

    def __str__(self) -> str:
        if self.max_days < 1:
            min_hours = round(self.min_days * 24)
            max_hours = round(self.max_days * 24)
            if min_hours == max_hours:
                unit = "hour" if min_hours == 1 else "hours"
                return f"{min_hours} {unit}"
            return f"{min_hours}–{max_hours} hours"
        if self.min_days < 1:
            min_hours = round(self.min_days * 24)
            max_days = round(self.max_days)
            hours_unit = "hour" if min_hours == 1 else "hours"
            days_unit = "day" if max_days == 1 else "days"
            return f"{min_hours} {hours_unit}–{max_days} {days_unit}"
        min_days = round(self.min_days)
        max_days = round(self.max_days)
        if min_days == max_days:
            unit = "day" if min_days == 1 else "days"
            return f"{min_days} {unit}"
        return f"{min_days}–{max_days} days"


@dataclass
class TTFREstimate:
    summary: TimeRange
    phases: Dict[str, TimeRange]
    assumptions: List[str]


ACCESS_DAYS = {
    AccessType.OPEN: (0, 0.5),
    AccessType.LOGIN: (0.5, 2),
    AccessType.APPROVAL: (2, 14),
}

FORMAT_DAYS = {
    FormatType.BIDS: (0.5, 2),
    FormatType.NWB: (0.5, 2),
    FormatType.CUSTOM: (1, 5),
    FormatType.STANDARD_IMAGE: (0.25, 1),
}

MODALITY_DAYS = {
    ModalityComplexity.LOW: (0.25, 1),
    ModalityComplexity.MEDIUM: (0.5, 2),
    ModalityComplexity.HIGH: (1, 3),
    ModalityComplexity.VERY_HIGH: (2, 5),
}

DOC_QUALITY_MULTIPLIER = {"high": 1.0, "medium": 1.2, "low": 1.5}

# Per-datasource defaults: access, format, typical_modality, doc_quality.
# Add or edit entries here when adding new datasources.
DATASOURCE_CONFIG: Dict[str, Dict[str, Any]] = {
    "scr_005031_openneuro": {
        "access": AccessType.OPEN,
        "format": FormatType.BIDS,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high",
    },
    "scr_017571_dandi": {
        "access": AccessType.OPEN,
        "format": FormatType.NWB,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high",
    },
    "scr_007271_modeldb_models": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium",
    },
    "scr_017612_ebrains": {
        "access": AccessType.LOGIN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.VERY_HIGH,
        "doc_quality": "high",
    },
    "scr_003510_cil_images": {
        "access": AccessType.OPEN,
        "format": FormatType.STANDARD_IMAGE,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium",
    },
    "scr_002145_neuromorpho_modelimage": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "high",
    },
    "scr_017041_sparc": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "high",
    },
    "scr_002978_aba_expression": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high",
    },
    "scr_005069_brainminds": {
        "access": AccessType.APPROVAL,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.VERY_HIGH,
        "doc_quality": "medium",
    },
    "scr_002721_gensat_geneexpression": {
        "access": AccessType.OPEN,
        "format": FormatType.STANDARD_IMAGE,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "medium",
    },
    "scr_003105_neurondb_currents": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "medium",
    },
    "scr_006131_hba_atlas": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high",
    },
    "scr_014194_icg_ionchannels": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.LOW,
        "doc_quality": "medium",
    },
    "scr_013705_neuroml_models": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high",
    },
    "scr_014306_bbp_cellmorphology": {
        "access": AccessType.LOGIN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high",
    },
    "scr_016433_conp": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.HIGH,
        "doc_quality": "medium",
    },
    "scr_006274_neuroelectro_ephys": {
        "access": AccessType.OPEN,
        "format": FormatType.CUSTOM,
        "typical_modality": ModalityComplexity.MEDIUM,
        "doc_quality": "high",
    },
}

# Classification checks keywords from VERY_HIGH down to LOW; first match wins.
MODALITY_KEYWORDS = {
    ModalityComplexity.VERY_HIGH: ["multimodal", "multi-modal", "combined", "integrative"],
    ModalityComplexity.HIGH: ["mri", "fmri", "pet", "meg", "eeg", "bold", "neuroimaging"],
    ModalityComplexity.MEDIUM: [
        "microscopy",
        "image",
        "gene expression",
        "single cell",
        "ephys",
        "electrophysiology",
    ],
    ModalityComplexity.LOW: ["simulated", "model", "morphology", "ion channel", "database"],
}


def infer_modality_from_keywords(text: str) -> Optional[ModalityComplexity]:
    if not (text and text.strip()):
        return None
    lower = text.lower()
    for level in (ModalityComplexity.VERY_HIGH, ModalityComplexity.HIGH, ModalityComplexity.MEDIUM, ModalityComplexity.LOW):
        if any(kw in lower for kw in MODALITY_KEYWORDS[level]):
            return level
    return None


def detect_multimodal(text: Optional[str]) -> bool:
    if not text:
        return False
    lower = text.lower()
    keywords = ["multimodal", "multi-modal", "combined", "integrative", " and "]
    return any(kw in lower for kw in keywords)


def assess_documentation_quality(metadata: Optional[Dict[str, Any]]) -> str:
    if not metadata:
        return "low"
    desc = metadata.get("description") or ""
    dc = metadata.get("dc")
    if isinstance(dc, dict):
        desc = desc or dc.get("description") or ""
    links = metadata.get("documentation_url") or metadata.get("url") or metadata.get("identifier")
    if isinstance(links, list):
        links = len(links) > 0
    has_doc_link = bool(links)
    if len(str(desc)) > 200 and has_doc_link:
        return "high"
    if len(str(desc)) > 50 or has_doc_link:
        return "medium"
    return "low"


def estimate_ttfr(
    datasource_id: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    content: Optional[str] = None,
) -> TTFREstimate:
    """
    Estimate time to first result for a datasource.

    Parameters
    ----------
    datasource_id : str, optional
        Known datasource key from DATASOURCE_CONFIG. If present, its access,
        format, typical_modality, and doc_quality are used.
    metadata : dict, optional
        Datasource metadata (e.g. title, description, dc, documentation_url).
        Used for documentation quality and to refine modality.
    content : str, optional
        Free-text description; used to infer modality and multimodality.

    Returns
    -------
    TTFREstimate
        Summary time range, per-phase breakdown, and assumptions.

    Notes
    -----
    If datasource_id is missing or not in DATASOURCE_CONFIG, falls back to
    heuristics from metadata and content. Assumptions list explains how the
    estimate was derived.
    """
    assumptions: List[str] = []
    cfg = DATASOURCE_CONFIG.get(datasource_id or "") if datasource_id else None

    if cfg:
        access = cfg["access"]
        fmt = cfg["format"]
        modality = cfg["typical_modality"]
        doc_quality = cfg.get("doc_quality", "medium")
        assumptions.append(f"Using datasource config for {datasource_id}")
    else:
        access = AccessType.OPEN
        fmt = FormatType.CUSTOM
        modality = ModalityComplexity.MEDIUM
        doc_quality = "medium"
        assumptions.append("Unknown datasource; using default access OPEN, format CUSTOM, modality MEDIUM")
        inferred = infer_modality_from_keywords((content or "") + " " + str(metadata or ""))
        if inferred:
            modality = inferred
            assumptions.append(f"Inferred modality {modality.value} from content/metadata")
        if metadata:
            doc_quality = assess_documentation_quality(metadata)
            assumptions.append(f"Documentation quality: {doc_quality}")

    content_for_multimodal = (content or "") + " " + ((metadata or {}).get("description") or "")
    if detect_multimodal(content_for_multimodal.strip() or None):
        modality = ModalityComplexity.VERY_HIGH
        assumptions.append("Multimodal content detected; using VERY_HIGH modality")

    mult = DOC_QUALITY_MULTIPLIER.get(doc_quality, 1.2)
    a_min, a_max = ACCESS_DAYS[access]
    p_min, p_max = FORMAT_DAYS[fmt]
    m_min, m_max = MODALITY_DAYS[modality]
    p_min, p_max = p_min * mult, p_max * mult
    m_min, m_max = m_min * mult, m_max * mult

    total_min = a_min + p_min + m_min
    total_max = a_max + p_max + m_max
    phases = {
        "access": TimeRange(a_min, a_max),
        "preprocessing": TimeRange(p_min, p_max),
        "first_output": TimeRange(m_min, m_max),
    }
    return TTFREstimate(
        summary=TimeRange(total_min, total_max),
        phases=phases,
        assumptions=assumptions,
    )
