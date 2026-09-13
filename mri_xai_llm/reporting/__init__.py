"""Structured, hallucination-controlled MRI report generation."""
from mri_xai_llm.reporting.findings import (
    apply_confidence_filtering,
    deduplicate_findings,
    verify_evidence,
)
from mri_xai_llm.reporting.report_generator import (
    SYSTEM_PROMPT,
    build_llm_user_prompt,
    generate_report_deterministic,
    generate_report_with_llm,
    render_text_report,
)
from mri_xai_llm.reporting.schemas import (
    Finding,
    ImageQuality,
    MRIReport,
    StudyInfo,
    XAIExplanation,
)

__all__ = [
    "apply_confidence_filtering",
    "deduplicate_findings",
    "verify_evidence",
    "SYSTEM_PROMPT",
    "build_llm_user_prompt",
    "generate_report_deterministic",
    "generate_report_with_llm",
    "render_text_report",
    "Finding",
    "ImageQuality",
    "MRIReport",
    "StudyInfo",
    "XAIExplanation",
]
