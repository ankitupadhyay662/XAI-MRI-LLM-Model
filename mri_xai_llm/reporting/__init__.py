"""Structured, hallucination-controlled MRI report generation."""
from reporting.findings import (
    apply_confidence_filtering,
    deduplicate_findings,
    verify_evidence,
)
from reporting.report_generator import (
    SYSTEM_PROMPT,
    build_llm_user_prompt,
    generate_report_deterministic,
    generate_report_with_llm,
    render_text_report,
)
from reporting.schemas import (
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
