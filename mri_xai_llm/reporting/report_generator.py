"""LLM-grounded and deterministic MRI report generation.

Pipeline position: Vision model -> structured findings -> confidence
filtering -> evidence verification -> [THIS MODULE: LLM report generation].

The LLM is never allowed to invent findings: it is only permitted to phrase
and interpret the structured evidence handed to it in the user prompt. If the
LLM's output cannot be parsed/validated into an MRIReport, generation falls
back to a deterministic, template-based report so the pipeline never crashes.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Callable, Optional

from config.config import DISCLAIMER, MIN_CONFIDENCE  # noqa: F401 - MIN_CONFIDENCE kept for callers that need a shared default
from reporting.schemas import (
    Finding,
    ImageQuality,
    MRIReport,
    StudyInfo,
    XAIExplanation,
)

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a research-assistant AI that drafts structured MRI findings summaries \
for an explainable-AI research system. You are NOT a physician and this is NOT a clinical \
diagnostic tool. You must strictly follow these rules:

1. Do not invent findings. Only report findings that are explicitly present in the structured \
evidence provided to you.
2. Do not infer unsupported abnormalities. If the evidence does not support an abnormality, do \
not suggest one exists.
3. Clearly distinguish observations (what the data shows) from interpretations (what it might \
mean) -- never blend the two without labeling which is which.
4. State uncertainty explicitly wherever the evidence is incomplete, low-confidence, or \
ambiguous.
5. Do not provide a definitive clinical diagnosis. You may describe findings and possible \
differential considerations only.
6. Do not invent measurements. Only report dimensions/measurements that are present in the \
structured evidence.
7. Do not invent anatomical locations. Only use locations present in the structured evidence.
8. If image quality is inadequate for confident interpretation, say so explicitly and qualify \
all downstream findings accordingly.
9. If the model is uncertain about a finding, explicitly state that uncertainty rather than \
presenting it with false confidence.
10. Include important limitations of the analysis (model limitations, data limitations, lack of \
clinical correlation, etc.).
11. Do not claim that an XAI heatmap (Grad-CAM, attention, etc.) proves the presence of a \
disease -- it only shows where the model focused, which may or may not correspond to a true \
pathological region.
12. Use professional radiology terminology and structure.
13. Explicitly state that the report must be reviewed by a qualified radiologist before any \
clinical use.

Return structured JSON first and then a human-readable report."""


def build_llm_user_prompt(
    findings: list[Finding],
    xai: list[XAIExplanation],
    study: StudyInfo,
    image_quality: ImageQuality,
) -> str:
    """Serialize the structured evidence into a grounding payload for the LLM.

    The LLM is instructed to use ONLY this payload as its source of truth so
    it cannot invent findings, measurements, or locations not present here.
    """
    payload = {
        "study": study.model_dump(),
        "image_quality": image_quality.model_dump(),
        "findings": [f.model_dump() for f in findings],
        "xai_explanations": [x.model_dump() for x in xai],
    }
    evidence_json = json.dumps(payload, indent=2, ensure_ascii=False)
    return (
        "Below is the ONLY structured evidence available for this study. "
        "Do not use any information outside this payload. Produce an MRIReport-shaped "
        "JSON object (fields: study, image_quality, findings, differential_considerations, "
        "impression, xai, recommendation, medical_disclaimer) followed by a human-readable "
        "report.\n\n"
        f"STRUCTURED EVIDENCE:\n{evidence_json}"
    )


_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def _extract_json_block(text: str) -> Optional[dict]:
    """Best-effort extraction of the first top-level JSON object in text."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidates = [fenced.group(1)] if fenced else []
    match = _JSON_BLOCK_RE.search(text)
    if match:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue
    return None


def generate_report_with_llm(
    model_manager=None,
    findings: list[Finding] | None = None,
    xai: list[XAIExplanation] | None = None,
    study: StudyInfo | None = None,
    image_quality: ImageQuality | None = None,
    tokenizer=None,
    llm_generate_fn: Optional[Callable[[str, str], str]] = None,
) -> MRIReport:
    """Generate an MRIReport by calling an LLM, with a deterministic fallback.

    `llm_generate_fn(system_prompt, user_prompt) -> str` may be supplied directly
    (useful for testing / swapping backends). Otherwise `model_manager` is expected
    to expose a `load_language_model()` -> (model, tokenizer)-like interface plus
    a `.generate(system_prompt, user_prompt)` convenience method. If the LLM call,
    JSON extraction, or pydantic validation fails for any reason, this function
    falls back to `generate_report_deterministic` rather than raising.
    """
    findings = findings or []
    xai = xai or []
    study = study or StudyInfo()
    image_quality = image_quality or ImageQuality()

    user_prompt = build_llm_user_prompt(findings, xai, study, image_quality)

    try:
        if llm_generate_fn is not None:
            raw_text = llm_generate_fn(SYSTEM_PROMPT, user_prompt)
        elif model_manager is not None and hasattr(model_manager, "generate"):
            raw_text = model_manager.generate(SYSTEM_PROMPT, user_prompt)
        elif model_manager is not None and hasattr(model_manager, "load_language_model"):
            llm, tok = model_manager.load_language_model()
            tok = tokenizer or tok
            prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"
            inputs = tok(prompt, return_tensors="pt")
            output_ids = llm.generate(**inputs, max_new_tokens=1024)
            raw_text = tok.decode(output_ids[0], skip_special_tokens=True)
        else:
            raise ValueError("No LLM callable provided (llm_generate_fn or model_manager).")

        parsed = _extract_json_block(raw_text)
        if parsed is None:
            raise ValueError("No parseable JSON object found in LLM output.")
        return MRIReport.model_validate(parsed)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: never let LLM failure crash the pipeline
        logger.warning("LLM report generation failed (%s); falling back to deterministic report.", exc)
        return generate_report_deterministic(findings, xai, study, image_quality)


def generate_report_deterministic(
    findings: list[Finding],
    xai: list[XAIExplanation],
    study: StudyInfo,
    image_quality: ImageQuality,
) -> MRIReport:
    """Assemble a valid MRIReport purely from structured data, without an LLM.

    Guarantees the pipeline produces a usable report even if no LLM is available
    or the LLM call fails.
    """
    scored = sorted(
        findings,
        key=lambda f: (f.model_score if f.model_score is not None else 0.0),
        reverse=True,
    )
    top_findings = [f for f in scored if f.finding.lower() != "normal"][:5]

    if not findings:
        impression = ["No findings were generated by the pipeline for this study."]
    elif not top_findings:
        impression = ["No abnormal findings above threshold; study appears grossly normal."]
    else:
        impression = []
        for f in top_findings:
            qualifier = " (uncertain)" if f.uncertainty else ""
            score_str = f" (score={f.model_score:.2f})" if f.model_score is not None else ""
            impression.append(f"{f.finding} at {f.location}, severity: {f.severity}{score_str}{qualifier}")

    differential = sorted({f.finding for f in findings if f.finding.lower() != "normal"})

    recommendation = (
        "Findings are AI-generated and preliminary. Correlate clinically. "
        "This report must be reviewed and validated by a qualified radiologist/physician "
        "before any clinical decision-making."
    )

    return MRIReport(
        study=study,
        image_quality=image_quality,
        findings=findings,
        differential_considerations=differential,
        impression=impression,
        xai=xai,
        recommendation=recommendation,
        medical_disclaimer=DISCLAIMER,
    )


def _section_or_default(lines: list[str], default: str = "No specific findings reported in this category.") -> str:
    return "\n".join(lines) if lines else default


def render_text_report(report: MRIReport) -> str:
    """Render the human-readable structured text template for an MRIReport."""
    lines: list[str] = []
    lines.append("MRI ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append("")

    lines.append(f"Examination: {report.study.modality} - {report.study.body_region}")
    lines.append(f"Sequences: {', '.join(report.study.sequences) if report.study.sequences else 'Unknown'}")
    lines.append("")

    lines.append("TECHNIQUE")
    lines.append("-" * 60)
    lines.append(
        f"Image quality: {report.image_quality.quality} (score={report.image_quality.score:.2f})"
    )
    if report.image_quality.limitations:
        lines.append("Limitations: " + "; ".join(report.image_quality.limitations))
    lines.append("")

    lines.append("FINDINGS")
    lines.append("-" * 60)

    lines.append(f"1. Brain/Anatomical region: {report.study.body_region}")

    lesion_findings = [
        f for f in report.findings if f.finding.lower() not in ("normal", "unknown")
    ]
    lines.append("2. Lesions:")
    if lesion_findings:
        for i, f in enumerate(lesion_findings, start=1):
            lines.append(f"   Number: {i}")
            lines.append(f"   Approximate location: {f.location}")
            lines.append("   Approximate dimensions: Not measured (no calibrated measurement available)")
            lines.append(f"   Signal characteristics: {f.finding}, severity: {f.severity}")
            if f.uncertainty:
                lines.append(f"   Uncertainty: {f.uncertainty}")
    else:
        lines.append("   No specific findings reported in this category.")

    ventricular = [f for f in report.findings if "ventric" in f.finding.lower() or "hydrocephalus" in f.finding.lower()]
    lines.append("3. Ventricular system:")
    lines.append(_section_or_default([f"   {f.finding} ({f.location})" for f in ventricular]))

    midline = [f for f in report.findings if "midline" in f.finding.lower()]
    lines.append("4. Midline structures:")
    lines.append(_section_or_default([f"   {f.finding} ({f.location})" for f in midline]))

    extra_axial = [f for f in report.findings if "extra-axial" in f.finding.lower() or "hemorrhage" in f.finding.lower()]
    lines.append("5. Extra-axial spaces:")
    lines.append(_section_or_default([f"   {f.finding} ({f.location})" for f in extra_axial]))

    known_ids = {id(f) for f in (lesion_findings + ventricular + midline + extra_axial)}
    other = [f for f in report.findings if id(f) not in known_ids]
    lines.append("6. Other structures:")
    lines.append(_section_or_default([f"   {f.finding} ({f.location})" for f in other]))
    lines.append("")

    lines.append("IMPRESSION")
    lines.append("-" * 60)
    lines.append(_section_or_default([f"  - {imp}" for imp in report.impression]))
    lines.append("")

    lines.append("DIFFERENTIAL CONSIDERATIONS")
    lines.append("-" * 60)
    lines.append(_section_or_default([f"  - {d}" for d in report.differential_considerations]))
    lines.append("")

    lines.append("CONFIDENCE / UNCERTAINTY")
    lines.append("-" * 60)
    uncertainty_lines = [f"  - {f.finding} ({f.location}): {f.uncertainty}" for f in report.findings if f.uncertainty]
    lines.append(_section_or_default(uncertainty_lines, "No specific uncertainty flags reported."))
    lines.append("")

    lines.append("XAI EXPLANATION")
    lines.append("-" * 60)
    xai_lines = [
        f"  - [{x.method}] {x.finding}: regions={', '.join(x.important_regions) or 'none'}. {x.explanation}"
        for x in report.xai
    ]
    lines.append(_section_or_default(xai_lines))
    lines.append(
        "  Note: XAI attribution indicates where the model focused, not confirmed pathology."
    )
    lines.append("")

    lines.append("RECOMMENDATION")
    lines.append("-" * 60)
    lines.append(report.recommendation)
    lines.append("")

    lines.append("=" * 60)
    # Always end with the exact required disclaimer text, regardless of
    # whatever medical_disclaimer text an LLM-produced report may carry.
    lines.append(DISCLAIMER)

    return "\n".join(lines)
