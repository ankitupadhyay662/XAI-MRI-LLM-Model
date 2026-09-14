"""Confidence filtering, evidence verification, and deduplication of findings.

Implements the hallucination-control middle stages of the pipeline:
    Vision model -> structured findings -> confidence filtering ->
    evidence verification -> LLM report generation.

Findings are never silently dropped for low confidence -- they are always
surfaced, but flagged as uncertain so a human reviewer can triage them.
"""
from __future__ import annotations

from config.config import MIN_CONFIDENCE
from reporting.schemas import Finding

_UNCERTAIN_NOTE = "Uncertain finding — requires expert review"
_XAI_MISMATCH_NOTE = "location not corroborated by XAI attribution; treat with caution."

# severities more specific than "unknown"/"mild" that get clamped down when
# the underlying model confidence does not support such specificity
_SEVERITY_RANK = {"unknown": 0, "mild": 1, "moderate": 2, "severe": 3}


def _append_note(existing: str, note: str) -> str:
    if not existing:
        return note
    if note in existing:
        return existing
    return f"{existing}; {note}"


def apply_confidence_filtering(
    raw_findings: list[dict],
    min_confidence: float = MIN_CONFIDENCE,
) -> list[Finding]:
    """Convert raw detector output dicts into validated, confidence-flagged Findings.

    Findings whose model_score is below min_confidence are NOT dropped: they
    are surfaced with an explicit uncertainty note, and an over-specific
    severity is clamped toward "unknown" since the evidence does not support it.
    """
    results: list[Finding] = []
    for raw in raw_findings:
        finding = Finding(
            finding=raw.get("finding", "Unknown"),
            location=raw.get("location", "Unknown"),
            severity=raw.get("severity", "unknown"),
            model_score=raw.get("model_score"),
            evidence=raw.get("evidence", ""),
            uncertainty=raw.get("uncertainty", ""),
        )
        score = finding.model_score
        if score is None or score < min_confidence:
            finding.uncertainty = _append_note(finding.uncertainty, _UNCERTAIN_NOTE)
            # clamp severity toward "unknown" if the evidence-supported rank
            # (proportional to confidence) is lower than the claimed severity
            if score is not None:
                supported_rank = round(score * 3)  # 0..3 mapped onto severity ranks
            else:
                supported_rank = 0
            if _SEVERITY_RANK.get(finding.severity, 0) > supported_rank:
                finding.severity = "unknown"
        results.append(finding)
    return results


def verify_evidence(finding: Finding, xai_regions: list[str]) -> Finding:
    """Cross-check that a finding's claimed location overlaps XAI attribution.

    If there's no overlap between the finding's location and the XAI-derived
    important regions, appends a caution note to uncertainty rather than
    modifying or discarding the finding.
    """
    if not xai_regions:
        return finding
    location_lower = finding.location.lower().strip()
    overlap = any(
        location_lower in region.lower() or region.lower() in location_lower
        for region in xai_regions
        if region
    )
    if not overlap and location_lower not in ("", "unknown"):
        finding.uncertainty = _append_note(finding.uncertainty, _XAI_MISMATCH_NOTE)
    return finding


def deduplicate_findings(findings: list[Finding]) -> list[Finding]:
    """Merge near-duplicate findings (same finding + location), keeping the
    higher-confidence (higher model_score) instance of each."""
    best: dict[tuple[str, str], Finding] = {}
    for f in findings:
        key = (f.finding.strip().lower(), f.location.strip().lower())
        existing = best.get(key)
        if existing is None:
            best[key] = f
            continue
        existing_score = existing.model_score if existing.model_score is not None else -1.0
        new_score = f.model_score if f.model_score is not None else -1.0
        if new_score > existing_score:
            best[key] = f
    return list(best.values())
