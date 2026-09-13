"""Pydantic schemas for structured, validated MRI report output.

These schemas are the contract between the imaging pipeline (vision model,
finding detection, XAI) and the report-generation LLM. The LLM is required
to fill this structure -- it never freely invents fields.
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field, field_validator

Severity = Literal["mild", "moderate", "severe", "unknown"]
Quality = Literal["acceptable", "limited", "non-diagnostic", "unknown"]


class ImageQuality(BaseModel):
    quality: Quality = "unknown"
    score: float = Field(0.0, ge=0.0, le=1.0, description="Model-derived quality score, not clinically validated")
    limitations: list[str] = Field(default_factory=list)


class Finding(BaseModel):
    finding: str
    location: str = "Unknown"
    severity: Severity = "unknown"
    model_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    uncertainty: str = ""
    evidence: str = ""

    @field_validator("finding")
    @classmethod
    def not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("finding must not be empty")
        return v.strip()


class XAIExplanation(BaseModel):
    finding: str
    method: Literal["Grad-CAM", "Integrated Gradients", "Attention", "Occlusion"]
    important_regions: list[str] = Field(default_factory=list)
    explanation: str = ""


class StudyInfo(BaseModel):
    modality: str = "MRI"
    body_region: str = "Unknown"
    sequences: list[str] = Field(default_factory=list)


class MRIReport(BaseModel):
    study: StudyInfo
    image_quality: ImageQuality
    findings: list[Finding] = Field(default_factory=list)
    differential_considerations: list[str] = Field(default_factory=list)
    impression: list[str] = Field(default_factory=list)
    xai: list[XAIExplanation] = Field(default_factory=list)
    recommendation: str = "Radiologist/physician review required."
    medical_disclaimer: str = (
        "Research output only; requires review by a qualified radiologist."
    )

    def to_human_readable(self) -> str:
        from reporting.report_generator import render_text_report
        return render_text_report(self)
