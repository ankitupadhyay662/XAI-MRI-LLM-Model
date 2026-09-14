"""CLI / notebook entrypoint for the Explainable MRI AI Research System.

Research/educational prototype -- NOT a clinical device. Wires together:
    load -> quality assessment -> (vision model + findings + XAI, best-effort)
    -> confidence-filtered structured findings -> deterministic report.

Designed to run standalone (``python main.py``) or be imported from a
notebook cell (``from mri_xai_llm.main import run_pipeline``). Every heavy
or sibling-module import is best-effort: if a model download fails (e.g. no
internet in a sandbox) or a sibling module isn't built yet, the pipeline
degrades gracefully instead of crashing.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any, Optional

# --- sys.path bootstrap -----------------------------------------------------
# Other modules in this codebase import siblings two different ways
# (`from config import config` and `from mri_xai_llm.config import config`).
# Adding both this file's directory (the `mri_xai_llm` package root) and its
# parent to sys.path lets both styles resolve as Python 3 namespace packages.
_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent
for _p in (str(_THIS_DIR), str(_PARENT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- sibling module imports (best-effort) -----------------------------------
try:
    from config import config
except ImportError:
    from mri_xai_llm.config import config  # type: ignore[no-redef]

try:
    from reporting import schemas
except ImportError:
    from mri_xai_llm.reporting import schemas  # type: ignore[no-redef]

try:
    from reporting import findings as findings_mod
except ImportError:
    try:
        from mri_xai_llm.reporting import findings as findings_mod  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("reporting.findings unavailable: %s", exc)
        findings_mod = None  # type: ignore[assignment]

try:
    from reporting import report_generator
except ImportError:
    try:
        from mri_xai_llm.reporting import report_generator  # type: ignore[no-redef]
    except ImportError:
        report_generator = None  # type: ignore[assignment]

try:
    from preprocessing import dicom as dicom_mod
except ImportError:
    try:
        from mri_xai_llm.preprocessing import dicom as dicom_mod  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("preprocessing.dicom unavailable: %s", exc)
        dicom_mod = None  # type: ignore[assignment]

try:
    from preprocessing import nifti as nifti_mod
except ImportError:
    try:
        from mri_xai_llm.preprocessing import nifti as nifti_mod  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("preprocessing.nifti unavailable: %s", exc)
        nifti_mod = None  # type: ignore[assignment]

try:
    from preprocessing import transforms
except ImportError:
    try:
        from mri_xai_llm.preprocessing import transforms  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("preprocessing.transforms unavailable: %s", exc)
        transforms = None  # type: ignore[assignment]

try:
    from models import ModelManager
except ImportError:
    try:
        from mri_xai_llm.models import ModelManager  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("models.ModelManager unavailable: %s", exc)
        ModelManager = None  # type: ignore[assignment]

try:
    from models.multimodal_model import FindingClassifierHead
except ImportError:
    try:
        from mri_xai_llm.models.multimodal_model import FindingClassifierHead  # type: ignore[no-redef]
    except ImportError:
        FindingClassifierHead = None  # type: ignore[assignment]

try:
    from xai import gradcam
except ImportError:
    try:
        from mri_xai_llm.xai import gradcam  # type: ignore[no-redef]
    except ImportError as exc:
        logger.warning("xai.gradcam unavailable: %s", exc)
        gradcam = None  # type: ignore[assignment]

try:
    import numpy as np
except ImportError as exc:  # numpy is a hard requirement for this module
    raise ImportError("numpy is required. Install it with `pip install numpy`.") from exc

try:
    import torch
    import torch.nn as nn
except ImportError:
    torch = None
    nn = None

try:
    from PIL import Image
except ImportError:
    Image = None

MODEL_UNAVAILABLE_MSG = (
    "Vision model inference is unavailable in this environment (missing "
    "dependency, no GPU, or no internet access to download weights). "
    "Run this notebook on a GPU-enabled Colab runtime with internet access "
    "for full model-based analysis. The rest of the pipeline (loading, "
    "quality assessment, and report structure) is still fully demoable."
)

_DICOM_EXTS = {".dcm", ".dicom", ""}
_NIFTI_EXTS = {".nii", ".nii.gz"}
_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tiff", ".tif"}


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
def _suffixes(path: Path) -> str:
    return "".join(path.suffixes).lower()


def load_any(path: str | Path) -> tuple["np.ndarray", dict, str]:
    """Load a DICOM file/series, NIfTI volume, or standard image into an array.

    Returns (array, metadata, modality_hint) where modality_hint is one of
    "dicom", "nifti", "image".

    Raises:
        ValueError: If the path does not exist or the extension is unsupported.
        RuntimeError: If the appropriate loader module was not importable.
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"File does not exist: {path}")

    suffix = _suffixes(path)

    if path.is_dir() or suffix in (".dcm", ".dicom", ""):
        if dicom_mod is None:
            raise RuntimeError("preprocessing.dicom is not available in this environment.")
        array, metadata = dicom_mod.load_dicom_series(path)
        return array, metadata, "dicom"

    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        if nifti_mod is None:
            raise RuntimeError("preprocessing.nifti is not available in this environment.")
        array, metadata = nifti_mod.load_nifti(path)
        return array, metadata, "nifti"

    if path.suffix.lower() in _IMAGE_EXTS:
        if Image is None:
            raise RuntimeError("Pillow (PIL) is required to load standard image files.")
        with Image.open(path) as im:
            array = np.asarray(im.convert("L"), dtype=np.float32)
        return array, {"format": path.suffix.lower(), "shape": array.shape}, "image"

    raise ValueError(
        f"Unsupported file type: {path.suffix!r}. Expected DICOM (.dcm), "
        "NIfTI (.nii/.nii.gz), or a standard image (.png/.jpg/.jpeg/.tiff)."
    )


def to_display_slice(array: "np.ndarray", modality_hint: str = "image") -> "np.ndarray":
    """Reduce any loaded array (2D/3D/4D) to a single representative 2D slice for display/inference.

    ``modality_hint`` selects the known "depth" (slice) axis per loader
    convention: DICOM series are stacked as ``(num_slices, rows, cols)``
    (axis 0), NIfTI volumes are typically ``(rows, cols, num_slices)``
    (axis -1). Falls back to a size heuristic for unknown inputs.
    """
    arr = np.asarray(array)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 4:
        arr = arr[..., arr.shape[-1] // 2]
    if arr.ndim == 3:
        if modality_hint == "dicom":
            depth_axis = 0
        elif modality_hint == "nifti":
            depth_axis = arr.ndim - 1
        else:
            depth_axis = int(np.argmin(arr.shape)) if min(arr.shape) < 8 else int(np.argmax(arr.shape))
        moved = np.moveaxis(arr, depth_axis, 0)
        if transforms is not None:
            try:
                idx = transforms.select_representative_slices(
                    np.moveaxis(moved, 0, -1), num_slices=1, axis=2
                )[0]
                return moved[idx]
            except Exception as exc:  # noqa: BLE001 - heuristic slice selection, fall back below
                logger.debug("select_representative_slices failed, using middle slice: %s", exc)
        return moved[moved.shape[0] // 2]
    raise ValueError(f"Cannot reduce array of shape {arr.shape} to a 2D slice.")


def assess_quality(slice_2d: "np.ndarray") -> "schemas.ImageQuality":
    """Best-effort image-quality assessment, degrading gracefully if cv2/transforms are unavailable."""
    if transforms is not None:
        try:
            return transforms.assess_image_quality(slice_2d)
        except Exception as exc:  # noqa: BLE001 - cv2 missing, degenerate slice, etc.
            logger.warning("assess_image_quality failed, using minimal fallback: %s", exc)
        try:
            return transforms.validate_mri_array(slice_2d)
        except Exception as exc:  # noqa: BLE001
            logger.warning("validate_mri_array fallback also failed: %s", exc)
    finite = slice_2d[np.isfinite(slice_2d)]
    if finite.size == 0:
        return schemas.ImageQuality(quality="non-diagnostic", score=0.0, limitations=["Array has no finite voxels."])
    return schemas.ImageQuality(
        quality="unknown",
        score=0.5,
        limitations=["Automated quality assessment unavailable in this environment (missing dependency)."],
    )


# ---------------------------------------------------------------------------
# Vision model + XAI (best-effort, never crashes the caller)
# ---------------------------------------------------------------------------
_ComboBase = nn.Module if nn is not None else object


class _ComboModel(_ComboBase):  # type: ignore[misc]
    """Wraps vision_encoder + finding_head so Grad-CAM sees a single forward(pixel_values) -> logits callable."""

    def __init__(self, vision_encoder: Any, finding_head: Any) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.finding_head = finding_head

    def forward(self, pixel_values: Any) -> dict:
        vision_out = self.vision_encoder(pixel_values)
        head_out = self.finding_head(vision_out["pooled"])
        return {"finding_logits": head_out["finding_logits"], "spatial": vision_out["spatial"]}


def run_vision_and_xai(slice_2d: "np.ndarray") -> dict[str, Any]:
    """Best-effort vision-model finding detection + Grad-CAM.

    Returns a dict with keys: available (bool), message (str), raw_findings
    (list[dict]), heatmap_overlay (np.ndarray | None), xai_explanations
    (list[schemas.XAIExplanation]).

    NOTE: Without a trained checkpoint, the finding classifier head is
    randomly initialized -- its scores are not meaningful predictions, only
    a demonstration of the architecture's data flow (vision -> findings ->
    XAI). This is reflected in the returned message and in low/uncertain
    confidence scores.
    """
    result: dict[str, Any] = {
        "available": False,
        "message": MODEL_UNAVAILABLE_MSG,
        "raw_findings": [],
        "heatmap_overlay": None,
        "xai_explanations": [],
    }

    if torch is None or ModelManager is None or FindingClassifierHead is None:
        return result

    try:
        from models.vision_encoder import preprocess_for_encoder
    except ImportError:
        try:
            from mri_xai_llm.models.vision_encoder import preprocess_for_encoder  # type: ignore[no-redef]
        except ImportError as exc:
            result["message"] = f"{MODEL_UNAVAILABLE_MSG} (import error: {exc})"
            return result

    try:
        mm = ModelManager()
        vision_encoder = mm.load_vision_model()
        finding_head = FindingClassifierHead(vision_encoder.hidden_dim)
        finding_head.eval()
        combo = _ComboModel(vision_encoder, finding_head).to(mm.device)
        combo.eval()

        input_tensor = preprocess_for_encoder(slice_2d).to(mm.device)
        # The vision encoder is frozen (requires_grad=False on its params); Grad-CAM
        # still needs a gradient-tracked forward pass to hook activations/gradients
        # on an internal layer, so the graph is rooted at the input instead.
        input_tensor.requires_grad_(True)

        with torch.no_grad():
            out = combo(input_tensor)
            probs = torch.sigmoid(out["finding_logits"])[0].cpu().numpy()

        taxonomy = finding_head.taxonomy
        top_indices = np.argsort(probs)[::-1][:5]
        raw_findings = []
        for idx in top_indices:
            raw_findings.append(
                {
                    "finding": taxonomy[int(idx)],
                    "location": config.ANATOMICAL_REGIONS[0] if config.ANATOMICAL_REGIONS else "Unknown",
                    "severity": "unknown",
                    "model_score": float(probs[idx]),
                    "evidence": (
                        "Vision-model attribution score from an untrained research-demo "
                        "classifier head (no clinical checkpoint loaded)."
                    ),
                    "uncertainty": "",
                }
            )

        heatmap_overlay = None
        xai_explanations: list[Any] = []
        if gradcam is not None:
            try:
                target_layer = vision_encoder.get_target_layer()
                with gradcam.GradCAM(target_layer) as cam:
                    top_idx = int(top_indices[0])
                    heatmap = cam.generate(combo, input_tensor, target_class_idx=top_idx)
                heatmap_overlay = gradcam.overlay_heatmap(slice_2d.astype(np.float32), heatmap, alpha=0.4)
                xai_explanations.append(
                    schemas.XAIExplanation(
                        finding=raw_findings[0]["finding"],
                        method="Grad-CAM",
                        important_regions=[raw_findings[0]["location"]],
                        explanation=(
                            f"Grad-CAM {gradcam.XAI_DISCLAIMER}. Generated from an untrained "
                            "research-demo model; not a validated attribution."
                        ),
                    )
                )
            except Exception as exc:  # noqa: BLE001 - Grad-CAM is best-effort
                logger.warning("Grad-CAM failed: %s", exc)

        result.update(
            available=True,
            message=(
                "Inference ran using an UNTRAINED research-demo classifier head "
                "(no clinical checkpoint loaded); scores are architecture "
                "demonstrations only, not validated predictions."
            ),
            raw_findings=raw_findings,
            heatmap_overlay=heatmap_overlay,
            xai_explanations=xai_explanations,
        )
        return result
    except Exception as exc:  # noqa: BLE001 - model loading/inference must never crash the UI
        logger.warning("Vision model pipeline unavailable: %s", exc)
        result["message"] = f"{MODEL_UNAVAILABLE_MSG} (error: {exc})"
        return result


# ---------------------------------------------------------------------------
# Report assembly (uses reporting.report_generator if available, else a local fallback)
# ---------------------------------------------------------------------------
def _local_render_text_report(report: "schemas.MRIReport") -> str:
    lines = [
        "=" * 70,
        "EXPLAINABLE MRI AI RESEARCH SYSTEM -- RESEARCH REPORT",
        "=" * 70,
        f"Modality: {report.study.modality}  Region: {report.study.body_region}",
        f"Sequences: {', '.join(report.study.sequences) or 'Unknown'}",
        "",
        f"Image quality: {report.image_quality.quality} (score={report.image_quality.score})",
    ]
    for lim in report.image_quality.limitations:
        lines.append(f"  - {lim}")
    lines.append("")
    lines.append("FINDINGS:")
    if not report.findings:
        lines.append("  (none detected)")
    for f in report.findings:
        lines.append(f"  - {f.finding} [{f.severity}] @ {f.location} "
                      f"(score={f.model_score if f.model_score is not None else 'n/a'})")
        if f.evidence:
            lines.append(f"      evidence: {f.evidence}")
        if f.uncertainty:
            lines.append(f"      note: {f.uncertainty}")
    lines.append("")
    lines.append("IMPRESSION:")
    for imp in report.impression or ["No impression generated."]:
        lines.append(f"  - {imp}")
    lines.append("")
    lines.append("XAI EXPLANATIONS:")
    for x in report.xai:
        lines.append(f"  - [{x.method}] {x.finding}: {x.explanation}")
    lines.append("")
    lines.append(f"RECOMMENDATION: {report.recommendation}")
    lines.append("")
    lines.append(report.medical_disclaimer)
    lines.append("=" * 70)
    return "\n".join(lines)


def build_report(
    findings_list: list["schemas.Finding"],
    xai_explanations: list["schemas.XAIExplanation"],
    study: "schemas.StudyInfo",
    image_quality: "schemas.ImageQuality",
) -> "schemas.MRIReport":
    """Builds an MRIReport, preferring reporting.report_generator when available."""
    if report_generator is not None and hasattr(report_generator, "generate_report_deterministic"):
        try:
            return report_generator.generate_report_deterministic(
                findings=findings_list, xai=xai_explanations, study=study, image_quality=image_quality
            )
        except Exception as exc:  # noqa: BLE001 - fall back to local assembly
            logger.warning("generate_report_deterministic failed, using local fallback: %s", exc)

    impression = [f"{f.finding} ({f.severity}) at {f.location}" for f in findings_list] or [
        "No findings surfaced by the pipeline."
    ]
    return schemas.MRIReport(
        study=study,
        image_quality=image_quality,
        findings=findings_list,
        differential_considerations=[],
        impression=impression,
        xai=xai_explanations,
        recommendation="Radiologist/physician review required.",
        medical_disclaimer=config.DISCLAIMER,
    )


def render_report_text(report: "schemas.MRIReport") -> str:
    if report_generator is not None and hasattr(report_generator, "render_text_report"):
        try:
            return report_generator.render_text_report(report)
        except Exception as exc:  # noqa: BLE001
            logger.warning("render_text_report failed, using local fallback: %s", exc)
    return _local_render_text_report(report)


def save_report_files(report: "schemas.MRIReport", report_text: str, out_dir: Optional[Path] = None) -> tuple[Path, Path]:
    """Writes the report as .txt and .json to out_dir (default: config.SAMPLE_DIR or a temp dir)."""
    if out_dir is None:
        try:
            out_dir = config.SAMPLE_DIR / "reports"
        except Exception:  # noqa: BLE001
            out_dir = Path(tempfile.gettempdir()) / "mri_xai_llm_reports"
    out_dir.mkdir(parents=True, exist_ok=True)

    stem = f"report_{uuid.uuid4().hex[:8]}"
    txt_path = out_dir / f"{stem}.txt"
    json_path = out_dir / f"{stem}.json"
    txt_path.write_text(report_text, encoding="utf-8")
    json_path.write_text(json.dumps(report.model_dump(), indent=2), encoding="utf-8")
    return txt_path, json_path


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_pipeline_full(image_path: str, sequence: str = "Unknown", body_region: str = "brain") -> dict[str, Any]:
    """Runs the full load -> quality -> vision/XAI -> report pipeline.

    Returns a dict with keys: report (MRIReport), report_text (str),
    original_slice (np.ndarray), heatmap_overlay (np.ndarray | None),
    segmentation_overlay (np.ndarray | None), model_message (str),
    txt_path (Path), json_path (Path).
    """
    array, metadata, modality_hint = load_any(image_path)
    original_slice = to_display_slice(array, modality_hint=modality_hint)
    image_quality = assess_quality(original_slice)

    vision_result = run_vision_and_xai(original_slice)

    if findings_mod is not None:
        try:
            findings_list = findings_mod.apply_confidence_filtering(
                vision_result["raw_findings"], min_confidence=config.MIN_CONFIDENCE
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("apply_confidence_filtering failed: %s", exc)
            findings_list = [schemas.Finding(**f) for f in vision_result["raw_findings"]]
    else:
        findings_list = [
            schemas.Finding(
                finding=f["finding"],
                location=f["location"],
                severity=f.get("severity", "unknown"),
                model_score=f.get("model_score"),
                evidence=f.get("evidence", ""),
                uncertainty=f.get("uncertainty", ""),
            )
            for f in vision_result["raw_findings"]
        ]

    sequences = [sequence] if sequence and sequence != "Auto Detect" else []
    study = schemas.StudyInfo(modality="MRI", body_region=body_region, sequences=sequences)

    report = build_report(findings_list, vision_result["xai_explanations"], study, image_quality)
    report_text = render_report_text(report)
    txt_path, json_path = save_report_files(report, report_text)

    return {
        "report": report,
        "report_text": report_text,
        "original_slice": original_slice,
        "heatmap_overlay": vision_result["heatmap_overlay"],
        "segmentation_overlay": None,
        "model_message": vision_result["message"],
        "modality_hint": modality_hint,
        "metadata": metadata,
        "txt_path": txt_path,
        "json_path": json_path,
    }


def run_pipeline(image_path: str, sequence: str = "Unknown", body_region: str = "brain") -> "schemas.MRIReport":
    """Notebook-friendly entrypoint: runs the pipeline and returns just the MRIReport."""
    return run_pipeline_full(image_path, sequence=sequence, body_region=body_region)["report"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Explainable MRI AI Research System Gradio demo.")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio share link.")
    args = parser.parse_args()

    try:
        from ui.gradio_app import launch_demo
    except ImportError:
        from mri_xai_llm.ui.gradio_app import launch_demo  # type: ignore[no-redef]

    launch_demo(share=args.share)


if __name__ == "__main__":
    main()
