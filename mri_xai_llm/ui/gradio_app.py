"""Gradio Blocks UI for the Explainable MRI AI Research System.

Research/educational prototype -- NOT a clinical device. This module is a
thin presentation layer: all pipeline logic (loading, quality assessment,
vision/XAI inference, report generation) lives in ``main.py`` /
``main.run_pipeline_full`` and is reused here rather than duplicated.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
_PARENT_ROOT = _PROJECT_ROOT.parent
for _p in (str(_PROJECT_ROOT), str(_PARENT_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError as exc:
    raise ImportError(
        "gradio is required for the UI. Install it with `pip install gradio`."
    ) from exc

import numpy as np

try:
    from config import config
except ImportError:
    from mri_xai_llm.config import config  # type: ignore[no-redef]

try:
    import main as pipeline
except ImportError:
    from mri_xai_llm import main as pipeline  # type: ignore[no-redef]

SEQUENCE_CHOICES = list(config.SUPPORTED_SEQUENCES) + ["Auto Detect"]
REGION_CHOICES = ["Brain", "Spine", "Joint", "Other"]

DISCLAIMER_BANNER = f"> **DISCLAIMER:** {config.DISCLAIMER}"


def _to_display_array(arr: Optional["np.ndarray"]) -> Optional["np.ndarray"]:
    """Normalize any float/NaN-containing array to a uint8 image gr.Image can render."""
    if arr is None:
        return None
    arr = np.asarray(arr)
    if arr.dtype == np.uint8:
        return arr
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = float(finite.min()), float(finite.max())
    arr = np.nan_to_num(arr, nan=lo, posinf=hi, neginf=lo)
    if hi - lo < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    return ((arr - lo) / (hi - lo) * 255.0).astype(np.uint8)


def _findings_table(report: "pipeline.schemas.MRIReport") -> list[list[Any]]:
    rows = []
    for f in report.findings:
        confidence = f"{f.model_score:.2f}" if f.model_score is not None else "n/a"
        evidence = f.evidence or ""
        if f.uncertainty:
            evidence = f"{evidence}  [{f.uncertainty}]" if evidence else f"[{f.uncertainty}]"
        rows.append([f.finding, confidence, f.location, evidence])
    return rows


def _xai_markdown(report: "pipeline.schemas.MRIReport", model_message: str) -> str:
    lines = [f"**Model status:** {model_message}", ""]
    if not report.xai:
        lines.append("_No XAI explanations were generated (model unavailable or no findings above display threshold)._")
    for x in report.xai:
        lines.append(f"**[{x.method}] {x.finding}**")
        lines.append(f"Regions: {', '.join(x.important_regions) or 'Unknown'}")
        lines.append(x.explanation)
        lines.append("")
    return "\n\n".join(lines)


def analyze_mri(file: Any, sequence: str, body_region: str) -> tuple:
    """Callback for the ANALYZE MRI button.

    Returns a tuple matching the Blocks output list:
        (status_md, original_img, heatmap_img, seg_img, findings_df,
         report_text, xai_md, txt_download, json_download)
    Never raises -- all failures are surfaced as a status message so the
    Gradio server stays up.
    """
    empty = (
        "Please upload an MRI file (.dcm, .nii, .nii.gz, .png, .jpg, .jpeg, .tiff) and click ANALYZE MRI.",
        None,
        None,
        None,
        [],
        "",
        "",
        None,
        None,
    )
    if file is None:
        return empty

    file_path = file if isinstance(file, str) else getattr(file, "name", None)
    if not file_path:
        return ("Could not read the uploaded file path.", None, None, None, [], "", "", None, None)

    try:
        result = pipeline.run_pipeline_full(file_path, sequence=sequence, body_region=body_region.lower())
    except Exception as exc:  # noqa: BLE001 - the UI must never crash on a bad/unsupported upload
        logger.exception("Pipeline failed for %s", file_path)
        error_msg = (
            f"**Analysis failed:** {exc}\n\n"
            "Check that the file is a valid DICOM/NIfTI/image file. "
            f"{DISCLAIMER_BANNER}"
        )
        return (error_msg, None, None, None, [], "", "", None, None)

    report = result["report"]
    status = f"Analysis complete. {DISCLAIMER_BANNER}"
    original_disp = _to_display_array(result["original_slice"])
    heatmap_disp = _to_display_array(result["heatmap_overlay"])
    seg_disp = _to_display_array(result["segmentation_overlay"])
    findings_rows = _findings_table(report)
    xai_md = _xai_markdown(report, result["model_message"])

    return (
        status,
        original_disp,
        heatmap_disp,
        seg_disp,
        findings_rows,
        result["report_text"],
        xai_md,
        str(result["txt_path"]),
        str(result["json_path"]),
    )


def build_interface() -> "gr.Blocks":
    with gr.Blocks(title="Explainable MRI AI Research System") as demo:
        gr.Markdown("# EXPLAINABLE MRI AI RESEARCH SYSTEM")
        gr.Markdown(DISCLAIMER_BANNER)

        with gr.Row():
            file_input = gr.File(
                label="Upload MRI",
                file_types=[".dcm", ".nii", ".nii.gz", ".png", ".jpg", ".jpeg", ".tiff"],
            )
            sequence_dd = gr.Dropdown(choices=SEQUENCE_CHOICES, value="Auto Detect", label="MRI Sequence")
            region_dd = gr.Dropdown(choices=REGION_CHOICES, value="Brain", label="Body Region")

        analyze_btn = gr.Button("ANALYZE MRI", variant="primary")
        status_md = gr.Markdown()

        with gr.Row():
            original_img = gr.Image(label="Original MRI", type="numpy")
            heatmap_img = gr.Image(label="XAI Heatmap", type="numpy")
        with gr.Row():
            seg_img = gr.Image(label="Segmentation Overlay", type="numpy")

        gr.Markdown("## Findings")
        findings_df = gr.Dataframe(
            headers=["Finding", "Confidence", "Location", "Evidence"],
            datatype=["str", "str", "str", "str"],
            row_count=(0, "dynamic"),
            wrap=True,
        )

        gr.Markdown("## Generated Report")
        report_text = gr.Textbox(label="Report", lines=20, max_lines=40)
        with gr.Row():
            txt_download = gr.File(label="Download report (.txt)")
            json_download = gr.File(label="Download report (.json)")

        gr.Markdown("## XAI Explanation")
        with gr.Tabs():
            with gr.Tab("Grad-CAM"):
                gr.Markdown("Grad-CAM highlights the image region that most strongly influenced the model's finding score.")
            with gr.Tab("Attention Map"):
                gr.Markdown("Attention-based explanations (transformer backbone only). Not populated for CNN/torchvision fallback encoders.")
            with gr.Tab("Occlusion Analysis"):
                gr.Markdown("Occlusion sensitivity: measures the score drop when a region is masked out.")
            xai_md = gr.Markdown()

        gr.Markdown("---")
        gr.Markdown(DISCLAIMER_BANNER)

        analyze_btn.click(
            fn=analyze_mri,
            inputs=[file_input, sequence_dd, region_dd],
            outputs=[
                status_md,
                original_img,
                heatmap_img,
                seg_img,
                findings_df,
                report_text,
                xai_md,
                txt_download,
                json_download,
            ],
        )

    return demo


def launch_demo(share: bool = False) -> None:
    demo = build_interface()
    demo.launch(share=share)


if __name__ == "__main__":
    launch_demo()
