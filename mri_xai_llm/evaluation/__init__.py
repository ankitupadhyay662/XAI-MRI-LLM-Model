"""Evaluation metrics: classification, segmentation/localization, and report text similarity."""
from mri_xai_llm.evaluation.classification import (
    compute_classification_metrics,
    finding_level_metrics,
)
from mri_xai_llm.evaluation.report_metrics import TEXT_METRIC_CAVEAT, compute_text_metrics
from mri_xai_llm.evaluation.segmentation import (
    dice_coefficient,
    hausdorff_distance,
    iou_score,
    localization_overlap,
)

__all__ = [
    "compute_classification_metrics",
    "finding_level_metrics",
    "dice_coefficient",
    "iou_score",
    "hausdorff_distance",
    "localization_overlap",
    "compute_text_metrics",
    "TEXT_METRIC_CAVEAT",
]
