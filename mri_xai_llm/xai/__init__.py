"""Explainable AI (XAI) utilities for the MRI vision-language model.

Research/educational prototype -- NOT a clinical device. XAI outputs must
never be described as proof of pathology; any user-facing explanation string
should read as "indicates the image region that contributed most strongly to
the model's prediction," never "proves."
"""

from xai.gradcam import GradCAM, overlay_heatmap
from xai.integrated_gradients import (
    compute_integrated_gradients,
    visualize_attributions,
)
from xai.attention import (
    extract_attention_maps,
    attention_rollout,
    visualize_attention,
)
from xai.occlusion import (
    occlusion_sensitivity,
    deletion_insertion_curve,
)

__all__ = [
    "GradCAM",
    "overlay_heatmap",
    "compute_integrated_gradients",
    "visualize_attributions",
    "extract_attention_maps",
    "attention_rollout",
    "visualize_attention",
    "occlusion_sensitivity",
    "deletion_insertion_curve",
]
