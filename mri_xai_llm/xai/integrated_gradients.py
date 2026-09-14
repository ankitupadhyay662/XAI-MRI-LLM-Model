"""Integrated Gradients (Sundararajan et al., 2017) via Captum.

Assumed interfaces: see `mri_xai_llm.xai.gradcam` module docstring — the
target `model` is any callable whose forward returns a dict with
"finding_logits" (or "logits"), or a raw Tensor[B, num_findings].

Explainability disclaimer: outputs of this module indicate the image region
that contributed most strongly to the model's prediction, not proof of
pathology.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from captum.attr import IntegratedGradients

from mri_xai_llm.xai.gradcam import _extract_logits, overlay_heatmap


class _LogitExtractorWrapper(nn.Module):
    """Wraps `model` so its forward returns a single scalar-per-sample logit
    vector suitable for Captum (which expects a Tensor output)."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return _extract_logits(output)


def compute_integrated_gradients(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    baseline: Optional[torch.Tensor] = None,
    n_steps: int = 50,
) -> np.ndarray:
    """Compute a normalized (0-1) Integrated Gradients attribution heatmap.

    Attributions are summed across channels and min-max normalized.
    """
    model.eval()
    device = input_tensor.device
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device).requires_grad_(True)

    if baseline is None:
        baseline = torch.zeros_like(input_tensor)
    else:
        baseline = baseline.to(device)
        if baseline.dim() == 3:
            baseline = baseline.unsqueeze(0)

    wrapped = _LogitExtractorWrapper(model)
    ig = IntegratedGradients(wrapped)

    attributions = ig.attribute(
        input_tensor,
        baselines=baseline,
        target=target_class_idx,
        n_steps=n_steps,
    )

    attr = attributions.detach().cpu().numpy()[0]  # [C, H, W]
    attr_map = np.sum(attr, axis=0)  # [H, W]

    attr_min, attr_max = attr_map.min(), attr_map.max()
    if attr_max - attr_min > 1e-8:
        attr_map = (attr_map - attr_min) / (attr_max - attr_min)
    else:
        attr_map = np.zeros_like(attr_map)

    return attr_map.astype(np.float32)


def visualize_attributions(attributions: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """Overlay an Integrated Gradients heatmap onto the original image.

    Reuses `gradcam.overlay_heatmap` for a visually consistent style.
    """
    return overlay_heatmap(original_image, attributions, alpha=0.4)
