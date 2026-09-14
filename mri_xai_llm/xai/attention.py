"""Transformer attention extraction and Attention Rollout (Abnar & Zuidema, 2020).

Assumed interfaces: `mri_xai_llm.models.vision_encoder.MRIVisionEncoder` wraps
either a HuggingFace-style ViT (supports `output_attentions=True` and returns
an object/dict with an `attentions` field — a tuple of Tensor[B, heads,
tokens, tokens] per layer) or a non-transformer (e.g. torchvision CNN)
fallback that has no notion of attention. This module probes for the former
and degrades gracefully (returns None + warning) for the latter.

Explainability disclaimer: outputs of this module indicate the image region
that contributed most strongly to the model's prediction, not proof of
pathology.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

from mri_xai_llm.xai.gradcam import overlay_heatmap

logger = logging.getLogger(__name__)


def extract_attention_maps(model: nn.Module, pixel_values: torch.Tensor) -> Optional[List[torch.Tensor]]:
    """Attempt to run `model` with `output_attentions=True` and collect
    per-layer attention weight tensors. Returns None if unsupported.
    """
    model.eval()
    if pixel_values.dim() == 3:
        pixel_values = pixel_values.unsqueeze(0)

    encoder = model
    if hasattr(model, "get_encoder"):
        try:
            encoder = model.get_encoder()
        except Exception:
            encoder = model

    with torch.no_grad():
        try:
            output = encoder(pixel_values, output_attentions=True)
        except TypeError:
            logger.warning(
                "Underlying vision model does not accept 'output_attentions'; "
                "attention-based explanations are unavailable for this backbone."
            )
            return None
        except Exception as exc:
            logger.warning("Failed to run model with output_attentions=True: %s", exc)
            return None

    attentions = None
    if isinstance(output, dict):
        attentions = output.get("attentions")
    elif hasattr(output, "attentions"):
        attentions = output.attentions
    elif isinstance(output, (tuple, list)):
        for item in output:
            if isinstance(item, (tuple, list)) and len(item) > 0 and torch.is_tensor(item[0]) and item[0].dim() == 4:
                attentions = item
                break

    if attentions is None:
        logger.warning(
            "Model ran with output_attentions=True but no 'attentions' field "
            "was found in its output; attention-based explanations unavailable."
        )
        return None

    return list(attentions)


def attention_rollout(attentions: List[torch.Tensor]) -> np.ndarray:
    """Attention Rollout across transformer layers.

    Averages heads per layer, adds an identity matrix to account for the
    residual (skip) connection around each attention block (otherwise
    rollout would only trace information flow through attention itself and
    ignore the residual path that also carries signal forward), renormalizes
    rows to sum to 1, then chains layers via matrix multiplication. The
    CLS-to-patch row of the resulting joint attention matrix approximates
    how much each patch influences the classification token.
    """
    if not attentions:
        raise ValueError("attentions list is empty")

    joint = None
    for layer_attn in attentions:
        # layer_attn: [B, heads, tokens, tokens]
        avg_heads = layer_attn.mean(dim=1)  # [B, tokens, tokens]
        avg_heads = avg_heads[0].detach().cpu()  # [tokens, tokens], batch=1 assumed

        identity = torch.eye(avg_heads.shape[-1])
        avg_heads = avg_heads + identity
        avg_heads = avg_heads / avg_heads.sum(dim=-1, keepdim=True)

        joint = avg_heads if joint is None else torch.matmul(avg_heads, joint)

    cls_to_patch = joint[0, 1:]  # drop CLS-to-CLS, keep CLS-to-patch
    num_patches = cls_to_patch.shape[0]
    grid_size = int(round(num_patches ** 0.5))
    if grid_size * grid_size != num_patches:
        raise ValueError(
            f"Number of patch tokens ({num_patches}) is not a perfect square; "
            "cannot reshape to a square grid."
        )

    rollout_map = cls_to_patch.reshape(grid_size, grid_size).numpy().astype(np.float32)

    rmin, rmax = rollout_map.min(), rollout_map.max()
    if rmax - rmin > 1e-8:
        rollout_map = (rollout_map - rmin) / (rmax - rmin)
    else:
        rollout_map = np.zeros_like(rollout_map)

    return rollout_map


def _upsample_to_image(attention_map: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    resized = cv2.resize(attention_map, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    rmin, rmax = resized.min(), resized.max()
    if rmax - rmin > 1e-8:
        resized = (resized - rmin) / (rmax - rmin)
    return resized.astype(np.float32)


def visualize_attention(attention_map: np.ndarray, original_image: np.ndarray) -> np.ndarray:
    """Upsample `attention_map` to `original_image` size and overlay it,
    reusing `gradcam.overlay_heatmap` for a consistent visual style."""
    h, w = original_image.shape[:2]
    upsampled = _upsample_to_image(attention_map, h, w)
    return overlay_heatmap(original_image, upsampled, alpha=0.4)
