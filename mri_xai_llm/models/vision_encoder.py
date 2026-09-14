"""MRI vision encoder: HF ViT backbone with a torchvision fallback.

Produces both a pooled embedding (for classification / language grounding)
and a spatial feature map (for Grad-CAM style explanations downstream).
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config.config import MODEL_CONFIG, detect_device

logger = logging.getLogger(__name__)

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:  # pragma: no cover - environment guard
    AutoImageProcessor = None  # type: ignore[assignment,misc]
    AutoModel = None  # type: ignore[assignment,misc]

try:
    import torchvision
    from torchvision.models import ViT_B_16_Weights, vit_b_16
except ImportError:  # pragma: no cover - environment guard
    torchvision = None
    ViT_B_16_Weights = None  # type: ignore[assignment,misc]
    vit_b_16 = None  # type: ignore[assignment,misc]

# ImageNet stats used by the torchvision ViT fallback. HF backbones carry
# their own normalization via AutoImageProcessor when available.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def preprocess_for_encoder(image: np.ndarray, image_size: int = MODEL_CONFIG["image_size"]) -> torch.Tensor:
    """Resize/normalize a single-channel MRI slice into a [1, 3, H, W] tensor.

    Most pretrained vision backbones (medical or ImageNet-derived) expect
    3-channel RGB input, so the single MRI channel is replicated across
    channels rather than retraining a 1-channel stem.
    """
    if image.ndim == 3 and image.shape[-1] in (1, 3):
        image = image[..., 0] if image.shape[-1] == 1 else image.mean(axis=-1)
    image = image.astype(np.float32)

    # Per-image min-max normalize to [0, 1] before applying dataset stats;
    # MRI intensity ranges are scanner/sequence dependent, unlike natural images.
    lo, hi = float(image.min()), float(image.max())
    if hi > lo:
        image = (image - lo) / (hi - lo)
    else:
        image = np.zeros_like(image)

    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False)
    tensor = tensor.repeat(1, 3, 1, 1)  # replicate to 3 channels

    mean = torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    return tensor


class MRIVisionEncoder(nn.Module):
    """Wraps a pretrained ViT-style backbone, frozen by default.

    The encoder is kept frozen during downstream fine-tuning: only the
    lightweight multimodal projector and LoRA adapters on the language model
    are trained, which keeps compute/memory tractable on a single Colab GPU
    and avoids catastrophic forgetting of the pretrained visual features.
    """

    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self.device = detect_device()
        self.backend: str  # "hf" or "torchvision"
        self.hidden_dim: int
        self._hf_model: Any = None
        self._tv_model: Any = None
        self._tv_target_layer: Any = None

        self._load_backbone()
        if freeze:
            for p in self.parameters():
                p.requires_grad = False
            self.eval()

    def _load_backbone(self) -> None:
        hf_name = MODEL_CONFIG["vision_encoder"]
        if AutoModel is not None:
            try:
                logger.info("Loading HF vision encoder '%s' ...", hf_name)
                self._hf_model = AutoModel.from_pretrained(hf_name)
                self.backend = "hf"
                self.hidden_dim = int(getattr(self._hf_model.config, "hidden_size", 768))
                self._hf_model.to(self.device)
                logger.info("Loaded HF vision encoder '%s' (hidden_dim=%d).", hf_name, self.hidden_dim)
                return
            except Exception as exc:  # noqa: BLE001 - any network/weights failure falls back
                logger.warning(
                    "Failed to load HF vision encoder '%s' (%s). Falling back to torchvision ViT.",
                    hf_name, exc,
                )

        self._load_torchvision_fallback()

    def _load_torchvision_fallback(self) -> None:
        if vit_b_16 is None:
            raise RuntimeError(
                "Neither transformers HF backbone nor torchvision vit_b_16 is available; "
                "install `transformers` or `torchvision` to use MRIVisionEncoder."
            )
        try:
            weights = ViT_B_16_Weights.DEFAULT
            self._tv_model = vit_b_16(weights=weights)
        except Exception as exc:  # noqa: BLE001 - offline sandbox: no pretrained weights reachable
            logger.warning("Could not fetch pretrained torchvision ViT weights (%s). Using random init.", exc)
            self._tv_model = vit_b_16(weights=None)

        self.backend = "torchvision"
        self.hidden_dim = self._tv_model.hidden_dim
        # Drop the classification head; we only need the encoder representation.
        self._tv_model.heads = nn.Identity()
        self._tv_model.to(self.device)
        # Last encoder block, used as the Grad-CAM target layer.
        self._tv_target_layer = self._tv_model.encoder.layers[-1]
        logger.info("Loaded torchvision vit_b_16 fallback (hidden_dim=%d).", self.hidden_dim)

    def get_target_layer(self) -> nn.Module:
        """Returns the last attention/conv block for Grad-CAM hook registration."""
        if self.backend == "hf":
            try:
                return self._hf_model.encoder.layer[-1]
            except AttributeError:
                # Some HF vision backbones nest layers differently; fall back to
                # the last named module as a best-effort target.
                return list(self._hf_model.modules())[-1]
        assert self._tv_target_layer is not None
        return self._tv_target_layer

    def _forward_hf(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self._hf_model(pixel_values=pixel_values, output_hidden_states=True)
        last_hidden = outputs.last_hidden_state  # [B, N, D] (N = 1 CLS + patches, or just patches)
        pooled = getattr(outputs, "pooler_output", None)
        if pooled is None:
            pooled = last_hidden[:, 0, :] if last_hidden.shape[1] > 1 else last_hidden.mean(dim=1)

        patch_tokens = last_hidden[:, 1:, :] if last_hidden.shape[1] > 1 else last_hidden
        num_patches = patch_tokens.shape[1]
        side = int(round(num_patches ** 0.5))
        if side * side == num_patches:
            spatial = patch_tokens.transpose(1, 2).reshape(patch_tokens.shape[0], -1, side, side)
        else:
            # Non-square patch grid (rare); keep a [B, D, N, 1] pseudo-spatial map.
            spatial = patch_tokens.transpose(1, 2).unsqueeze(-1)
        return {"pooled": pooled, "spatial": spatial}

    def _forward_torchvision(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        model = self._tv_model
        x = model._process_input(pixel_values)
        n = x.shape[0]
        batch_class_token = model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = model.encoder(x)

        pooled = x[:, 0]
        patch_tokens = x[:, 1:]
        num_patches = patch_tokens.shape[1]
        side = int(round(num_patches ** 0.5))
        if side * side == num_patches:
            spatial = patch_tokens.transpose(1, 2).reshape(n, -1, side, side)
        else:
            spatial = patch_tokens.transpose(1, 2).unsqueeze(-1)
        return {"pooled": pooled, "spatial": spatial}

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        pixel_values = pixel_values.to(self.device)
        if self.backend == "hf":
            return self._forward_hf(pixel_values)
        return self._forward_torchvision(pixel_values)
