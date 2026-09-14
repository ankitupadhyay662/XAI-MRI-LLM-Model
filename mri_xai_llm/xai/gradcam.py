"""Grad-CAM (Selvaraju et al., 2017) for the MRI vision encoder.

Assumed interfaces (owned by other in-progress modules):
    - `mri_xai_llm.models.vision_encoder.MRIVisionEncoder.forward(pixel_values)`
      returns a dict with keys "pooled" (Tensor[B, D]) and "spatial"
      (Tensor[B, C, H, W]).
    - `MRIVisionEncoder.get_target_layer()` returns the `nn.Module` whose
      output activations/gradients Grad-CAM should hook (typically the last
      convolutional / patch-embedding stage feeding "spatial").
    - The classifier consuming "pooled" (e.g.
      `mri_xai_llm.models.multimodal_model.FindingClassifierHead`) returns
      finding logits/probabilities of shape [B, num_findings]; Grad-CAM here
      is written against any callable `model` whose forward returns either a
      dict containing "finding_logits" (preferred) or "logits", or a raw
      Tensor[B, num_findings], selected defensively via `_extract_logits`.

Inference-time only: no training logic lives in this module.

Explainability disclaimer: outputs of this module indicate the image region
that contributed most strongly to the model's prediction. They never prove
the presence of pathology and must not be presented as diagnostic proof.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

XAI_DISCLAIMER = (
    "indicates the image region that contributed most strongly to the "
    "model's prediction, not proof of pathology"
)


def _extract_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        for key in ("finding_logits", "logits", "pooled"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
        raise KeyError(
            "Model output dict does not contain 'finding_logits', 'logits', "
            "or 'pooled' tensors."
        )
    if torch.is_tensor(output):
        return output
    raise TypeError(f"Unsupported model output type: {type(output)!r}")


class GradCAM:
    """Standard Grad-CAM using forward/backward hooks on a target layer.

    Usage:
        with GradCAM(target_layer) as cam:
            heatmap = cam.generate(model, input_tensor, target_class_idx=3)
    """

    def __init__(self, target_layer: nn.Module) -> None:
        self.target_layer = target_layer
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        try:
            fwd_handle = self.target_layer.register_forward_hook(self._forward_hook)
            bwd_handle = self.target_layer.register_full_backward_hook(self._backward_hook)
            self._handles.extend([fwd_handle, bwd_handle])
        except Exception:
            self.close()
            raise

    def _forward_hook(self, module: nn.Module, inputs: Any, output: Any) -> None:
        activation = output[0] if isinstance(output, (tuple, list)) else output
        self._activations = activation.detach()

    def _backward_hook(self, module: nn.Module, grad_input: Any, grad_output: Any) -> None:
        grad = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        self._gradients = grad.detach()

    def close(self) -> None:
        for handle in self._handles:
            try:
                handle.remove()
            except Exception:
                pass
        self._handles = []

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def generate(
        self,
        model: nn.Module,
        input_tensor: torch.Tensor,
        target_class_idx: int,
    ) -> np.ndarray:
        """Compute a normalized (0-1) 2D Grad-CAM heatmap upsampled to input size."""
        device = next(model.parameters(), input_tensor).device if hasattr(model, "parameters") else input_tensor.device
        model.eval()
        input_tensor = input_tensor.to(device)
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        # The vision encoder is frozen (requires_grad=False on its params), but
        # gradients can still flow back through frozen weights as long as some
        # upstream tensor requires grad. Root the graph at the input so the
        # backward hook below fires regardless of which layers are frozen.
        input_tensor = input_tensor.clone().detach().requires_grad_(True)

        self._activations = None
        self._gradients = None

        model.zero_grad(set_to_none=True)
        output = model(input_tensor)
        logits = _extract_logits(output)
        if logits.dim() == 1:
            logits = logits.unsqueeze(0)

        score = logits[:, target_class_idx].sum()
        score.backward()

        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients; check "
                "that target_layer participates in the forward/backward graph."
            )

        activations = self._activations
        gradients = self._gradients

        if activations.dim() == 4:
            # Conv-style target layer: [B, C, H, W].
            weights = gradients.mean(dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            cam = torch.relu((weights * activations).sum(dim=1))  # [B, H, W]
        elif activations.dim() == 3:
            # Transformer block target layer: [B, N_tokens, D] (ViT-style, CLS
            # token first). Drop the CLS token, treat D as the channel dim and
            # each remaining token as a spatial location, then fold the patch
            # tokens back into a square grid -- same convention used by
            # MRIVisionEncoder to build its "spatial" feature map.
            n_tokens = activations.shape[1]
            has_cls = n_tokens > 1 and int(round((n_tokens - 1) ** 0.5)) ** 2 == n_tokens - 1
            patch_acts = activations[:, 1:, :] if has_cls else activations
            patch_grads = gradients[:, 1:, :] if has_cls else gradients

            weights = patch_grads.mean(dim=1, keepdim=True)  # [B, 1, D] channel importance
            cam_tokens = torch.relu((weights * patch_acts).sum(dim=-1))  # [B, N_patches]

            n_patches = cam_tokens.shape[1]
            side = int(round(n_patches ** 0.5))
            if side * side != n_patches:
                raise RuntimeError(
                    f"Grad-CAM target layer produced {n_patches} patch tokens, which is not a "
                    "perfect square -- cannot fold into a spatial grid."
                )
            cam = cam_tokens.reshape(cam_tokens.shape[0], side, side)
        else:
            raise RuntimeError(
                f"Grad-CAM target layer activations have unsupported shape {tuple(activations.shape)} "
                "(expected 4D [B,C,H,W] conv features or 3D [B,N,D] transformer tokens)."
            )

        cam = cam[0].cpu().numpy().astype(np.float32)

        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        h, w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam_resized = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
        return cam_resized.astype(np.float32)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    """Alpha-blend a 0-1 heatmap (JET colormap) onto a grayscale/RGB base image.

    Returns a uint8 RGB array the same H, W as `image`.
    """
    if image.ndim == 2:
        base_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.ndim == 3 and image.shape[2] == 1:
        base_rgb = cv2.cvtColor(image[:, :, 0], cv2.COLOR_GRAY2RGB)
    else:
        base_rgb = image[:, :, :3]

    if base_rgb.dtype != np.uint8:
        base_min, base_max = base_rgb.min(), base_rgb.max()
        if base_max - base_min > 1e-8:
            base_rgb = ((base_rgb - base_min) / (base_max - base_min) * 255).astype(np.uint8)
        else:
            base_rgb = np.zeros_like(base_rgb, dtype=np.uint8)

    h, w = base_rgb.shape[:2]
    if heatmap.shape != (h, w):
        heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_LINEAR)

    heatmap_uint8 = np.clip(heatmap, 0.0, 1.0)
    heatmap_uint8 = (heatmap_uint8 * 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    overlay = (alpha * heatmap_rgb.astype(np.float32) + (1 - alpha) * base_rgb.astype(np.float32))
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    return overlay
