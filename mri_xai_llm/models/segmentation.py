"""Lesion segmentation: a MONAI 2D UNet research-demo head.

NOTE: Unless a trained checkpoint is explicitly loaded via ``torch.load`` and
``model.load_state_dict``, this network is randomly initialized and its
output masks carry no clinical meaning. It exists to demonstrate the
segmentation stage of the architecture (and to feed lesion-metric structured
evidence to the LLM), not to provide accurate lesion boundaries.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from config.config import detect_device

logger = logging.getLogger(__name__)

try:
    from monai.networks.nets import UNet
    from monai.networks.layers import Norm
except ImportError as exc:  # pragma: no cover - environment guard
    UNet = None  # type: ignore[assignment,misc]
    Norm = None  # type: ignore[assignment,misc]
    _MONAI_IMPORT_ERROR = exc
else:
    _MONAI_IMPORT_ERROR = None

try:
    from scipy import ndimage
except ImportError:  # pragma: no cover - environment guard
    ndimage = None


class LesionSegmentationModel(nn.Module):
    """Thin wrapper around ``monai.networks.nets.UNet`` for 2D MRI slices.

    Randomly initialized by default (research/architecture demonstration).
    Load a checkpoint via ``load_state_dict`` before trusting any output.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        channels: tuple[int, ...] = (16, 32, 64, 128, 256),
        strides: tuple[int, ...] = (2, 2, 2, 2),
        num_res_units: int = 2,
    ) -> None:
        super().__init__()
        if UNet is None:
            raise ImportError(
                "MONAI is required for LesionSegmentationModel. "
                f"Original import error: {_MONAI_IMPORT_ERROR}"
            )
        self.net = UNet(
            spatial_dims=2,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
            norm=Norm.BATCH,
        )
        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        if self.out_channels == 1:
            return torch.sigmoid(logits)
        return torch.softmax(logits, dim=1)


def segment(
    model: LesionSegmentationModel,
    volume_slice: np.ndarray,
    threshold: float | None = 0.5,
) -> np.ndarray:
    """Runs the segmentation model on a single 2D slice.

    Returns a soft probability mask if ``threshold`` is None, otherwise a
    binary mask (uint8, values {0,1}).
    """
    device = detect_device()
    model = model.to(device)
    model.eval()

    arr = volume_slice.astype(np.float32)
    lo, hi = float(arr.min()), float(arr.max())
    arr = (arr - lo) / (hi - lo) if hi > lo else np.zeros_like(arr)

    tensor = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,H,W]
    with torch.no_grad():
        prob = model(tensor)[0, 0].cpu().numpy()

    if threshold is None:
        return prob.astype(np.float32)
    return (prob >= threshold).astype(np.uint8)


def compute_lesion_metrics(
    mask: np.ndarray,
    voxel_spacing: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Computes approximate structural metrics from a binary/soft lesion mask.

    All values are ESTIMATES derived from a research-demo (possibly
    untrained) segmentation model and must not be treated as measured
    clinical quantities. Every key is prefixed ``approx_`` to make this
    explicit to downstream consumers (LLM prompt builders, reports).
    """
    binary_mask = (mask >= 0.5).astype(np.uint8) if mask.dtype != np.uint8 else mask

    if ndimage is not None:
        labeled, num_components = ndimage.label(binary_mask)
    else:  # pragma: no cover - scipy should normally be present
        logger.warning("scipy is unavailable; falling back to a trivial single-component estimate.")
        labeled = binary_mask
        num_components = 1 if binary_mask.any() else 0

    pixel_area = int(binary_mask.sum())

    if voxel_spacing is not None:
        # 2D slice: use the in-plane spacing (x, y) for area/volume-per-slice.
        px_area_mm2 = float(voxel_spacing[0] * voxel_spacing[1])
        approx_volume_mm3 = float(pixel_area * px_area_mm2 * voxel_spacing[2])
        approx_area_or_volume = {"approx_volume_mm3": approx_volume_mm3}
    else:
        approx_area_or_volume = {"approx_pixel_area": pixel_area}

    bbox: dict[str, int] | None = None
    if binary_mask.any():
        ys, xs = np.where(binary_mask > 0)
        bbox = {
            "approx_bbox_min_row": int(ys.min()),
            "approx_bbox_max_row": int(ys.max()),
            "approx_bbox_min_col": int(xs.min()),
            "approx_bbox_max_col": int(xs.max()),
            "approx_bbox_height": int(ys.max() - ys.min() + 1),
            "approx_bbox_width": int(xs.max() - xs.min() + 1),
        }

    return {
        "approx_lesion_count": int(num_components),
        "approx_pixel_area": pixel_area,
        **approx_area_or_volume,
        "approx_bbox": bbox,
    }
