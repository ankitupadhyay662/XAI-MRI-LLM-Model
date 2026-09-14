"""Occlusion sensitivity and deletion/insertion faithfulness curves.

Assumed interfaces: see `mri_xai_llm.xai.gradcam` module docstring — `model`
forward returns a dict with "finding_logits" (or "logits"), or a raw
Tensor[B, num_findings], selected via `_extract_logits`.

Deletion/insertion curves (Petsiuk et al., "RISE", 2018) measure whether a
saliency map is *faithful* to the model, not merely whether it looks
plausible: deletion asks "how fast does the predicted score collapse as we
remove the pixels the map claims are most important" (a faithful map should
cause a fast drop), and insertion asks the mirror question building the
image back up from a blank baseline. Area-under-curve on these summarizes
faithfulness independently of visual appeal.

Explainability disclaimer: outputs of this module indicate the image region
that contributed most strongly to the model's prediction, not proof of
pathology.
"""

from __future__ import annotations

from typing import Literal, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn

from xai.gradcam import _extract_logits


def _target_score(model: nn.Module, batch: torch.Tensor, target_class_idx: int) -> torch.Tensor:
    with torch.no_grad():
        output = model(batch)
    logits = _extract_logits(output)
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)
    return torch.sigmoid(logits[:, target_class_idx])


def occlusion_sensitivity(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    patch_size: int = 16,
    stride: int = 8,
    baseline_value: float = 0.0,
) -> np.ndarray:
    """Occlude sliding square patches and record the drop in target-class
    score at each location, upsampled to a full-resolution 0-1 map."""
    model.eval()
    device = input_tensor.device
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    _, c, h, w = input_tensor.shape

    with torch.no_grad():
        base_output = model(input_tensor)
    base_logits = _extract_logits(base_output)
    if base_logits.dim() == 1:
        base_logits = base_logits.unsqueeze(0)
    base_score = torch.sigmoid(base_logits[:, target_class_idx]).item()

    ys = list(range(0, max(h - patch_size, 0) + 1, stride))
    xs = list(range(0, max(w - patch_size, 0) + 1, stride))
    if not ys:
        ys = [0]
    if not xs:
        xs = [0]

    grid = np.zeros((len(ys), len(xs)), dtype=np.float32)

    occluded_batch = []
    coords = []
    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            occluded = input_tensor.clone()
            y_end = min(y + patch_size, h)
            x_end = min(x + patch_size, w)
            occluded[:, :, y:y_end, x:x_end] = baseline_value
            occluded_batch.append(occluded)
            coords.append((i, j))

    batch_size = 32
    scores = []
    for start in range(0, len(occluded_batch), batch_size):
        chunk = torch.cat(occluded_batch[start:start + batch_size], dim=0)
        with torch.no_grad():
            out = model(chunk)
        logits = _extract_logits(out)
        chunk_scores = torch.sigmoid(logits[:, target_class_idx]).cpu().numpy()
        scores.extend(chunk_scores.tolist())

    for (i, j), score in zip(coords, scores):
        grid[i, j] = base_score - score  # positive = important (score dropped)

    gmin, gmax = grid.min(), grid.max()
    if gmax - gmin > 1e-8:
        grid = (grid - gmin) / (gmax - gmin)
    else:
        grid = np.zeros_like(grid)

    sensitivity_map = cv2.resize(grid, (w, h), interpolation=cv2.INTER_LINEAR)
    return sensitivity_map.astype(np.float32)


def deletion_insertion_curve(
    model: nn.Module,
    input_tensor: torch.Tensor,
    target_class_idx: int,
    saliency_map: np.ndarray,
    n_steps: int = 20,
    mode: Literal["deletion", "insertion"] = "deletion",
) -> Tuple[np.ndarray, np.ndarray]:
    """Progressively remove (deletion) or reveal (insertion) the most-salient
    pixels per `saliency_map` and record the target-class score at each
    fraction, for computing an AUC faithfulness metric."""
    model.eval()
    device = input_tensor.device
    if input_tensor.dim() == 3:
        input_tensor = input_tensor.unsqueeze(0)
    input_tensor = input_tensor.to(device)

    _, c, h, w = input_tensor.shape

    if saliency_map.shape != (h, w):
        saliency_map = cv2.resize(saliency_map, (w, h), interpolation=cv2.INTER_LINEAR)

    flat_order = np.argsort(-saliency_map.flatten())  # descending importance
    total_pixels = flat_order.shape[0]

    if mode == "deletion":
        canvas = input_tensor.clone()
    else:
        blurred = cv2.GaussianBlur(
            input_tensor[0].permute(1, 2, 0).cpu().numpy(), (31, 31), sigmaX=10
        )
        canvas = torch.from_numpy(blurred).permute(2, 0, 1).unsqueeze(0).to(device).to(input_tensor.dtype)

    fractions = np.linspace(0.0, 1.0, n_steps + 1)
    scores = np.zeros(n_steps + 1, dtype=np.float32)

    mask_flat = np.zeros(total_pixels, dtype=bool)

    for step, frac in enumerate(fractions):
        n_pixels = int(round(frac * total_pixels))
        mask_flat[:] = False
        mask_flat[flat_order[:n_pixels]] = True
        mask = mask_flat.reshape(h, w)
        mask_tensor = torch.from_numpy(mask).to(device)

        if mode == "deletion":
            step_input = input_tensor.clone()
            for ch in range(c):
                step_input[0, ch][mask_tensor] = 0.0
        else:
            step_input = canvas.clone()
            for ch in range(c):
                step_input[0, ch][mask_tensor] = input_tensor[0, ch][mask_tensor]

        score = _target_score(model, step_input, target_class_idx)
        scores[step] = score.item()

    return fractions.astype(np.float32), scores
