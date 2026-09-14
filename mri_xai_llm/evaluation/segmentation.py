"""Segmentation and XAI-localization evaluation metrics."""
from __future__ import annotations

import numpy as np


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Dice similarity coefficient between two binary masks.

    Returns 1.0 if both masks are empty (trivially agree on "nothing here"),
    and 0.0 if only one of the two masks is empty.
    """
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    denom = pred.sum() + gt.sum()
    if denom == 0:
        return 1.0
    return float(2.0 * intersection / denom)


def iou_score(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Intersection-over-union between two binary masks.

    Returns 1.0 if both masks are empty, 0.0 if the union is empty otherwise
    impossible (guarded for symmetry with dice_coefficient).
    """
    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)
    intersection = np.logical_and(pred, gt).sum()
    union = np.logical_or(pred, gt).sum()
    if union == 0:
        return 1.0
    return float(intersection / union)


def _boundary_points(mask: np.ndarray) -> np.ndarray:
    """Extract boundary (contour) points of a binary mask using cv2.findContours.

    Falls back to all foreground pixel coordinates if cv2 is unavailable or
    finds no contours (e.g. a single-pixel mask).
    """
    mask_u8 = (np.asarray(mask).astype(np.uint8)) * 255
    try:
        import cv2

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            points = np.concatenate([c.reshape(-1, 2) for c in contours], axis=0)
            return points.astype(np.float64)
    except ImportError:
        pass
    return np.argwhere(mask_u8 > 0).astype(np.float64)


def hausdorff_distance(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Symmetric (max of both directed) Hausdorff distance between mask boundaries.

    Note: returns 0.0 if both masks are empty (identical, trivially), and
    float('inf') if exactly one of the two masks is empty (undefined distance
    to a non-existent boundary) rather than raising.
    """
    from scipy.spatial.distance import directed_hausdorff

    pred = np.asarray(pred_mask).astype(bool)
    gt = np.asarray(gt_mask).astype(bool)

    if pred.sum() == 0 and gt.sum() == 0:
        return 0.0
    if pred.sum() == 0 or gt.sum() == 0:
        return float("inf")

    pred_points = _boundary_points(pred)
    gt_points = _boundary_points(gt)
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float("inf")

    d_forward = directed_hausdorff(pred_points, gt_points)[0]
    d_backward = directed_hausdorff(gt_points, pred_points)[0]
    return float(max(d_forward, d_backward))


def localization_overlap(
    saliency_map: np.ndarray,
    gt_mask: np.ndarray,
    threshold: float = 0.5,
) -> float:
    """IoU-style overlap between a thresholded XAI saliency map and a ground-truth
    lesion mask. Used to evaluate explainability faithfulness (how well the model's
    attribution localizes to the true lesion location) -- this does NOT establish
    that the model's underlying reasoning is correct, only spatial agreement.
    """
    saliency = np.asarray(saliency_map, dtype=np.float64)
    s_max = saliency.max()
    if s_max > 0:
        saliency = saliency / s_max
    thresholded = saliency >= threshold
    return iou_score(thresholded, gt_mask)
