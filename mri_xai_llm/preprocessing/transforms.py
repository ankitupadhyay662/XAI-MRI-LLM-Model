"""MRI intensity normalization, quality assessment, and MONAI transform pipeline.

All quality/orientation scoring here is heuristic and model-free (simple
statistics + classical image processing). It is NOT a clinically validated
diagnostic quality metric -- it exists only to flag obviously unusable
inputs before they reach the vision model.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

try:
    import cv2
except ImportError as exc:  # pragma: no cover - environment guard
    cv2 = None
    _CV2_IMPORT_ERROR = exc
else:
    _CV2_IMPORT_ERROR = None

try:
    import nibabel as nib
except ImportError:
    nib = None

from reporting.schemas import ImageQuality

NormalizationMethod = Literal["minmax", "zscore", "percentile"]


def build_mri_transform_pipeline(image_size: int = 224):
    """Build a MONAI ``Compose`` pipeline for preparing an MRI volume for a vision model.

    Import of MONAI is deferred to call time so that this module (and the
    rest of the ``preprocessing`` package) remains importable in
    environments where MONAI is not yet installed. Raises ``ImportError``
    with an actionable message only when this function is actually called.

    Pipeline: EnsureChannelFirst -> Orientation(RAS) -> Spacing (resample to
    isotropic 1mm) -> ScaleIntensityRangePercentiles (robust to bright-voxel
    outliers common in MRI) -> Resize to ``(image_size, image_size)``.

    Args:
        image_size: Target square spatial size for the final resize step.

    Returns:
        A ``monai.transforms.Compose`` instance.

    Raises:
        ImportError: If MONAI is not installed.
    """
    try:
        from monai.transforms import (
            Compose,
            EnsureChannelFirst,
            Orientation,
            Resize,
            ScaleIntensityRangePercentiles,
            Spacing,
        )
    except ImportError as exc:
        raise ImportError(
            "MONAI is required for build_mri_transform_pipeline but is not "
            "installed. Install it with `pip install monai`."
        ) from exc

    return Compose(
        [
            EnsureChannelFirst(channel_dim="no_channel"),
            Orientation(axcodes="RAS"),
            Spacing(pixdim=(1.0, 1.0, 1.0), mode="bilinear"),
            ScaleIntensityRangePercentiles(
                lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True
            ),
            Resize(spatial_size=(image_size, image_size, -1), mode="trilinear"),
        ]
    )


def normalize_intensity(array: np.ndarray, method: NormalizationMethod = "percentile") -> np.ndarray:
    """Normalize MRI voxel intensities to roughly [0, 1] using one of three strategies.

    - ``"percentile"`` (recommended default): clips to the 1st/99th intensity
      percentiles before min-max scaling. MRI scanners frequently produce a
      handful of very bright voxels (e.g. flow artifacts, fat, metal), so
      clipping outliers before scaling is the most robust choice for
      preserving contrast in the diagnostically relevant tissue range.
    - ``"zscore"``: subtracts the mean and divides by the standard deviation.
      Appropriate when downstream steps assume roughly Gaussian-distributed
      tissue intensities (e.g. some statistical/registration pipelines), but
      is still sensitive to extreme outliers pulling the mean/std.
    - ``"minmax"``: simple linear rescale using the true min/max. Fastest,
      but a single outlier voxel can compress the rest of the dynamic range
      to near-zero -- least robust of the three for raw MRI.

    Args:
        array: Input array of any shape.
        method: Normalization strategy to use.

    Returns:
        A float32 array of the same shape as ``array``.

    Raises:
        ValueError: If ``method`` is not one of the supported options, or
            the array is empty / has zero variance where required.
    """
    if array.size == 0:
        raise ValueError("Cannot normalize an empty array.")

    arr = array.astype(np.float32)

    if method == "minmax":
        lo, hi = float(np.min(arr)), float(np.max(arr))
        if hi - lo < 1e-8:
            raise ValueError(
                "minmax normalization failed: array has (near) zero intensity range "
                f"(min={lo}, max={hi})."
            )
        return (arr - lo) / (hi - lo)

    if method == "zscore":
        mean, std = float(np.mean(arr)), float(np.std(arr))
        if std < 1e-8:
            raise ValueError(
                f"zscore normalization failed: array has (near) zero standard deviation ({std})."
            )
        return (arr - mean) / std

    if method == "percentile":
        p1, p99 = np.percentile(arr, [1, 99])
        if p99 - p1 < 1e-8:
            raise ValueError(
                f"percentile normalization failed: 1st/99th percentiles are (near) "
                f"identical (p1={p1}, p99={p99}), likely a blank/degenerate image."
            )
        clipped = np.clip(arr, p1, p99)
        return (clipped - p1) / (p99 - p1)

    raise ValueError(f"Unknown normalization method: {method!r}. Expected one of 'minmax', 'zscore', 'percentile'.")


def validate_mri_array(array: np.ndarray) -> ImageQuality:
    """Sanity-check a loaded MRI array and return a heuristic ImageQuality verdict.

    NOTE: This score is a simple rule-based heuristic (dimension/NaN/dynamic
    range checks only) -- it is NOT a clinically validated image-quality
    metric and must not be presented to a clinician as diagnostic.

    Args:
        array: The loaded volume/slice to validate.

    Returns:
        An ``ImageQuality`` instance summarizing the checks.
    """
    limitations: list[str] = []
    score = 1.0

    if array is None or array.size == 0:
        return ImageQuality(quality="non-diagnostic", score=0.0, limitations=["Array is empty."])

    if array.ndim not in (2, 3, 4):
        limitations.append(f"Unexpected array dimensionality: {array.ndim}D.")
        score -= 0.4

    nan_count = int(np.isnan(array).sum())
    inf_count = int(np.isinf(array).sum())
    if nan_count > 0:
        limitations.append(f"Array contains {nan_count} NaN voxel(s).")
        score -= 0.3
    if inf_count > 0:
        limitations.append(f"Array contains {inf_count} infinite voxel(s).")
        score -= 0.3

    finite = array[np.isfinite(array)]
    if finite.size == 0:
        limitations.append("Array has no finite voxel values.")
        score = 0.0
    else:
        intensity_range = float(np.max(finite) - np.min(finite))
        if intensity_range < 1e-6:
            limitations.append("Degenerate (near-constant) intensity range; image may be blank.")
            score -= 0.4

    score = max(0.0, min(1.0, score))
    if score >= 0.75:
        quality: Literal["acceptable", "limited", "non-diagnostic", "unknown"] = "acceptable"
    elif score >= 0.4:
        quality = "limited"
    elif score > 0.0:
        quality = "non-diagnostic"
    else:
        quality = "non-diagnostic"

    return ImageQuality(quality=quality, score=round(score, 3), limitations=limitations)


def detect_orientation(affine: np.ndarray) -> str:
    """Best-effort anatomical orientation code (e.g. "RAS") from a NIfTI affine.

    Args:
        affine: A 4x4 affine matrix.

    Returns:
        A string like ``"RAS"``/``"LPI"`` from ``nibabel.aff2axcodes``, or
        ``"Unknown"`` if nibabel is unavailable or the affine is invalid.
    """
    if nib is None:
        return "Unknown"
    try:
        affine = np.asarray(affine)
        if affine.shape != (4, 4):
            return "Unknown"
        codes = nib.aff2axcodes(affine)
        return "".join(codes)
    except Exception:
        return "Unknown"


def select_representative_slices(
    volume: np.ndarray, num_slices: int = 5, axis: int = 2
) -> list[int]:
    """Select informative slice indices along an axis using an intensity-variance/entropy score.

    Rather than sampling uniformly or randomly, each candidate slice is
    scored by a combination of intensity variance (structural contrast) and
    Shannon entropy of its intensity histogram (information content) --
    slices that are mostly background/air score low on both and are
    deprioritized.

    Args:
        volume: A 3D array.
        num_slices: Number of slice indices to return (clipped to the
            number of slices available along ``axis``).
        axis: Which axis to slice along -- 0 (sagittal-ish), 1
            (coronal-ish), or 2 (axial-ish), matching the array's own axis
            ordering (not a fixed radiological convention).

    Returns:
        A sorted list of selected slice indices.

    Raises:
        ValueError: If ``volume`` is not 3D or ``axis`` is out of range.
    """
    if volume.ndim != 3:
        raise ValueError(f"select_representative_slices expects a 3D volume, got {volume.ndim}D.")
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0, 1, or 2, got {axis}.")

    n = volume.shape[axis]
    if n == 0:
        raise ValueError(f"Volume has zero slices along axis {axis}.")

    k = max(1, min(num_slices, n))

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        sl = np.take(volume, indices=i, axis=axis)
        finite = sl[np.isfinite(sl)]
        if finite.size == 0:
            scores[i] = -np.inf
            continue
        variance = float(np.var(finite))
        hist, _ = np.histogram(finite, bins=32, density=True)
        hist = hist[hist > 0]
        entropy = float(-np.sum(hist * np.log2(hist))) if hist.size > 0 else 0.0
        # Normalize variance's scale roughly to entropy's via log1p so neither
        # metric dominates purely due to units.
        scores[i] = np.log1p(variance) + entropy

    ranked = np.argsort(scores)[::-1]
    top_k = ranked[:k]
    return sorted(int(i) for i in top_k)


def _require_cv2() -> None:
    if cv2 is None:  # pragma: no cover - environment guard
        raise ImportError(
            "opencv-python (cv2) is required for assess_image_quality but is not installed. "
            "Install it with `pip install opencv-python`."
        ) from _CV2_IMPORT_ERROR


def assess_image_quality(array: np.ndarray) -> ImageQuality:
    """Heuristically assess blur, noise, clipping, and resolution of a 2D MRI slice.

    Uses the variance of the Laplacian as a blur proxy (low variance ->
    blurry/out-of-focus), a simple high-frequency noise estimate, the
    fraction of saturated/clipped voxels, and a minimum-resolution check.
    These are classical, unvalidated heuristics -- useful for filtering out
    obviously corrupt/degenerate inputs, not for clinical quality grading.

    Args:
        array: A 2D slice (if a 3D volume is passed, its middle slice along
            the last axis is used).

    Returns:
        An ``ImageQuality`` instance.

    Raises:
        ImportError: If OpenCV is not installed.
        ValueError: If the array is empty.
    """
    _require_cv2()

    if array is None or array.size == 0:
        raise ValueError("Cannot assess image quality of an empty array.")

    if array.ndim == 3:
        mid = array.shape[-1] // 2
        slice_2d = array[..., mid]
    elif array.ndim == 2:
        slice_2d = array
    else:
        raise ValueError(f"assess_image_quality expects a 2D or 3D array, got {array.ndim}D.")

    slice_2d = np.nan_to_num(slice_2d.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    limitations: list[str] = []
    score = 1.0

    MIN_RESOLUTION = 32
    if min(slice_2d.shape) < MIN_RESOLUTION:
        limitations.append(
            f"Resolution {slice_2d.shape} is below the minimum usable threshold "
            f"({MIN_RESOLUTION}px on the shortest side)."
        )
        score -= 0.3

    # Normalize to 0-255 uint8 for OpenCV blur/noise operators.
    lo, hi = float(slice_2d.min()), float(slice_2d.max())
    if hi - lo < 1e-8:
        limitations.append("Slice has (near) zero intensity range; likely blank.")
        score = 0.0
        return ImageQuality(quality="non-diagnostic", score=0.0, limitations=limitations)

    normalized = ((slice_2d - lo) / (hi - lo) * 255.0).astype(np.uint8)

    laplacian_var = float(cv2.Laplacian(normalized, cv2.CV_64F).var())
    BLUR_THRESHOLD = 50.0
    if laplacian_var < BLUR_THRESHOLD:
        limitations.append(
            f"Low Laplacian variance ({laplacian_var:.1f} < {BLUR_THRESHOLD}) suggests blur or low detail."
        )
        score -= 0.25

    # Simple noise estimate: high-frequency residual after Gaussian smoothing.
    smoothed = cv2.GaussianBlur(normalized, (5, 5), 0)
    noise_estimate = float(np.std(normalized.astype(np.float32) - smoothed.astype(np.float32)))
    NOISE_THRESHOLD = 25.0
    if noise_estimate > NOISE_THRESHOLD:
        limitations.append(
            f"High estimated noise level ({noise_estimate:.1f} > {NOISE_THRESHOLD})."
        )
        score -= 0.2

    # Saturation / clipping: fraction of voxels at the extremes of the range.
    clipped_fraction = float(
        np.mean((normalized <= 1) | (normalized >= 254))
    )
    CLIP_THRESHOLD = 0.15
    if clipped_fraction > CLIP_THRESHOLD:
        limitations.append(
            f"{clipped_fraction * 100:.1f}% of voxels are at extreme intensity values "
            "(possible saturation/clipping artifact)."
        )
        score -= 0.2

    score = max(0.0, min(1.0, score))
    if score >= 0.75:
        quality: Literal["acceptable", "limited", "non-diagnostic", "unknown"] = "acceptable"
    elif score >= 0.4:
        quality = "limited"
    else:
        quality = "non-diagnostic"

    return ImageQuality(quality=quality, score=round(score, 3), limitations=limitations)
