"""NIfTI (.nii / .nii.gz) loading utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np

try:
    import nibabel as nib
except ImportError as exc:  # pragma: no cover - environment guard
    nib = None
    _NIBABEL_IMPORT_ERROR = exc
else:
    _NIBABEL_IMPORT_ERROR = None


class NiftiLoadError(Exception):
    """Raised when a NIfTI file cannot be read or is corrupted."""


def _require_nibabel() -> None:
    if nib is None:  # pragma: no cover - environment guard
        raise ImportError(
            "nibabel is required for NIfTI I/O but is not installed. "
            "Install it with `pip install nibabel`."
        ) from _NIBABEL_IMPORT_ERROR


def load_nifti(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a NIfTI volume and return its array plus geometric metadata.

    Args:
        path: Path to a ``.nii`` or ``.nii.gz`` file.

    Returns:
        ``(array, metadata)`` where ``metadata`` contains ``affine``
        (4x4 list), ``zooms`` (voxel spacing per axis from the header),
        ``shape``, ``dtype``, and ``dimensionality`` (``"2D"``/``"3D"``/``"4D"``).

    Raises:
        NiftiLoadError: If the path does not exist or the file is corrupted
            / cannot be parsed as NIfTI.
    """
    _require_nibabel()
    path = Path(path)

    if not path.exists():
        raise NiftiLoadError(f"NIfTI file does not exist: {path}")

    try:
        img = nib.load(str(path))
    except Exception as exc:
        raise NiftiLoadError(f"Could not parse NIfTI file (corrupted or invalid header): {path}") from exc

    try:
        array = np.asarray(img.get_fdata(), dtype=np.float32)
    except Exception as exc:
        raise NiftiLoadError(f"Could not load pixel/voxel data from NIfTI file: {path}") from exc

    if array.size == 0:
        raise NiftiLoadError(f"NIfTI file contains an empty volume: {path}")

    affine = np.asarray(img.affine)
    try:
        zooms = tuple(float(z) for z in img.header.get_zooms())
    except Exception:
        zooms = tuple(1.0 for _ in array.shape)

    metadata = {
        "affine": affine.tolist(),
        "zooms": zooms,
        "shape": array.shape,
        "dtype": str(array.dtype),
        "dimensionality": detect_dimensionality(array),
    }
    return array, metadata


def detect_dimensionality(array: np.ndarray) -> Literal["2D", "3D", "4D"]:
    """Classify a loaded volume by its number of non-trivial spatial/temporal axes."""
    ndim = array.ndim
    # Squeeze out trailing singleton axes some NIfTI writers add (e.g. shape (X,Y,1,1)).
    effective_ndim = len([d for d in array.shape if d > 1]) or ndim
    if effective_ndim <= 2:
        return "2D"
    if effective_ndim == 3:
        return "3D"
    return "4D"


def split_4d_volume(array: np.ndarray) -> tuple[list[np.ndarray], dict]:
    """Split a 4D multi-sequence/multi-timepoint NIfTI volume into 3D volumes.

    Args:
        array: A 4D array shaped ``(X, Y, Z, T)`` as produced by ``load_nifti``.

    Returns:
        ``(volumes, index)`` where ``volumes`` is a list of 3D arrays (one
        per index along the last axis) and ``index`` maps ``"count"`` to the
        number of volumes extracted.

    Raises:
        NiftiLoadError: If the input array is not 4D.
    """
    if array.ndim != 4:
        raise NiftiLoadError(
            f"split_4d_volume expects a 4D array, got array with shape {array.shape}"
        )
    num_volumes = array.shape[-1]
    volumes = [array[..., i] for i in range(num_volumes)]
    return volumes, {"count": num_volumes}
