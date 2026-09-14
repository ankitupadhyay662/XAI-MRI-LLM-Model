"""Preprocessing package: DICOM/NIfTI loading, anonymization, and MRI transforms."""
from __future__ import annotations

from .dicom import (
    DicomLoadError,
    anonymize_dicom_metadata,
    get_safe_metadata,
    load_dicom_series,
)
from .nifti import (
    NiftiLoadError,
    detect_dimensionality,
    load_nifti,
    split_4d_volume,
)
from .transforms import (
    assess_image_quality,
    build_mri_transform_pipeline,
    detect_orientation,
    normalize_intensity,
    select_representative_slices,
    validate_mri_array,
)

__all__ = [
    "DicomLoadError",
    "load_dicom_series",
    "anonymize_dicom_metadata",
    "get_safe_metadata",
    "NiftiLoadError",
    "load_nifti",
    "detect_dimensionality",
    "split_4d_volume",
    "build_mri_transform_pipeline",
    "normalize_intensity",
    "validate_mri_array",
    "detect_orientation",
    "select_representative_slices",
    "assess_image_quality",
]
