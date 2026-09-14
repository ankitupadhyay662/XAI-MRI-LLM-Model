"""DICOM loading and de-identification utilities.

Reads single DICOM files or whole series directories via ``pydicom`` and
exposes a strict "safe metadata" surface that never leaks patient-identifying
fields. Callers throughout the rest of the pipeline should prefer
``get_safe_metadata`` over touching raw pydicom Datasets directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import pydicom
    from pydicom.dataset import FileDataset
    from pydicom.errors import InvalidDicomError
except ImportError as exc:  # pragma: no cover - environment guard
    pydicom = None
    FileDataset = Any  # type: ignore[assignment,misc]
    InvalidDicomError = Exception
    _PYDICOM_IMPORT_ERROR = exc
else:
    _PYDICOM_IMPORT_ERROR = None


class DicomLoadError(Exception):
    """Raised when a DICOM file or series directory cannot be read."""


# Fields that can identify a patient (or the referring institution/staff) and
# must never be surfaced by this pipeline, even in research/educational mode.
_IDENTIFYING_FIELDS = (
    "PatientName",
    "PatientID",
    "PatientBirthDate",
    "PatientAddress",
    "PatientTelephoneNumbers",
    "PatientSex",
    "PatientAge",
    "OtherPatientIDs",
    "OtherPatientNames",
    "AccessionNumber",
    "InstitutionName",
    "InstitutionAddress",
    "ReferringPhysicianName",
    "PerformingPhysicianName",
    "OperatorsName",
    "RequestingPhysician",
    "StationName",
    "StudyID",
    "IssuerOfPatientID",
)

# Technical/acquisition fields that are safe to keep for research purposes.
_SAFE_TECHNICAL_FIELDS = (
    "Modality",
    "Manufacturer",
    "ManufacturerModelName",
    "SliceThickness",
    "PixelSpacing",
    "Rows",
    "Columns",
    "MagneticFieldStrength",
    "EchoTime",
    "RepetitionTime",
    "SequenceName",
    "ScanningSequence",
    "SequenceVariant",
    "FlipAngle",
    "SpacingBetweenSlices",
    "ImageOrientationPatient",
    "PhotometricInterpretation",
    "BitsAllocated",
)


def _require_pydicom() -> None:
    if pydicom is None:  # pragma: no cover - environment guard
        raise ImportError(
            "pydicom is required for DICOM I/O but is not installed. "
            "Install it with `pip install pydicom`."
        ) from _PYDICOM_IMPORT_ERROR


def _read_single_file(file_path: Path):
    _require_pydicom()
    try:
        return pydicom.dcmread(str(file_path))
    except InvalidDicomError as exc:
        raise DicomLoadError(f"File is not a valid DICOM file: {file_path}") from exc
    except (OSError, IOError) as exc:
        raise DicomLoadError(f"Could not read DICOM file (I/O error): {file_path}") from exc


def load_dicom_series(path: str | Path) -> tuple[np.ndarray, dict]:
    """Load a single DICOM file or a directory of DICOM slices.

    Args:
        path: Path to a single ``.dcm`` file, or a directory containing one
            or more DICOM slices belonging to the same series.

    Returns:
        A tuple ``(volume, metadata)`` where ``volume`` is a numpy array
        (2D for a single slice, 3D ``(num_slices, rows, cols)`` for a
        series) and ``metadata`` is the anonymized technical metadata dict
        from the representative (first) slice, plus ``num_slices``.

    Raises:
        DicomLoadError: If the path does not exist, contains no valid DICOM
            data, or the files cannot be assembled into a consistent volume.
    """
    _require_pydicom()
    path = Path(path)

    if not path.exists():
        raise DicomLoadError(f"DICOM path does not exist: {path}")

    if path.is_file():
        ds = _read_single_file(path)
        if not hasattr(ds, "pixel_array"):
            raise DicomLoadError(f"DICOM file has no pixel data: {path}")
        try:
            array = ds.pixel_array.astype(np.float32)
        except Exception as exc:
            raise DicomLoadError(f"Could not decode pixel data for: {path}") from exc
        metadata = anonymize_dicom_metadata(ds)
        metadata["num_slices"] = 1
        return array, metadata

    if path.is_dir():
        candidate_files = sorted(p for p in path.rglob("*") if p.is_file())
        if not candidate_files:
            raise DicomLoadError(f"DICOM directory contains no files: {path}")

        datasets = []
        for f in candidate_files:
            try:
                ds = _read_single_file(f)
            except DicomLoadError:
                continue  # skip non-DICOM files (e.g. DICOMDIR, .txt) silently
            if hasattr(ds, "pixel_array"):
                datasets.append(ds)

        if not datasets:
            raise DicomLoadError(f"DICOM directory contains no valid slices: {path}")

        # Sort slices into anatomical order when InstanceNumber is available,
        # falling back to filename order otherwise.
        def _sort_key(ds) -> float:
            return float(getattr(ds, "InstanceNumber", 0) or 0)

        try:
            datasets.sort(key=_sort_key)
        except (TypeError, ValueError):
            pass  # keep filename order

        try:
            slices = [ds.pixel_array.astype(np.float32) for ds in datasets]
            volume = np.stack(slices, axis=0)
        except ValueError as exc:
            raise DicomLoadError(
                f"DICOM slices in {path} have inconsistent shapes and cannot "
                "be stacked into a single volume."
            ) from exc

        metadata = anonymize_dicom_metadata(datasets[0])
        metadata["num_slices"] = len(datasets)
        return volume, metadata

    raise DicomLoadError(f"DICOM path is neither a file nor a directory: {path}")


def anonymize_dicom_metadata(ds: "FileDataset") -> dict:
    """Return only non-identifying technical metadata from a pydicom Dataset.

    Explicitly drops PatientName, PatientID, PatientBirthDate,
    PatientAddress, PatientTelephoneNumbers, AccessionNumber,
    InstitutionName, ReferringPhysicianName, and similar identifying tags,
    regardless of whether the caller asks for them.
    """
    safe: dict = {}
    for field_name in _SAFE_TECHNICAL_FIELDS:
        if hasattr(ds, field_name):
            value = getattr(ds, field_name)
            # pydicom multi-value elements (e.g. PixelSpacing) -> plain list
            try:
                if hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
                    value = [float(v) if _is_numeric(v) else str(v) for v in value]
                elif _is_numeric(value):
                    value = float(value)
                else:
                    value = str(value)
            except Exception:
                value = str(value)
            safe[field_name] = value

    # Defensive check: guarantee no identifying field ever leaks through,
    # even if a future edit accidentally widens _SAFE_TECHNICAL_FIELDS.
    for identifying_field in _IDENTIFYING_FIELDS:
        safe.pop(identifying_field, None)

    return safe


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def get_safe_metadata(path: str | Path) -> dict:
    """Load a DICOM file/series and return only anonymized technical metadata.

    Convenience wrapper intended to be the default entry point used
    everywhere else in the pipeline -- it never exposes patient name, DOB,
    address, hospital ID, phone number, or accession number.
    """
    _, metadata = load_dicom_series(path)
    return metadata
