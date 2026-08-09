"""The only module allowed to touch raw DICOM (CLAUDE.md §3.3).

Spec 01 needs header-level metadata for the series census and site fingerprints, so this
module starts with the metadata half. Spec 03 adds `read_series` (pixels, ordering, rescale)
to this same file; the inference path imports from here too.

Only 86 metadata tags are allowlisted in this competition and none are guaranteed, so every
tag access goes through `_get` — never `ds.Tag` attribute access.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydicom

# Tags read for the Spec 01 census + site fingerprint. All optional.
FINGERPRINT_TAGS = (
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "ImplementationVersionName",
    "SoftwareVersions",
    "InstitutionName",
)


class SeriesUnreadable(RuntimeError):
    """Raised when too large a fraction of a series fails to decode (Spec 03)."""


def _get(ds: pydicom.Dataset, name: str, default: Any = None) -> Any:
    """Guarded tag access; also unwraps MultiValue and pydicom's numeric proxies."""
    value = ds.get(name, default)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    try:
        if isinstance(value, pydicom.multival.MultiValue):
            return [float(v) if _is_numeric(v) else str(v) for v in value]
        return float(value) if _is_numeric(value) else str(value)
    except (TypeError, ValueError):
        return str(value)


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


@dataclass
class SeriesMeta:
    """Header-level facts about one series. Every field may be None."""

    study_uid: str
    series_uid: str
    n_files: int
    n_read: int
    rows: int | None
    cols: int | None
    pixel_spacing: list[float] | None
    slice_thickness: float | None
    transfer_syntax: str | None
    mixed_shapes: bool
    patient_id: str | None
    fingerprint: dict[str, Any]
    error: str | None = None

    def to_row(self) -> dict[str, Any]:
        ps = self.pixel_spacing or [None, None]
        return {
            "StudyInstanceUID": self.study_uid,
            "SeriesInstanceUID": self.series_uid,
            "n_files": self.n_files,
            "n_read": self.n_read,
            "rows": self.rows,
            "cols": self.cols,
            "pixel_spacing_y": ps[0],
            "pixel_spacing_x": ps[1] if len(ps) > 1 else None,
            "slice_thickness": self.slice_thickness,
            "transfer_syntax": self.transfer_syntax,
            "mixed_shapes": self.mixed_shapes,
            "PatientID": self.patient_id,
            **{k: self.fingerprint.get(k) for k in FINGERPRINT_TAGS},
            "error": self.error,
        }


def list_series_files(series_dir: Path) -> list[Path]:
    """Sorted .dcm paths in a series directory (filename order — NOT slice order)."""
    return sorted(p for p in Path(series_dir).iterdir() if p.suffix.lower() == ".dcm")


def read_series_meta(series_dir: str | Path, probe_slices: int = 3) -> SeriesMeta:
    """Read headers only (no pixel decode) from a few slices of a series.

    `probe_slices` files are sampled evenly across the series to detect mixed in-plane shapes
    without paying for a full directory read.
    """
    series_dir = Path(series_dir)
    files = list_series_files(series_dir)
    base = SeriesMeta(
        study_uid=series_dir.parent.name,
        series_uid=series_dir.name,
        n_files=len(files),
        n_read=0,
        rows=None,
        cols=None,
        pixel_spacing=None,
        slice_thickness=None,
        transfer_syntax=None,
        mixed_shapes=False,
        patient_id=None,
        fingerprint={},
    )
    if not files:
        base.error = "no .dcm files"
        return base

    idx = sorted({0, len(files) // 2, len(files) - 1})[:probe_slices]
    shapes: set[tuple[int | None, int | None]] = set()
    for i in idx:
        try:
            # force=True tolerates files missing the DICM preamble, which do occur in the
            # wild; the emptiness check below catches genuinely unparseable bytes.
            ds = pydicom.dcmread(files[i], stop_before_pixels=True, force=True)
        except Exception as exc:  # a single unreadable header must not kill the census
            base.error = f"{type(exc).__name__}: {exc}"
            continue
        rows, cols = _get(ds, "Rows"), _get(ds, "Columns")
        if rows is None or cols is None:
            # force=True happily invents elements from arbitrary bytes, so "parsed without
            # raising" is not enough: a usable slice must declare its in-plane size.
            base.error = f"unparseable header (no Rows/Columns): {files[i].name}"
            continue
        base.n_read += 1
        shapes.add((rows, cols))
        if base.rows is None:
            base.rows, base.cols = rows, cols
            base.pixel_spacing = _get(ds, "PixelSpacing")
            base.slice_thickness = _get(ds, "SliceThickness")
            base.patient_id = _get(ds, "PatientID")
            base.fingerprint = {t: _get(ds, t) for t in FINGERPRINT_TAGS}
            # ImplementationVersionName and the transfer syntax live in the file meta group,
            # not the main dataset.
            file_meta = getattr(ds, "file_meta", None)
            if file_meta is not None:
                ts = file_meta.get("TransferSyntaxUID")
                base.transfer_syntax = str(ts) if ts is not None else None
                if base.fingerprint.get("ImplementationVersionName") is None:
                    impl = file_meta.get("ImplementationVersionName")
                    base.fingerprint["ImplementationVersionName"] = str(impl) if impl else None

    base.mixed_shapes = len(shapes) > 1
    return base
