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


def _as_text(value: Any) -> str | None:
    """One string per cell. Multi-valued tags join on '\\', DICOM's own separator."""
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        joined = "\\".join(str(v) for v in value if v is not None)
        return joined or None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    """First numeric value, or None. Tolerates VM>1 (takes element 0) and junk."""
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int(value: Any) -> int | None:
    number = _as_float(value)
    return None if number is None else int(number)


def _as_float_pair(value: Any) -> tuple[float | None, float | None]:
    """PixelSpacing is [row, col] — but may arrive as a single value or be absent."""
    if value is None:
        return None, None
    if not isinstance(value, (list, tuple)):
        single = _as_float(value)
        return single, single
    first = _as_float(value[0]) if len(value) > 0 else None
    second = _as_float(value[1]) if len(value) > 1 else first
    return first, second


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
        """Flatten to one parquet-safe row: every column gets a single scalar type.

        Several allowlisted tags are VM>1 in real data (SoftwareVersions especially), so a
        naive dump produces a column mixing lists and strings, which parquet rejects — and it
        only shows up part-way through a long sweep.
        """
        y, x = _as_float_pair(self.pixel_spacing)
        return {
            "StudyInstanceUID": self.study_uid,
            "SeriesInstanceUID": self.series_uid,
            "n_files": self.n_files,
            "n_read": self.n_read,
            "rows": _as_int(self.rows),
            "cols": _as_int(self.cols),
            "pixel_spacing_y": y,
            "pixel_spacing_x": x,
            "slice_thickness": _as_float(self.slice_thickness),
            "transfer_syntax": _as_text(self.transfer_syntax),
            "mixed_shapes": bool(self.mixed_shapes),
            "PatientID": _as_text(self.patient_id),
            # Fingerprint fields are always text: MagneticFieldStrength is numeric in most
            # studies but multi-valued in some, and every consumer stringifies it anyway.
            **{k: _as_text(self.fingerprint.get(k)) for k in FINGERPRINT_TAGS},
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
