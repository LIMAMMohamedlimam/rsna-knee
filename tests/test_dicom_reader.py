"""Contract tests for the metadata half of src/data/dicom_reader.py.

Spec 03 adds pixel decoding here and its own fixtures for the 4 transfer syntaxes; these
tests cover what Spec 01 depends on: guarded tag access that never raises on missing tags.
"""

from __future__ import annotations

import pydicom
import pytest

from src.data.dicom_reader import list_series_files, read_series_meta
from tests.conftest import SITES, write_synthetic_dicom


@pytest.fixture
def series_dir(tmp_path):
    directory = tmp_path / "1.2.3" / "1.2.3.1"
    for i in range(5):
        write_synthetic_dicom(directory / f"{i}.dcm", "1.2.3", "1.2.3.1", i, "PAT1", SITES[0])
    return directory


def test_reads_headers_without_decoding_pixels(series_dir):
    meta = read_series_meta(series_dir)
    assert meta.study_uid == "1.2.3"
    assert meta.series_uid == "1.2.3.1"
    assert meta.n_files == 5
    assert meta.n_read == 3          # first / middle / last only
    assert (meta.rows, meta.cols) == (256, 256)
    assert meta.patient_id == "PAT1"
    assert meta.error is None
    assert not meta.mixed_shapes


def test_extracts_the_site_fingerprint_including_file_meta_tags(series_dir):
    meta = read_series_meta(series_dir)
    assert meta.fingerprint["Manufacturer"] == SITES[0]["Manufacturer"]
    assert meta.fingerprint["MagneticFieldStrength"] == pytest.approx(SITES[0]["MagneticFieldStrength"])
    # Lives in the file meta group, not the main dataset.
    assert meta.fingerprint["ImplementationVersionName"] == "PYDICOM_TEST"
    assert meta.transfer_syntax == str(pydicom.uid.ExplicitVRLittleEndian)


def test_absent_tags_yield_none_instead_of_raising(tmp_path):
    """The 86-tag allowlist means no tag is guaranteed (CLAUDE.md §4)."""
    directory = tmp_path / "study" / "series"
    path = write_synthetic_dicom(directory / "0.dcm", "1.2.3", "1.2.3.1", 0, "P", SITES[0])
    ds = pydicom.dcmread(path)
    for tag in ["Manufacturer", "ManufacturerModelName", "MagneticFieldStrength",
                "PixelSpacing", "SliceThickness", "PatientID"]:
        del ds[tag]
    ds.save_as(path, enforce_file_format=True)

    meta = read_series_meta(directory)
    assert meta.error is None
    assert meta.patient_id is None
    assert meta.pixel_spacing is None
    assert meta.fingerprint["Manufacturer"] is None
    assert (meta.rows, meta.cols) == (256, 256)  # still reads what is present


def test_mixed_shapes_are_detected(tmp_path):
    directory = tmp_path / "study" / "series"
    write_synthetic_dicom(directory / "0.dcm", "1.2.3", "1.2.3.2", 0, "P", SITES[0], rows=256, cols=256)
    write_synthetic_dicom(directory / "1.dcm", "1.2.3", "1.2.3.2", 1, "P", SITES[0], rows=320, cols=320)
    write_synthetic_dicom(directory / "2.dcm", "1.2.3", "1.2.3.2", 2, "P", SITES[0], rows=256, cols=256)
    assert read_series_meta(directory).mixed_shapes is True


def test_empty_directory_reports_an_error_rather_than_raising(tmp_path):
    directory = tmp_path / "empty"
    directory.mkdir()
    meta = read_series_meta(directory)
    assert meta.n_files == 0
    assert meta.error == "no .dcm files"


def test_corrupt_file_is_reported_not_raised(tmp_path):
    directory = tmp_path / "study" / "series"
    directory.mkdir(parents=True)
    (directory / "0.dcm").write_bytes(b"not a dicom file at all")
    meta = read_series_meta(directory)
    assert meta.n_read == 0
    assert meta.error is not None


def test_list_series_files_ignores_non_dicom(tmp_path):
    directory = tmp_path / "s"
    directory.mkdir()
    (directory / "a.dcm").write_bytes(b"")
    (directory / "notes.txt").write_bytes(b"")
    assert [p.name for p in list_series_files(directory)] == ["a.dcm"]


# --- parquet-safety of the flattened row --------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("MR 5.2", "MR 5.2"),
        (["syngo MR E11", "VE11C"], "syngo MR E11\\VE11C"),   # VM>1: DICOM's own separator
        ([], None),
        ("", None),
        (1.5, "1.5"),
    ],
)
def test_as_text_flattens_multi_valued_tags(value, expected):
    from src.data.dicom_reader import _as_text

    assert _as_text(value) == expected


@pytest.mark.parametrize(
    ("value", "expected"),
    [(None, (None, None)), ([0.3, 0.4], (0.3, 0.4)), (0.5, (0.5, 0.5)), ([0.31], (0.31, 0.31))],
)
def test_as_float_pair_handles_every_pixel_spacing_shape(value, expected):
    from src.data.dicom_reader import _as_float_pair

    assert _as_float_pair(value) == expected


def test_as_float_tolerates_junk():
    from src.data.dicom_reader import _as_float, _as_int

    assert _as_float("not a number") is None
    assert _as_float(["3.0", "4.0"]) == 3.0
    assert _as_int(256.0) == 256
    assert _as_int(None) is None


def test_to_row_never_contains_a_list(series_dir):
    """Schema boundary: one scalar per column, or parquet fails mid-sweep."""
    row = read_series_meta(series_dir).to_row()
    offenders = {k: v for k, v in row.items() if isinstance(v, (list, tuple, dict))}
    assert offenders == {}


def test_sweep_rows_write_to_parquet_even_with_multi_valued_tags(tmp_path):
    """Regression: SoftwareVersions is VM>1 on some scanners and VM=1 on others, which made
    pyarrow reject the column part-way through a 4407-study sweep."""
    import pandas as pd
    import pydicom

    rows = []
    for i, versions in enumerate([["syngo MR E11", "VE11C"], "MR 5.2", None]):
        directory = tmp_path / f"study{i}" / "series"
        path = write_synthetic_dicom(directory / "0.dcm", "1.2.3", f"1.2.3.{i}", 0, "P", SITES[0])
        ds = pydicom.dcmread(path)
        if versions is None:
            if "SoftwareVersions" in ds:
                del ds["SoftwareVersions"]
        else:
            ds.SoftwareVersions = versions
        ds.save_as(path, enforce_file_format=True)
        rows.append(read_series_meta(directory).to_row())

    frame = pd.DataFrame(rows)
    out = tmp_path / "headers.parquet"
    frame.to_parquet(out, index=False)          # must not raise

    reloaded = pd.read_parquet(out)
    assert reloaded["SoftwareVersions"].tolist() == ["syngo MR E11\\VE11C", "MR 5.2", None]
