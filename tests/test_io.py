"""Contract tests for src/utils/io.py.

The RSNA_RAW discovery tests exist because of a real failure: on Kaggle the data sits under
`/kaggle/input/competitions/<slug>`, and pointing RSNA_RAW at the parent produced a bare
pandas FileNotFoundError that said nothing about how to fix it.
"""

from __future__ import annotations

import pytest

from src.utils.constants import LABELS
from src.utils.io import REPO_ROOT, RawDataNotFound, find_data_dirs, load_raw, resolve


def test_resolve_anchors_relative_paths_at_the_repo_root():
    assert resolve("artifacts/cache") == REPO_ROOT / "artifacts" / "cache"


def test_resolve_leaves_absolute_paths_alone(tmp_path):
    assert resolve(tmp_path / "x") == tmp_path / "x"


def test_load_raw_reads_both_csvs(tmp_path, synthetic):
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    train, series_df = load_raw(raw)
    assert len(train) == len(synthetic.train)
    assert set(LABELS) <= set(train.columns)
    assert "SeriesInstanceUID" in series_df.columns


def test_load_raw_points_at_the_right_directory_when_set_one_level_too_high(tmp_path, synthetic):
    """The Kaggle `/kaggle/input/competitions` case."""
    actual = tmp_path / "competitions" / "rsna-knee-2026"
    synthetic.write_raw(actual)

    with pytest.raises(RawDataNotFound) as excinfo:
        load_raw(tmp_path / "competitions")

    message = str(excinfo.value)
    assert "did you mean" in message
    assert str(actual) in message
    assert "train.csv" in message


def test_load_raw_says_so_when_nothing_is_there(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(RawDataNotFound, match="nothing under"):
        load_raw(empty)


def test_load_raw_on_a_nonexistent_directory(tmp_path):
    with pytest.raises(RawDataNotFound):
        load_raw(tmp_path / "does-not-exist")


def test_load_raw_rejects_a_csv_missing_label_columns(tmp_path, synthetic):
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    synthetic.train.drop(columns=["ACL"]).to_csv(raw / "train.csv", index=False)

    with pytest.raises(ValueError, match="missing label columns"):
        load_raw(raw)


def test_find_data_dirs_finds_nested_and_direct_layouts(tmp_path, synthetic):
    synthetic.write_raw(tmp_path / "a" / "b")
    synthetic.write_raw(tmp_path / "direct")

    assert find_data_dirs(tmp_path) == sorted([tmp_path / "a" / "b", tmp_path / "direct"])
    assert find_data_dirs(tmp_path / "direct") == [tmp_path / "direct"]
    assert find_data_dirs(tmp_path / "nope") == []


def test_find_data_dirs_respects_max_depth(tmp_path, synthetic):
    synthetic.write_raw(tmp_path / "a" / "b" / "c" / "d")
    assert find_data_dirs(tmp_path, max_depth=2) == []
    assert find_data_dirs(tmp_path, max_depth=4) == [tmp_path / "a" / "b" / "c" / "d"]
