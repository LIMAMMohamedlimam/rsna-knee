"""Acceptance tests for Spec 01 Task 1.3 (the four numbered checks, plus guards)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from src.utils.constants import LABELS
from src.utils.folds import (
    FoldsFrozenError,
    assert_not_frozen,
    assign_folds,
    build_study_frame,
    fold_prevalence_table,
)
from src.utils.io import REPO_ROOT

N_FOLDS = 5
SEED = 42
MIN_POSITIVES_FOR_STRICT_CHECK = 50
RELATIVE_TOLERANCE = 0.25


@pytest.fixture(scope="module")
def folds_and_frame(synthetic):
    frame, report = build_study_frame(synthetic.train, synthetic.series, synthetic.study_meta)
    folds = assign_folds(frame, synthetic.train, n_folds=N_FOLDS, seed=SEED, report=report)
    return folds, frame, report


# --- acceptance test 1 -------------------------------------------------------------------
def test_every_study_appears_once_and_folds_are_0_to_4(folds_and_frame, synthetic):
    folds, _, _ = folds_and_frame
    assert len(folds) == len(synthetic.train)
    assert folds["StudyInstanceUID"].is_unique
    assert set(folds["StudyInstanceUID"]) == set(synthetic.train["StudyInstanceUID"])
    assert sorted(folds["fold"].unique()) == list(range(N_FOLDS))


def test_output_schema(folds_and_frame):
    folds, _, _ = folds_and_frame
    assert list(folds.columns) == ["StudyInstanceUID", "fold", "site_cluster", "has_gt_labels"]
    assert folds["fold"].dtype == "int8"
    assert folds["site_cluster"].dtype == "int16"
    assert folds["has_gt_labels"].dtype == "bool"


# --- acceptance test 2 -------------------------------------------------------------------
def test_per_label_prevalence_is_balanced_across_folds(folds_and_frame, synthetic):
    folds, _, _ = folds_and_frame
    table = fold_prevalence_table(folds, synthetic.train)
    gt = synthetic.train.set_index("StudyInstanceUID").loc[
        folds.loc[folds["has_gt_labels"], "StudyInstanceUID"], LABELS
    ]

    for label in LABELS:
        n_pos = int((gt[label] > 0.5).sum())
        global_rate = float(table.loc["global", label])
        fold_rates = table.loc[list(range(N_FOLDS)), label]

        if n_pos >= MIN_POSITIVES_FOR_STRICT_CHECK:
            lo = global_rate * (1 - RELATIVE_TOLERANCE)
            hi = global_rate * (1 + RELATIVE_TOLERANCE)
            assert fold_rates.between(lo, hi).all(), (
                f"{label}: fold rates {fold_rates.round(4).to_dict()} outside "
                f"[{lo:.4f}, {hi:.4f}] (global {global_rate:.4f}, n_pos={n_pos})"
            )
        else:  # rare label — only require presence in every fold
            assert (fold_rates > 0).all(), (
                f"rare label {label} (n_pos={n_pos}) absent from some fold: "
                f"{fold_rates.round(4).to_dict()}"
            )


# --- acceptance test 4 -------------------------------------------------------------------
def test_no_group_spans_two_folds(folds_and_frame):
    folds, frame, report = folds_and_frame
    assert report.group_key == "PatientID", "study_meta has full PatientID coverage here"

    merged = frame[["StudyInstanceUID", "group_id"]].merge(folds, on="StudyInstanceUID")
    per_group = merged.groupby("group_id")["fold"].nunique()
    assert (per_group == 1).all(), f"{(per_group > 1).sum()} groups span multiple folds"

    # The fixture must actually contain multi-study patients or this test proves nothing.
    assert (merged.groupby("group_id").size() > 1).any()


def test_grouping_falls_back_to_study_without_patient_id(synthetic):
    meta = synthetic.study_meta.drop(columns=["PatientID"])
    _, report = build_study_frame(synthetic.train, synthetic.series, meta)
    assert report.group_key == "StudyInstanceUID"
    assert report.proxy_available["PatientID"] is False


def test_unlabeled_studies_all_receive_a_fold(folds_and_frame):
    folds, _, _ = folds_and_frame
    unlabeled = folds[~folds["has_gt_labels"]]
    assert len(unlabeled) > 0
    assert sorted(unlabeled["fold"].unique()) == list(range(N_FOLDS))


def test_proxy_vector_marginals_are_balanced_across_folds(folds_and_frame):
    """Stage 2 balances the proxy vector jointly with stage 1, so the check is on the whole
    dataset — the unlabeled subset alone is deliberately lumpy where it compensates for an
    imbalance stratification left behind."""
    folds, frame, _ = folds_and_frame
    merged = frame.drop(columns=["site_cluster"]).merge(folds, on="StudyInstanceUID")

    fair_share = 1 / N_FOLDS
    for column in ["language", "sex", "site_cluster", "series_bucket"]:
        share = merged.groupby([column, "fold"]).size() / merged.groupby(column).size()
        assert share.between(0.7 * fair_share, 1.3 * fair_share).all(), (
            f"{column} is unbalanced across folds: {share.round(3).to_dict()}"
        )


def test_assignment_is_deterministic(synthetic):
    def build():
        frame, report = build_study_frame(synthetic.train, synthetic.series, synthetic.study_meta)
        return assign_folds(frame, synthetic.train, n_folds=N_FOLDS, seed=SEED, report=report)

    pd.testing.assert_frame_equal(build(), build())


# --- acceptance test 3 -------------------------------------------------------------------
def test_assert_not_frozen_raises_when_file_exists(tmp_path):
    path = tmp_path / "folds.parquet"
    assert_not_frozen(path)  # absent -> no raise

    pd.DataFrame({"a": [1]}).to_parquet(path)
    with pytest.raises(FoldsFrozenError, match="folds are frozen"):
        assert_not_frozen(path)


def test_cli_creates_then_refuses_to_regenerate(tmp_path, synthetic):
    raw_dir = tmp_path / "raw"
    synthetic.write_raw(raw_dir)
    out = tmp_path / "folds.parquet"
    meta = tmp_path / "study_meta.parquet"
    synthetic.study_meta.to_parquet(meta, index=False)

    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "make_folds.py"), "--override",
        f"paths.raw_dir={raw_dir}", f"paths.folds_path={out}", f"paths.study_meta_path={meta}",
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert first.returncode == 0, first.stderr
    assert out.exists()
    assert "FROZEN" in first.stderr

    second = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert second.returncode != 0
    assert "folds are frozen; delete manually to regenerate" in second.stderr


def test_cli_warns_when_study_meta_missing(tmp_path, synthetic):
    raw_dir = tmp_path / "raw"
    synthetic.write_raw(raw_dir)
    cmd = [
        sys.executable, str(REPO_ROOT / "scripts" / "make_folds.py"), "--override",
        f"paths.raw_dir={raw_dir}", f"paths.folds_path={tmp_path / 'f.parquet'}",
        f"paths.study_meta_path={tmp_path / 'missing.parquet'}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert result.returncode == 0
    assert "run `make eda` first" in result.stderr
    assert Path(tmp_path / "f.parquet").exists()
