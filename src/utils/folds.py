"""Frozen CV assignment (Spec 01, Task 1.3).

Two-stage assignment over *groups* (patients when available, else studies):

  stage 1  labeled groups   -> MultilabelStratifiedKFold on the 12 labels
  stage 2  unlabeled groups -> greedy balancing on a proxy vector
                               (PatientSex, report language, site_cluster, series-count bucket)

Groups never span folds, so a patient imaged twice cannot leak across the split.
Generated ONCE; `assert_not_frozen` refuses to overwrite an existing file (CLAUDE.md §3.2).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.constants import LABELS
from src.utils.io import file_hash

UNKNOWN = "unknown"
PROXY_COLUMNS = ["sex", "language", "site_cluster", "series_bucket"]
OUTPUT_SCHEMA = {
    "StudyInstanceUID": "object",
    "fold": "int8",
    "site_cluster": "int16",
    "has_gt_labels": "bool",
}


class FoldsFrozenError(RuntimeError):
    """Raised when folds.parquet already exists."""


@dataclass
class FoldReport:
    """Everything the CLI prints and LOG.md records about one fold build."""

    group_key: str
    n_studies: int
    n_groups: int
    n_labeled_groups: int
    n_unlabeled_groups: int
    proxy_available: dict[str, bool] = field(default_factory=dict)
    fold_sizes: dict[int, int] = field(default_factory=dict)

    def to_lines(self) -> list[str]:
        proxy = ", ".join(f"{k}={'yes' if v else 'MISSING'}" for k, v in self.proxy_available.items())
        return [
            f"group_key            : {self.group_key}",
            f"studies / groups     : {self.n_studies} / {self.n_groups}",
            f"labeled groups       : {self.n_labeled_groups}",
            f"unlabeled groups     : {self.n_unlabeled_groups}",
            f"proxy vector         : {proxy}",
            f"fold sizes (studies) : {dict(sorted(self.fold_sizes.items()))}",
        ]


def assert_not_frozen(path: str | Path) -> None:
    """Refuse to regenerate an existing folds file (CLAUDE.md §3.2)."""
    p = Path(path)
    if p.exists():
        raise FoldsFrozenError(
            f"folds are frozen; delete manually to regenerate\n"
            f"  path: {p}\n  sha256: {file_hash(p)}"
        )


def series_bucket(n_series: pd.Series, edges: list[int]) -> pd.Series:
    """Bucket the per-study series count. `edges` are inclusive right edges of the low buckets."""
    bins = [-np.inf, *edges, np.inf]
    return pd.cut(n_series, bins=bins, labels=False, right=True).astype("int16")


def build_study_frame(
    train_df: pd.DataFrame,
    series_df: pd.DataFrame,
    study_meta: pd.DataFrame | None = None,
    *,
    group_by: str = "auto",
    min_patient_id_coverage: float = 0.95,
    series_count_buckets: list[int] | None = None,
) -> tuple[pd.DataFrame, FoldReport]:
    """Assemble one row per study: group id, GT flag, and the proxy-vector columns.

    `study_meta` is the optional per-study table written by scripts/run_eda.py
    (StudyInstanceUID + any of PatientID / site_cluster / language). Missing columns
    degrade the proxy vector to `unknown` rather than failing — the report records which.
    """
    series_count_buckets = series_count_buckets or [1, 3, 5, 8]

    df = train_df[["StudyInstanceUID"]].drop_duplicates().copy()
    label_cols = [c for c in LABELS if c in train_df.columns]
    if len(label_cols) != len(LABELS):
        missing = sorted(set(LABELS) - set(label_cols))
        raise ValueError(f"train_df is missing label columns: {missing}")

    # A study is "labeled" only if the whole 12-vector is present; partial rows are treated as
    # unlabeled for stratification but keep whatever GT they have downstream.
    is_labeled = train_df.set_index("StudyInstanceUID")[LABELS].notna().all(axis=1)
    df["has_gt_labels"] = (
        is_labeled.reindex(df["StudyInstanceUID"]).fillna(False).to_numpy(dtype=bool)
    )

    counts = series_df.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    df["n_series"] = df["StudyInstanceUID"].map(counts).fillna(0).astype("int32")
    df["series_bucket"] = series_bucket(df["n_series"], series_count_buckets)

    if "PatientSex" in train_df.columns:
        sex = train_df.set_index("StudyInstanceUID")["PatientSex"]
        df["sex"] = df["StudyInstanceUID"].map(sex).fillna(UNKNOWN).astype(str)
        sex_available = df["sex"].ne(UNKNOWN).any()
    else:
        df["sex"] = UNKNOWN
        sex_available = False

    meta = study_meta.set_index("StudyInstanceUID") if study_meta is not None else None

    def from_meta(col: str, default):
        if meta is None or col not in meta.columns:
            return pd.Series(default, index=df.index), False
        mapped = df["StudyInstanceUID"].map(meta[col])
        return mapped, mapped.notna().any()

    lang, lang_available = from_meta("language", UNKNOWN)
    df["language"] = lang.fillna(UNKNOWN).astype(str)

    site, site_available = from_meta("site_cluster", -1)
    df["site_cluster"] = pd.to_numeric(site, errors="coerce").fillna(-1).astype("int16")

    pid, pid_available = from_meta("PatientID", None)
    pid_coverage = float(pid.notna().mean()) if pid_available else 0.0

    use_patient = (group_by == "patient") or (
        group_by == "auto" and pid_coverage >= min_patient_id_coverage
    )
    if group_by == "patient" and not pid_available:
        raise ValueError("folds.group_by='patient' but study_meta has no usable PatientID column")

    if use_patient:
        group_key = "PatientID"
        df["group_id"] = pid.astype(str)
    else:
        group_key = "StudyInstanceUID"
        df["group_id"] = df["StudyInstanceUID"].astype(str)

    report = FoldReport(
        group_key=group_key,
        n_studies=len(df),
        n_groups=df["group_id"].nunique(),
        n_labeled_groups=0,
        n_unlabeled_groups=0,
        proxy_available={
            "sex": bool(sex_available),
            "language": bool(lang_available),
            "site_cluster": bool(site_available),
            "series_bucket": True,
            "PatientID": bool(pid_available) and use_patient,
        },
    )
    return df.reset_index(drop=True), report


def _group_label_matrix(study_frame: pd.DataFrame, train_df: pd.DataFrame) -> tuple[list[str], np.ndarray]:
    """Per labeled group: OR of the label vectors of its labeled studies."""
    labeled = study_frame.loc[study_frame["has_gt_labels"], ["StudyInstanceUID", "group_id"]]
    y = (
        train_df.set_index("StudyInstanceUID")
        .loc[labeled["StudyInstanceUID"], LABELS]
        .fillna(0.0)
        .to_numpy()
        > 0.5
    )
    frame = pd.DataFrame(y.astype("int8"), columns=LABELS)
    frame["group_id"] = labeled["group_id"].to_numpy()
    agg = frame.groupby("group_id", sort=True)[LABELS].max()
    return agg.index.tolist(), agg.to_numpy().astype("int8")


def _stratify_labeled(
    groups: list[str], y: np.ndarray, n_folds: int, seed: int
) -> dict[str, int]:
    """Stage 1: MultilabelStratifiedKFold over labeled groups."""
    if len(groups) < n_folds:
        # Degenerate (tiny fixtures / smoke runs): deterministic round-robin.
        return {g: i % n_folds for i, g in enumerate(groups)}

    from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

    mskf = MultilabelStratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    assignment: dict[str, int] = {}
    x = np.zeros((len(groups), 1), dtype="int8")
    for fold, (_, val_idx) in enumerate(mskf.split(x, y)):
        for i in val_idx:
            assignment[groups[i]] = fold
    return assignment


def _balance_unlabeled(
    study_frame: pd.DataFrame,
    labeled_assignment: dict[str, int],
    n_folds: int,
) -> dict[str, int]:
    """Stage 2: greedy assignment of unlabeled groups balancing the proxy vector.

    Deterministic: strata and groups are visited in sorted order, and ties break on
    (stratum count, global count, fold index). No RNG.
    """
    frame = study_frame.copy()
    frame["_stratum"] = list(
        zip(*(frame[c].astype(str) for c in PROXY_COLUMNS), strict=True)
    )

    global_counts = np.zeros(n_folds, dtype="int64")
    stratum_counts: dict[tuple, np.ndarray] = defaultdict(lambda: np.zeros(n_folds, dtype="int64"))

    # Seed the counters with stage-1 placements so the two stages balance jointly.
    assigned = frame[frame["group_id"].isin(labeled_assignment)]
    for stratum, group_id in zip(assigned["_stratum"], assigned["group_id"], strict=True):
        fold = labeled_assignment[group_id]
        global_counts[fold] += 1
        stratum_counts[stratum][fold] += 1

    # A group's stratum is its modal study stratum; its size is its study count.
    pending = frame[~frame["group_id"].isin(labeled_assignment)]
    by_group = (
        pending.groupby("group_id", sort=True)
        .agg(size=("StudyInstanceUID", "size"), stratum=("_stratum", lambda s: s.mode().iat[0]))
        .reset_index()
    )
    # Largest groups first: placing them early avoids a lumpy tail.
    by_group = by_group.sort_values(["size", "group_id"], ascending=[False, True], kind="stable")

    assignment: dict[str, int] = {}
    for group_id, size, stratum in by_group.itertuples(index=False):
        counts = stratum_counts[stratum]
        fold = min(range(n_folds), key=lambda f: (counts[f], global_counts[f], f))
        assignment[group_id] = fold
        counts[fold] += size
        global_counts[fold] += size
    return assignment


def assign_folds(
    study_frame: pd.DataFrame,
    train_df: pd.DataFrame,
    *,
    n_folds: int = 5,
    seed: int = 42,
    report: FoldReport | None = None,
) -> pd.DataFrame:
    """Run both stages and return the frozen-folds table."""
    groups, y = _group_label_matrix(study_frame, train_df)
    labeled_assignment = _stratify_labeled(groups, y, n_folds, seed)
    unlabeled_assignment = _balance_unlabeled(study_frame, labeled_assignment, n_folds)

    assignment = {**labeled_assignment, **unlabeled_assignment}
    missing = set(study_frame["group_id"]) - set(assignment)
    if missing:
        raise RuntimeError(f"{len(missing)} groups were never assigned a fold")

    out = pd.DataFrame(
        {
            "StudyInstanceUID": study_frame["StudyInstanceUID"].astype(str),
            "fold": study_frame["group_id"].map(assignment).astype("int8"),
            "site_cluster": study_frame["site_cluster"].astype("int16"),
            "has_gt_labels": study_frame["has_gt_labels"].astype(bool),
        }
    ).sort_values("StudyInstanceUID", kind="stable").reset_index(drop=True)

    if report is not None:
        report.n_labeled_groups = len(labeled_assignment)
        report.n_unlabeled_groups = len(unlabeled_assignment)
        report.fold_sizes = out["fold"].value_counts().sort_index().to_dict()
    return out


def fold_prevalence_table(folds: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-label positive rate per fold on the GT-labeled subset (+ a `global` row)."""
    gt_uids = folds.loc[folds["has_gt_labels"], "StudyInstanceUID"]
    y = train_df.set_index("StudyInstanceUID").loc[gt_uids, LABELS]
    y = y.join(folds.set_index("StudyInstanceUID")["fold"])
    table = y.groupby("fold")[LABELS].mean()
    table.loc["global"] = y[LABELS].mean()
    return table
