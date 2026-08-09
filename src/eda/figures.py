"""The two figures Spec 01 requires. Matplotlib only, Agg backend, no seaborn."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.eda.series import PROTOCOL_KEYS  # noqa: E402
from src.utils.constants import LABELS  # noqa: E402


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def label_cooccurrence_heatmap(jaccard: pd.DataFrame, out_path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    data = jaccard.reindex(index=LABELS, columns=LABELS).to_numpy(dtype=float)
    im = ax.imshow(np.nan_to_num(data), cmap="viridis", vmin=0, vmax=np.nanmax(data) or 1)

    ax.set_xticks(range(len(LABELS)), LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(LABELS)), LABELS, fontsize=8)
    for i in range(len(LABELS)):
        for j in range(len(LABELS)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if data[i, j] < 0.5 * (np.nanmax(data) or 1) else "black")
    ax.set_title("Label co-occurrence (Jaccard) — GT-labeled subset")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Jaccard")
    return _save(fig, out_path)


def series_crosstab_figure(crosstab: pd.DataFrame, out_path: Path) -> Path:
    """Horizontal bars of study coverage per (plane, fluid, fatsat) combo."""
    df = crosstab.sort_values("study_coverage")
    names = df[PROTOCOL_KEYS].astype(str).agg(" | ".join, axis=1)

    fig, ax = plt.subplots(figsize=(9, max(3.0, 0.32 * len(df))))
    ax.barh(range(len(df)), df["study_coverage"] * 100, color="#3b6ea5")
    ax.set_yticks(range(len(df)), names, fontsize=8)
    ax.set_xlabel("study coverage (% of studies with ≥1 such series)")
    ax.set_title("Protocol cross-tab: plane × fluid-sensitive × fat-suppression")
    for i, (cov, n) in enumerate(zip(df["study_coverage"], df["n_series"], strict=True)):
        ax.text(cov * 100 + 0.5, i, f"{cov:.1%} ({n} series)", va="center", fontsize=7)
    ax.set_xlim(0, 108)
    return _save(fig, out_path)


def series_per_study_hist(series_df: pd.DataFrame, out_path: Path) -> Path:
    counts = series_df.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(counts, bins=range(0, int(counts.max()) + 2), color="#3b6ea5", edgecolor="white")
    ax.set_xlabel("series per study")
    ax.set_ylabel("studies")
    ax.set_title(f"Series per study (n={counts.size}, median={counts.median():.0f})")
    return _save(fig, out_path)


def report_length_hist(reports: pd.Series, out_path: Path) -> Path:
    chars = reports.fillna("").astype(str).str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(chars, bins=50, color="#3b6ea5", edgecolor="white")
    ax.set_xlabel("report length (characters)")
    ax.set_ylabel("reports")
    ax.set_title(f"Report length (median={chars.median():.0f}, p95={chars.quantile(0.95):.0f})")
    return _save(fig, out_path)
