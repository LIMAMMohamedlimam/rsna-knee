"""Label census and co-occurrence structure (Spec 01 §1.2.1).

All statistics here are computed on the GT-labeled subset only — the LLM labels do not
exist yet at Spec 01 time, and mixing them in later would silently change these baselines.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.constants import LABELS


def labeled_mask(train_df: pd.DataFrame) -> pd.Series:
    """True where the full 12-label vector is present."""
    return train_df[LABELS].notna().all(axis=1)


def label_coverage(train_df: pd.DataFrame) -> dict[str, int]:
    mask = labeled_mask(train_df)
    partial = train_df[LABELS].notna().any(axis=1) & ~mask
    return {
        "n_studies": int(len(train_df)),
        "n_labeled": int(mask.sum()),
        "n_partial": int(partial.sum()),
        "n_unlabeled": int((~mask & ~partial).sum()),
    }


def prevalence_table(train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-label n_pos / n_neg / pos_rate on the labeled subset, canonical label order."""
    y = train_df.loc[labeled_mask(train_df), LABELS]
    n_pos = (y > 0.5).sum()
    n_neg = (y <= 0.5).sum()
    return pd.DataFrame(
        {
            "label": LABELS,
            "n_pos": n_pos.reindex(LABELS).astype(int).to_numpy(),
            "n_neg": n_neg.reindex(LABELS).astype(int).to_numpy(),
            "pos_rate": (n_pos / (n_pos + n_neg)).reindex(LABELS).to_numpy(),
        }
    )


def flag_rare_skewed(
    prevalence: pd.DataFrame, lo: float = 0.05, hi: float = 0.60
) -> pd.DataFrame:
    """Labels outside [lo, hi] — reported under '**Rare/skewed labels**'."""
    out = prevalence[(prevalence["pos_rate"] < lo) | (prevalence["pos_rate"] > hi)].copy()
    out["reason"] = np.where(out["pos_rate"] < lo, f"pos_rate < {lo:.0%}", f"pos_rate > {hi:.0%}")
    return out.reset_index(drop=True)


def cooccurrence(train_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """12x12 Jaccard and conditional P(row | col) on the labeled subset.

    Both are returned: Jaccard is symmetric and good for the heatmap, the conditional matrix
    is the one that maps onto the Spec 07 §7.2 dependency ladder (e.g. P(ACL | MCL)).
    """
    y = (train_df.loc[labeled_mask(train_df), LABELS] > 0.5).to_numpy().astype("int64")
    inter = y.T @ y                                   # |A & B|
    counts = np.diag(inter).astype("float64")         # |A|
    union = counts[:, None] + counts[None, :] - inter

    with np.errstate(divide="ignore", invalid="ignore"):
        jaccard = np.where(union > 0, inter / union, np.nan)
        conditional = np.where(counts[None, :] > 0, inter / counts[None, :], np.nan)

    return {
        "jaccard": pd.DataFrame(jaccard, index=LABELS, columns=LABELS),
        "conditional": pd.DataFrame(conditional, index=LABELS, columns=LABELS),
        "intersection": pd.DataFrame(inter, index=LABELS, columns=LABELS),
    }
