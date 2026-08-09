"""Site clustering and site-label confound detection (Spec 01 §1.2.4).

A "site" is approximated by a scanner fingerprint. If a site's label prevalence deviates
sharply from global, a model can score well by recognising the scanner instead of the
pathology — and the folds must at minimum record the cluster so the effect is measurable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.eda.labels import labeled_mask
from src.utils.constants import LABELS

FINGERPRINT_FIELDS = [
    "Manufacturer",
    "ManufacturerModelName",
    "MagneticFieldStrength",
    "pixel_spacing_round",
    "ImplementationVersionName",
]
MISC_CLUSTER = -1
PROTOCOL_CLUSTER_OFFSET = 1000  # ids for the protocol-signature fallback clustering


def build_fingerprints(meta_df: pd.DataFrame, spacing_decimals: int = 2) -> pd.DataFrame:
    """One fingerprint string per *study* from its sampled series headers.

    Uses the modal value of each field across the study's sampled series, so one odd series
    does not split a study off its own site.
    """
    df = meta_df.copy()
    spacing = pd.to_numeric(df.get("pixel_spacing_y"), errors="coerce")
    df["pixel_spacing_round"] = spacing.round(spacing_decimals)

    for field in FINGERPRINT_FIELDS:
        if field not in df.columns:
            df[field] = None
        df[field] = df[field].fillna("<missing>").astype(str)

    def modal(s: pd.Series) -> str:
        mode = s.mode()
        return str(mode.iat[0]) if len(mode) else "<missing>"

    per_study = df.groupby("StudyInstanceUID")[FINGERPRINT_FIELDS].agg(modal)
    per_study["fingerprint"] = per_study[FINGERPRINT_FIELDS].agg("|".join, axis=1)
    return per_study.reset_index()


def assign_site_clusters(fingerprints: pd.DataFrame, min_cluster_size: int = 20) -> pd.DataFrame:
    """Group identical fingerprints into `site_cluster` ids; small ones pool into -1 (misc).

    Plain groupby rather than KMeans: fingerprints are categorical, and an exact match is
    both more interpretable and perfectly deterministic.
    """
    sizes = fingerprints["fingerprint"].value_counts()
    keep = sizes[sizes >= min_cluster_size].index.tolist()
    mapping = {fp: i for i, fp in enumerate(sorted(keep))}

    out = fingerprints.copy()
    out["site_cluster"] = out["fingerprint"].map(mapping).fillna(MISC_CLUSTER).astype("int16")
    return out


def protocol_signature_clusters(series_df: pd.DataFrame, min_cluster_size: int = 20) -> pd.DataFrame:
    """Fallback clustering for studies outside the DICOM sample (Spec 01 §1.2.4).

    Signature = the sorted set of (plane, fluid, fatsat) combos a study owns. Scanners and
    sites tend to run fixed protocols, so this is a weak but free proxy for the fingerprint.
    Ids start at `PROTOCOL_CLUSTER_OFFSET` so they never collide with DICOM cluster ids.
    """
    combos = (
        series_df["Anatomical_Plane"].astype(str)
        + "/" + series_df["Fluid_Sensitive"].astype(str)
        + "/" + series_df["Fat_Suppression"].astype(str)
    )
    signature = (
        pd.DataFrame({"StudyInstanceUID": series_df["StudyInstanceUID"], "combo": combos})
        .drop_duplicates()
        .groupby("StudyInstanceUID")["combo"]
        .agg(lambda s: "+".join(sorted(s)))
    )
    sizes = signature.value_counts()
    keep = sizes[sizes >= min_cluster_size].index.tolist()
    mapping = {sig: PROTOCOL_CLUSTER_OFFSET + i for i, sig in enumerate(sorted(keep))}
    return pd.DataFrame(
        {
            "StudyInstanceUID": signature.index,
            "protocol_signature": signature.to_numpy(),
            "site_cluster": [mapping.get(s, MISC_CLUSTER) for s in signature],
        }
    ).reset_index(drop=True)


def cluster_sizes(clusters: pd.DataFrame) -> pd.DataFrame:
    out = (
        clusters.groupby("site_cluster")
        .agg(n_studies=("StudyInstanceUID", "nunique"), fingerprint=("fingerprint", "first"))
        .reset_index()
        .sort_values("n_studies", ascending=False)
    )
    out.loc[out["site_cluster"] == MISC_CLUSTER, "fingerprint"] = "<misc: small fingerprints pooled>"
    return out.reset_index(drop=True)


def site_label_prevalence(clusters: pd.DataFrame, train_df: pd.DataFrame) -> pd.DataFrame:
    """Per-cluster label prevalence on the GT-labeled subset (+ a `global` row)."""
    gt = train_df.loc[labeled_mask(train_df), ["StudyInstanceUID", *LABELS]]
    merged = gt.merge(clusters[["StudyInstanceUID", "site_cluster"]], on="StudyInstanceUID", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=["site_cluster", "n_studies", *LABELS])

    table = merged.groupby("site_cluster")[LABELS].mean()
    table.insert(0, "n_studies", merged.groupby("site_cluster").size())
    global_row = pd.DataFrame([[len(merged), *merged[LABELS].mean().to_numpy()]],
                              columns=["n_studies", *LABELS], index=["global"])
    return pd.concat([table, global_row]).rename_axis("site_cluster").reset_index()


def flag_site_confounds(
    prevalence: pd.DataFrame, ratio: float = 2.0, min_studies: int = 20
) -> pd.DataFrame:
    """Cluster x label pairs whose prevalence deviates more than `ratio`x from global.

    Reported under '**Site-label confounds**'. Clusters below `min_studies` are skipped:
    with few studies a 2x swing is just noise.
    """
    if prevalence.empty or "global" not in set(prevalence["site_cluster"].astype(str)):
        return pd.DataFrame(columns=["site_cluster", "label", "cluster_rate", "global_rate", "ratio"])

    global_row = prevalence[prevalence["site_cluster"].astype(str) == "global"].iloc[0]
    rows = []
    for _, row in prevalence[prevalence["site_cluster"].astype(str) != "global"].iterrows():
        if row["n_studies"] < min_studies:
            continue
        for label in LABELS:
            g, c = float(global_row[label]), float(row[label])
            if g <= 0 or np.isnan(c):
                continue
            r = c / g
            if r >= ratio or r <= 1 / ratio:
                rows.append({"site_cluster": row["site_cluster"], "label": label,
                             "cluster_rate": c, "global_rate": g, "ratio": r,
                             "n_studies": int(row["n_studies"])})
    out = pd.DataFrame(rows)
    return out.sort_values("ratio", ascending=False).reset_index(drop=True) if len(out) else out
