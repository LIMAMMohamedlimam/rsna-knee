"""Series census and the canonical protocol table (Spec 01 §1.2.3).

The protocol table is the direct input to Spec 03's routing: the top-N (plane, fluid, fatsat)
combos by *study coverage* — how many studies own at least one such series — not by raw
series count. Coverage is what decides whether a routing slot can be filled without fallback.
"""

from __future__ import annotations

import pandas as pd

PROTOCOL_KEYS = ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"]


def series_per_study(series_df: pd.DataFrame) -> pd.DataFrame:
    counts = series_df.groupby("StudyInstanceUID")["SeriesInstanceUID"].nunique()
    q = counts.quantile([0.05, 0.5, 0.95]).to_dict()
    return pd.DataFrame(
        [
            {"stat": "n_studies", "value": float(counts.size)},
            {"stat": "p5", "value": float(q[0.05])},
            {"stat": "p50", "value": float(q[0.5])},
            {"stat": "p95", "value": float(q[0.95])},
            {"stat": "min", "value": float(counts.min())},
            {"stat": "max", "value": float(counts.max())},
            {"stat": "mean", "value": float(counts.mean())},
        ]
    )


def protocol_crosstab(series_df: pd.DataFrame) -> pd.DataFrame:
    """plane x fluid x fatsat: series count, study coverage count and fraction."""
    n_studies = series_df["StudyInstanceUID"].nunique()
    grouped = series_df.groupby(PROTOCOL_KEYS, dropna=False)
    out = grouped.agg(
        n_series=("SeriesInstanceUID", "nunique"),
        n_studies=("StudyInstanceUID", "nunique"),
    ).reset_index()
    out["study_coverage"] = out["n_studies"] / max(n_studies, 1)
    out["series_per_covered_study"] = out["n_series"] / out["n_studies"].clip(lower=1)
    return out.sort_values("study_coverage", ascending=False).reset_index(drop=True)


def canonical_protocol_table(series_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """Top-N protocol combos by study coverage — the routing input for Spec 03 Task 3.3."""
    return protocol_crosstab(series_df).head(top_n).reset_index(drop=True)


def plane_coverage(series_df: pd.DataFrame) -> pd.DataFrame:
    """Per-plane study coverage, split by whether a fluid-sensitive series exists.

    Directly predicts the Spec 03 fallback rate: a study with no axial fluid-sensitive series
    cannot fill routing slot 3 and must fall back.
    """
    n_studies = series_df["StudyInstanceUID"].nunique()
    rows = []
    for plane, group in series_df.groupby("Anatomical_Plane", dropna=False):
        fluid = group[group["Fluid_Sensitive"] == 1]
        fatsat_fluid = fluid[fluid["Fat_Suppression"] == 1]
        rows.append(
            {
                "plane": plane,
                "studies_any": group["StudyInstanceUID"].nunique(),
                "studies_fluid": fluid["StudyInstanceUID"].nunique(),
                "studies_fatsat_fluid": fatsat_fluid["StudyInstanceUID"].nunique(),
                "coverage_any": group["StudyInstanceUID"].nunique() / max(n_studies, 1),
                "coverage_fluid": fluid["StudyInstanceUID"].nunique() / max(n_studies, 1),
                "coverage_fatsat_fluid": fatsat_fluid["StudyInstanceUID"].nunique() / max(n_studies, 1),
            }
        )
    return pd.DataFrame(rows).sort_values("coverage_any", ascending=False).reset_index(drop=True)


def dicom_sample_census(meta_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Distributions over the sampled DICOM headers (guarding absent tags)."""

    def value_counts(col: str) -> pd.DataFrame:
        if col not in meta_df.columns or meta_df[col].isna().all():
            return pd.DataFrame({col: ["<tag absent in sample>"], "count": [0]})
        counts = meta_df[col].fillna("<missing>").astype(str).value_counts()
        return pd.DataFrame({col: counts.index, "count": counts.to_numpy()}).reset_index(drop=True)

    def numeric_stats(cols: list[str]) -> pd.DataFrame:
        rows = []
        for col in cols:
            if col not in meta_df.columns:
                continue
            values = pd.to_numeric(meta_df[col], errors="coerce").dropna()
            if values.empty:
                rows.append({"field": col, "n": 0, "p5": None, "p50": None, "p95": None,
                             "min": None, "max": None})
                continue
            q = values.quantile([0.05, 0.5, 0.95]).to_dict()
            rows.append({"field": col, "n": int(values.size), "p5": q[0.05], "p50": q[0.5],
                         "p95": q[0.95], "min": values.min(), "max": values.max()})
        return pd.DataFrame(rows)

    return {
        "numeric": numeric_stats(
            ["n_files", "rows", "cols", "pixel_spacing_y", "pixel_spacing_x", "slice_thickness"]
        ),
        "transfer_syntax": value_counts("transfer_syntax"),
        "manufacturer": value_counts("Manufacturer"),
        "model": value_counts("ManufacturerModelName"),
        "field_strength": value_counts("MagneticFieldStrength"),
        "shape": (
            meta_df.assign(shape=meta_df["rows"].astype("Int64").astype(str) + "x" + meta_df["cols"].astype("Int64").astype(str))
            ["shape"].value_counts().rename_axis("shape").reset_index(name="count")
            if {"rows", "cols"} <= set(meta_df.columns) else pd.DataFrame()
        ),
    }
