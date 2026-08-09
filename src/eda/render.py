"""Markdown rendering for the EDA report.

Deterministic by construction: no timestamps, fixed float formatting, stable row order.
`make eda` twice on the same data must produce a byte-identical report (Spec 01 §1.2).
"""

from __future__ import annotations

import pandas as pd


def fmt(value: object, decimals: int = 4) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.{decimals}f}"
    return str(value)


def md_table(df: pd.DataFrame, decimals: int = 4, max_rows: int | None = None) -> str:
    """Render a DataFrame as a GitHub markdown table."""
    if df is None or len(df) == 0:
        return "_(empty)_"
    shown, truncated = (df.head(max_rows), len(df) > max_rows) if max_rows else (df, False)

    header = "| " + " | ".join(str(c) for c in shown.columns) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = [
        "| " + " | ".join(fmt(v, decimals) for v in row) + " |"
        for row in shown.itertuples(index=False, name=None)
    ]
    out = "\n".join([header, rule, *body])
    if truncated:
        out += f"\n\n_… {len(df) - max_rows} more rows omitted._"
    return out


def md_matrix(df: pd.DataFrame, decimals: int = 2) -> str:
    """Render a labelled square matrix (co-occurrence) with the index as first column."""
    return md_table(df.round(decimals).reset_index().rename(columns={"index": ""}), decimals)


def section(title: str, level: int = 2) -> str:
    return f"\n{'#' * level} {title}\n"


def bullet_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "_(none)_"
