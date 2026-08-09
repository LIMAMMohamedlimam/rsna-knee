"""Report census: language, length, structure, duplicates (Spec 01 §1.2.2).

Runs over ALL reports (they are cheap text), unlike the DICOM census which is sampled.
Duplicate detection matters for leakage: template reports shared across studies mean the
LLM label engine will produce identical labels for studies that must not sit in the same
fold by accident.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

import pandas as pd

UNKNOWN = "unknown"

# Section headers per language. Extend as EDA discovers languages (Spec 01 §1.2.2).
SECTION_PATTERNS: dict[str, str] = {
    "en": r"\b(?:FINDINGS|IMPRESSION|CONCLUSION|TECHNIQUE|COMPARISON|HISTORY)\b",
    "fr": r"\b(?:R[EÉ]SULTATS?|CONCLUSION|TECHNIQUE|INDICATION|COMPARAISON)\b",
    "es": r"\b(?:HALLAZGOS|CONCLUSI[OÓ]N|T[EÉ]CNICA|IMPRESI[OÓ]N)\b",
    "de": r"\b(?:BEFUND|BEURTEILUNG|TECHNIK|KLINIK)\b",
    "it": r"\b(?:REPERTI|CONCLUSIONI|TECNICA|QUESITO)\b",
    "pt": r"\b(?:ACHADOS|CONCLUS[AÃ]O|T[EÉ]CNICA)\b",
    "nl": r"\b(?:BEVINDINGEN|CONCLUSIE|TECHNIEK)\b",
}

_WS = re.compile(r"\s+")
_NON_ALPHA = re.compile(r"[^a-z\s]+")


def detect_languages(reports: pd.Series) -> pd.Series:
    """ISO-639-1 language per report. Deterministic (langdetect factory is seeded).

    Falls back to `unknown` for empty/undetectable text rather than raising, so one odd
    report cannot abort the census.
    """
    from langdetect import DetectorFactory, detect
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0

    def one(text: object) -> str:
        if not isinstance(text, str) or not text.strip():
            return UNKNOWN
        try:
            return detect(text)
        except LangDetectException:
            return UNKNOWN

    return reports.map(one)


def language_table(languages: pd.Series) -> pd.DataFrame:
    counts = languages.value_counts()
    return pd.DataFrame(
        {"language": counts.index, "count": counts.to_numpy(), "pct": counts.to_numpy() / len(languages)}
    ).reset_index(drop=True)


def length_stats(reports: pd.Series) -> pd.DataFrame:
    """Char and approximate-token (chars/4) percentiles. Feeds Spec 02 cost estimation."""
    chars = reports.fillna("").astype(str).str.len()
    q = chars.quantile([0.05, 0.5, 0.95]).to_dict()
    rows = {
        "p5": q[0.05],
        "p50": q[0.5],
        "p95": q[0.95],
        "max": chars.max(),
        "mean": chars.mean(),
        "total": chars.sum(),
    }
    return pd.DataFrame(
        {
            "stat": list(rows),
            "chars": [float(v) for v in rows.values()],
            "approx_tokens": [float(v) / 4 for v in rows.values()],
        }
    )


def section_header_counts(reports: pd.Series, languages: pd.Series) -> pd.DataFrame:
    """Per language: how many of its reports contain a recognised section header."""
    rows = []
    for lang, idx in languages.groupby(languages).groups.items():
        subset = reports.loc[idx].fillna("").astype(str)
        pattern = SECTION_PATTERNS.get(str(lang))
        if pattern is None:
            rows.append({"language": lang, "n_reports": len(subset), "n_with_header": None,
                         "pct_with_header": None, "pattern": "NO PATTERN — add to SECTION_PATTERNS"})
            continue
        hits = subset.str.upper().str.contains(pattern, regex=True, na=False)
        rows.append({"language": lang, "n_reports": len(subset), "n_with_header": int(hits.sum()),
                     "pct_with_header": float(hits.mean()), "pattern": pattern})
    return pd.DataFrame(rows).sort_values("n_reports", ascending=False).reset_index(drop=True)


def normalize_text(text: object) -> str:
    """Aggressive normalization for near-duplicate detection.

    Strips accents, case, digits and punctuation so that template reports differing only in
    measurements/dates/patient tokens collapse to the same key.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    return _WS.sub(" ", _NON_ALPHA.sub(" ", text)).strip()


def _hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:16]


def duplicate_clusters(
    reports: pd.Series, uids: pd.Series, min_chars: int = 40
) -> dict[str, pd.DataFrame]:
    """Exact (raw hash) and near (normalized hash) duplicate clusters, size>1 only.

    `min_chars` applies to the normalized text: very short reports collapse trivially and
    would swamp the near-duplicate table with noise.
    """
    df = pd.DataFrame({"StudyInstanceUID": uids.to_numpy(), "report": reports.fillna("").astype(str).to_numpy()})
    df["exact_key"] = df["report"].map(_hash)
    df["norm"] = df["report"].map(normalize_text)
    df["near_key"] = df["norm"].map(_hash).where(df["norm"].str.len() >= min_chars)

    def clusters(key: str) -> pd.DataFrame:
        grouped = df.dropna(subset=[key]).groupby(key)
        out = grouped.agg(
            cluster_size=("StudyInstanceUID", "size"),
            example_uid=("StudyInstanceUID", "first"),
            example_chars=("report", lambda s: len(s.iat[0])),
        )
        out = out[out["cluster_size"] > 1]
        return out.sort_values("cluster_size", ascending=False).reset_index(drop=True)

    exact, near = clusters("exact_key"), clusters("near_key")
    summary = pd.DataFrame(
        [
            {"kind": "exact", "n_clusters": len(exact),
             "n_studies_in_clusters": int(exact["cluster_size"].sum()) if len(exact) else 0,
             "largest_cluster": int(exact["cluster_size"].max()) if len(exact) else 0},
            {"kind": "near", "n_clusters": len(near),
             "n_studies_in_clusters": int(near["cluster_size"].sum()) if len(near) else 0,
             "largest_cluster": int(near["cluster_size"].max()) if len(near) else 0},
        ]
    )
    return {"exact": exact, "near": near, "summary": summary, "keys": df[["StudyInstanceUID", "exact_key", "near_key"]]}
