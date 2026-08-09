"""EDA orchestration (Spec 01 Task 1.2). `scripts/run_eda.py` is a thin wrapper on this.

Produces, deterministically:
  docs/eda_report.md            the report, sections in spec order
  docs/figures/*.png            co-occurrence heatmap, protocol cross-tab, 2 distributions
  artifacts/eda/study_meta.parquet   PatientID / site_cluster / language per study,
                                     consumed by scripts/make_folds.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig

from src.data.dicom_reader import normalize_header_rows, read_series_meta
from src.eda import figures, labels, reports, series, sites
from src.eda.render import bullet_list, md_matrix, md_table, section
from src.utils.config import cfg_path, config_hash
from src.utils.constants import LABELS
from src.utils.io import git_sha, resolve, write_parquet
from src.utils.io import load_raw as io_load_raw

# Below this fat-sat fluid study coverage, protocol incompleteness becomes a reported risk.
PROTOCOL_COVERAGE_RISK_THRESHOLD = 0.90


@dataclass
class EdaResult:
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    scalars: dict[str, Any] = field(default_factory=dict)
    risks: list[str] = field(default_factory=list)
    figures: dict[str, Path] = field(default_factory=dict)
    study_meta: pd.DataFrame | None = None
    report_md: str = ""


# Re-exported so the EDA keeps its own entry point for reading the competition CSVs.
load_raw = io_load_raw


def _first_series_per_study(series_df: pd.DataFrame) -> pd.DataFrame:
    return (
        series_df.sort_values(["StudyInstanceUID", "SeriesInstanceUID"], kind="stable")
        .groupby("StudyInstanceUID", sort=True)
        .head(1)
        .reset_index(drop=True)
    )


def sweep_study_headers(
    raw_dir: Path,
    series_df: pd.DataFrame,
    out_path: Path,
    *,
    probe_slices: int = 1,
    flush_every: int = 500,
    ctx: Any | None = None,
) -> pd.DataFrame:
    """Read ONE DICOM header per study, for every study. Resumable.

    Headers are a few KB, so this is not the 570 GB scan Spec 01 §1.2 forbids — but it is the
    only way to get `PatientID` for every study, which decides whether folds can group by
    patient (§1.3) instead of falling back to study level. §1.2.4 explicitly sanctions
    extending the sample.

    Progress is checkpointed to `out_path` every `flush_every` studies, so an interrupted
    session (Colab timeout, Drive hiccup) resumes where it stopped instead of restarting.
    """
    from src.utils.logging import ProgressLogger, get_logger

    log = get_logger(__name__)
    root = raw_dir / "train_series"
    if not root.exists():
        log.warning("%s does not exist — skipping header sweep", root)
        return pd.DataFrame()

    done: set[str] = set()
    existing = pd.DataFrame()
    if out_path.exists():
        existing = normalize_header_rows(pd.read_parquet(out_path))
        done = set(existing["StudyInstanceUID"].astype(str))
        log.info("resuming header sweep: %d studies already read", len(done))

    targets = _first_series_per_study(series_df)
    todo = targets[~targets["StudyInstanceUID"].astype(str).isin(done)]
    log.info("header sweep: %d studies to read (%d done)", len(todo), len(done))
    if todo.empty:
        return existing

    rows: list[dict] = []
    n_missing = 0
    progress = ProgressLogger(len(todo), "header_sweep", ctx)

    def flush() -> None:
        nonlocal existing, rows
        if not rows:
            return
        existing = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True)
        write_parquet(existing, out_path)
        rows = []

    for study_uid, series_uid in zip(todo["StudyInstanceUID"], todo["SeriesInstanceUID"], strict=True):
        path = root / str(study_uid) / str(series_uid)
        if path.exists():
            rows.append(read_series_meta(path, probe_slices=probe_slices).to_row())
        else:
            n_missing += 1
        progress.update(errors=n_missing)
        if len(rows) >= flush_every:
            flush()

    flush()
    progress.finish()
    if n_missing:
        log.warning("%d studies had no readable series directory", n_missing)
    return existing


def sample_series_meta(
    raw_dir: Path, series_df: pd.DataFrame, n_sample: int, seed: int
) -> pd.DataFrame:
    """Read headers for ≤`n_sample` series — one per study, across as many studies as possible.

    One series per study maximises site-fingerprint coverage (§1.2.4) while staying inside the
    sample budget; never scan the full 570 GB (§1.2).
    """
    root = raw_dir / "train_series"
    if not root.exists():
        return pd.DataFrame()

    first_per_study = _first_series_per_study(series_df)
    if len(first_per_study) > n_sample:  # deterministic subsample
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(len(first_per_study), size=n_sample, replace=False))
        first_per_study = first_per_study.iloc[idx].reset_index(drop=True)

    rows = []
    for study_uid, series_uid in zip(
        first_per_study["StudyInstanceUID"], first_per_study["SeriesInstanceUID"], strict=True
    ):
        path = root / str(study_uid) / str(series_uid)
        if not path.exists():
            continue
        rows.append(read_series_meta(path).to_row())
    return pd.DataFrame(rows)


def run_eda(cfg: DictConfig, ctx: Any | None = None) -> EdaResult:
    raw_dir = cfg_path(cfg, "raw_dir")
    fig_dir = cfg_path(cfg, "figures_dir")
    res = EdaResult()

    train, series_df = load_raw(raw_dir)

    # --- 1.2.1 label census -------------------------------------------------------------
    res.scalars.update(labels.label_coverage(train))
    prevalence = labels.prevalence_table(train)
    res.tables["prevalence"] = prevalence
    res.tables["rare_skewed"] = labels.flag_rare_skewed(
        prevalence, cfg.eda.rare_label_pos_rate_lo, cfg.eda.rare_label_pos_rate_hi
    )
    cooc = labels.cooccurrence(train)
    res.tables["cooc_jaccard"] = cooc["jaccard"]
    res.tables["cooc_conditional"] = cooc["conditional"]
    res.figures["label_cooccurrence"] = figures.label_cooccurrence_heatmap(
        cooc["jaccard"], fig_dir / "label_cooccurrence.png"
    )

    # --- 1.2.2 report census ------------------------------------------------------------
    report_text = train["Report"] if "Report" in train.columns else pd.Series([""] * len(train))
    languages = reports.detect_languages(report_text)
    res.tables["languages"] = reports.language_table(languages)
    res.tables["report_length"] = reports.length_stats(report_text)
    res.tables["section_headers"] = reports.section_header_counts(report_text, languages)
    dups = reports.duplicate_clusters(
        report_text, train["StudyInstanceUID"], cfg.eda.near_dup_min_chars
    )
    res.tables["dup_summary"] = dups["summary"]
    res.tables["dup_top"] = dups["near"].head(15)
    res.figures["report_length"] = figures.report_length_hist(
        report_text, fig_dir / "report_length.png"
    )

    # --- 1.2.3 series census ------------------------------------------------------------
    res.tables["series_per_study"] = series.series_per_study(series_df)
    crosstab = series.protocol_crosstab(series_df)
    res.tables["protocol_crosstab"] = crosstab
    res.tables["canonical_protocol"] = series.canonical_protocol_table(
        series_df, cfg.eda.protocol_table_top_n
    )
    res.tables["plane_coverage"] = series.plane_coverage(series_df)
    res.figures["series_crosstab"] = figures.series_crosstab_figure(
        crosstab, fig_dir / "series_crosstab.png"
    )
    res.figures["series_per_study"] = figures.series_per_study_hist(
        series_df, fig_dir / "series_per_study.png"
    )

    if cfg.eda.patient_sweep:
        meta_df = sweep_study_headers(
            raw_dir, series_df, resolve(cfg.paths.study_headers_path),
            flush_every=int(cfg.eda.sweep_flush_every), ctx=ctx,
        )
    else:
        meta_df = sample_series_meta(raw_dir, series_df, cfg.eda.dicom_sample_series, cfg.seed)
    res.scalars["n_sampled_series"] = len(meta_df)
    res.scalars["header_source"] = "sweep" if cfg.eda.patient_sweep else "sample"
    if len(meta_df):
        for name, table in series.dicom_sample_census(meta_df).items():
            res.tables[f"dicom_{name}"] = table

    # --- 1.2.4 site clustering ----------------------------------------------------------
    fallback = sites.protocol_signature_clusters(series_df, cfg.eda.min_cluster_size)
    if len(meta_df):
        fingerprints = sites.build_fingerprints(meta_df)
        clustered = sites.assign_site_clusters(fingerprints, cfg.eda.min_cluster_size)
        res.tables["site_cluster_sizes"] = sites.cluster_sizes(clustered)
        site_prev = sites.site_label_prevalence(clustered, train)
        res.tables["site_prevalence"] = site_prev
        res.tables["site_confounds"] = sites.flag_site_confounds(
            site_prev, cfg.eda.site_confound_ratio, cfg.eda.min_cluster_size
        )
    else:
        clustered = pd.DataFrame(columns=["StudyInstanceUID", "site_cluster"])
        for key in ("site_cluster_sizes", "site_prevalence", "site_confounds"):
            res.tables[key] = pd.DataFrame()

    res.study_meta = _build_study_meta(train, meta_df, clustered, fallback, languages)
    res.scalars["pct_site_cluster_from_dicom"] = float(
        (res.study_meta["site_cluster_source"] == "dicom").mean()
    )
    res.scalars["pct_patient_id"] = float(res.study_meta["PatientID"].notna().mean())

    res.risks = _collect_risks(res)
    res.report_md = _render(res, cfg)
    return res


def _build_study_meta(
    train: pd.DataFrame,
    meta_df: pd.DataFrame,
    clustered: pd.DataFrame,
    fallback: pd.DataFrame,
    languages: pd.Series,
) -> pd.DataFrame:
    """Per-study table consumed by make_folds.py: PatientID, site_cluster, language."""
    out = pd.DataFrame({"StudyInstanceUID": train["StudyInstanceUID"].astype(str)})
    out["language"] = languages.to_numpy() if len(languages) == len(out) else "unknown"

    if len(meta_df) and "PatientID" in meta_df.columns:
        pid = meta_df.dropna(subset=["PatientID"]).groupby("StudyInstanceUID")["PatientID"].first()
        out["PatientID"] = out["StudyInstanceUID"].map(pid)
    else:
        out["PatientID"] = pd.NA

    dicom_cluster = (
        clustered.set_index("StudyInstanceUID")["site_cluster"] if len(clustered) else pd.Series(dtype="int16")
    )
    proto_cluster = (
        fallback.set_index("StudyInstanceUID")["site_cluster"] if len(fallback) else pd.Series(dtype="int16")
    )
    from_dicom = out["StudyInstanceUID"].map(dicom_cluster)
    from_proto = out["StudyInstanceUID"].map(proto_cluster)

    out["site_cluster"] = from_dicom.fillna(from_proto).fillna(sites.MISC_CLUSTER).astype("int16")
    out["site_cluster_source"] = np.where(
        from_dicom.notna(), "dicom", np.where(from_proto.notna(), "protocol", "none")
    )
    return out


def _collect_risks(res: EdaResult) -> list[str]:
    """Auto-filled 'Top 5 risks observed' entries, most actionable first (§1.2.5)."""
    risks: list[str] = []

    rare = res.tables.get("rare_skewed")
    if rare is not None and len(rare):
        names = ", ".join(f"{r.label} ({r.pos_rate:.1%})" for r in rare.itertuples())
        risks.append(f"**Rare/skewed labels** — {names}. Expect unstable per-label AUC; these "
                     f"drive the Spec 04 §4.4 rare-label program and the ±25% fold-balance test.")

    confounds = res.tables.get("site_confounds")
    if confounds is not None and len(confounds):
        top = confounds.iloc[0]
        risks.append(f"**Site–label confounds** — {len(confounds)} cluster×label pairs deviate >2× "
                     f"from global (worst: cluster {top.site_cluster} / {top.label}, ratio "
                     f"{top.ratio:.2f}). Model may learn the scanner; check per-cluster AUC.")

    dup = res.tables.get("dup_summary")
    if dup is not None and len(dup):
        near = dup[dup["kind"] == "near"].iloc[0]
        if near["n_studies_in_clusters"] > 0:
            risks.append(f"**Template/duplicate reports** — {near['n_studies_in_clusters']} studies in "
                         f"{near['n_clusters']} near-duplicate clusters (largest {near['largest_cluster']}). "
                         f"Identical LLM labels for these; leakage risk if they straddle folds.")

    labeled_frac = res.scalars.get("n_labeled", 0) / max(res.scalars.get("n_studies", 1), 1)
    risks.append(f"**Label scarcity** — only {labeled_frac:.1%} of studies carry ground truth "
                 f"({res.scalars.get('n_labeled', 0)}/{res.scalars.get('n_studies', 0)}). "
                 f"Spec 02 label quality upper-bounds everything downstream.")

    coverage = res.tables.get("plane_coverage")
    if coverage is not None and len(coverage):
        weakest = coverage.sort_values("coverage_fatsat_fluid").iloc[0]
        # Only a risk when coverage is actually thin — Spec 07 §7.1: Contusion/Fracture are
        # near-undetectable without fat-sat fluid sequences.
        if weakest.coverage_fatsat_fluid < PROTOCOL_COVERAGE_RISK_THRESHOLD:
            risks.append(f"**Protocol incompleteness** — only {weakest.coverage_fatsat_fluid:.1%} "
                         f"of studies have a {weakest.plane} fat-sat fluid series. Contusion and "
                         f"Fracture are near-undetectable without one (Spec 07 §7.1) — consider "
                         f"the loss-masking option in §7.4.5 for the uncovered stratum.")

    pct_pid = res.scalars.get("pct_patient_id", 0.0)
    if pct_pid < 0.95:
        risks.append(f"**Folds cannot group by patient** — PatientID is available for only "
                     f"{pct_pid:.1%} of studies (need ≥95%). `make folds` will group by study, "
                     f"so a patient imaged twice can land in two folds and inflate CV. Run the "
                     f"header sweep (`eda.patient_sweep: true`) to completion first.")

    pct = res.scalars.get("pct_site_cluster_from_dicom", 0.0)
    if pct < 0.5:
        risks.append(f"**Site clusters are mostly proxies** — only {pct:.1%} of studies have a DICOM "
                     f"fingerprint (sample budget); the rest use the protocol-signature fallback. "
                     f"Raise `eda.dicom_sample_series` before trusting the confound analysis.")
    return risks


def _render(res: EdaResult, cfg: DictConfig) -> str:
    t, s = res.tables, res.scalars
    parts: list[str] = [
        "# EDA report — RSNA Knee Abnormality Detection",
        "",
        "_Generated by `make eda` (`scripts/run_eda.py`). Deterministic: no timestamps — "
        "a rerun on unchanged data yields a byte-identical file._",
        "",
        f"- config hash: `{config_hash(cfg)}`",
        f"- git sha: `{git_sha()}`",
        f"- seed: `{cfg.seed}`",
        # Provenance: makes a report generated from a stand-in dataset obvious at a glance.
        f"- data (`RSNA_RAW`): `{cfg_path(cfg, 'raw_dir')}`",
        f"- DICOM headers read: `{s.get('n_sampled_series', 0)}` "
        f"(source: {s.get('header_source', 'sample')}; PatientID coverage "
        f"{s.get('pct_patient_id', 0.0):.1%})",
        section("1. Label census (GT-labeled subset only)"),
        md_table(pd.DataFrame([{k: s.get(k) for k in
                                ("n_studies", "n_labeled", "n_partial", "n_unlabeled")}])),
        "",
        "### Per-label prevalence", "", md_table(t.get("prevalence")), "",
        "### **Rare/skewed labels**", "", md_table(t.get("rare_skewed")), "",
        "### Co-occurrence — Jaccard", "", md_matrix(t.get("cooc_jaccard")), "",
        "### Co-occurrence — conditional P(row | col)", "", md_matrix(t.get("cooc_conditional")), "",
        "![label co-occurrence](figures/label_cooccurrence.png)",
        section("2. Report census"),
        "### Languages", "", md_table(t.get("languages")), "",
        "### Length distribution (feeds Spec 02 cost estimate)", "", md_table(t.get("report_length"), 1), "",
        "### Structural sections", "", md_table(t.get("section_headers")), "",
        "### Duplicate reports", "", md_table(t.get("dup_summary")), "",
        "Largest near-duplicate clusters:", "", md_table(t.get("dup_top")), "",
        "![report length](figures/report_length.png)",
        section("3. Series census"),
        "### Series per study", "", md_table(t.get("series_per_study"), 2), "",
        "### Protocol cross-tab (plane × fluid × fat-sat)", "", md_table(t.get("protocol_crosstab")), "",
        "### **Canonical protocol table** — routing input for Spec 03 Task 3.3", "",
        md_table(t.get("canonical_protocol")), "",
        "### Plane coverage (predicts routing fallback rate)", "", md_table(t.get("plane_coverage")), "",
        "![series crosstab](figures/series_crosstab.png)",
        "",
        "![series per study](figures/series_per_study.png)",
    ]

    if s.get("n_sampled_series", 0):
        parts += [
            "### Sampled DICOM headers", "",
            "Numeric fields:", "", md_table(t.get("dicom_numeric"), 3), "",
            "Transfer syntaxes:", "", md_table(t.get("dicom_transfer_syntax")), "",
            "Manufacturer:", "", md_table(t.get("dicom_manufacturer")), "",
            "Model:", "", md_table(t.get("dicom_model"), max_rows=15), "",
            "Field strength:", "", md_table(t.get("dicom_field_strength")), "",
            "In-plane shapes:", "", md_table(t.get("dicom_shape"), max_rows=15), "",
        ]
    else:
        parts += ["", "_No DICOM sample available — series-level header census skipped._", ""]

    parts += [
        section("4. Site clustering"),
        f"Studies whose site cluster comes from a DICOM fingerprint: "
        f"{s.get('pct_site_cluster_from_dicom', 0.0):.1%} (rest use the protocol-signature fallback).",
        "", "### Cluster sizes", "", md_table(t.get("site_cluster_sizes"), max_rows=20), "",
        "### Per-cluster label prevalence", "", md_table(t.get("site_prevalence"), 3), "",
        "### **Site–label confounds**", "", md_table(t.get("site_confounds"), 3, max_rows=25), "",
        section("5. Top 5 risks observed"),
        bullet_list(res.risks[:5]),
        "",
        "_Free-text additions from human review go below this line._",
        "",
    ]
    return "\n".join(parts) + "\n"


def write_outputs(res: EdaResult, cfg: DictConfig) -> dict[str, Path]:
    report_path = cfg_path(cfg, "docs_dir") / "eda_report.md"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(res.report_md, encoding="utf-8")

    meta_path = write_parquet(res.study_meta, cfg.paths.study_meta_path)
    return {"report": report_path, "study_meta": meta_path, **res.figures}
