"""Contract tests for Spec 01 Task 1.2 (EDA)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda import labels, reports, series, sites
from src.eda.pipeline import run_eda, write_outputs
from src.eda.render import md_table
from src.utils.config import load_config
from src.utils.constants import LABELS
from tests.conftest import write_synthetic_dicom_tree


# --- 1.2.1 label census ------------------------------------------------------------------
def test_label_coverage_counts_partial_rows_separately():
    train = pd.DataFrame({"StudyInstanceUID": ["a", "b", "c"], **{c: [1.0, np.nan, np.nan] for c in LABELS}})
    train.loc[1, "ACL"] = 1.0  # b has one label only -> partial, not labeled
    coverage = labels.label_coverage(train)
    assert coverage == {"n_studies": 3, "n_labeled": 1, "n_partial": 1, "n_unlabeled": 1}


def test_prevalence_table_is_in_canonical_order_and_uses_labeled_subset_only(synthetic):
    table = labels.prevalence_table(synthetic.train)
    assert table["label"].tolist() == LABELS
    assert (table["n_pos"] + table["n_neg"] == labels.labeled_mask(synthetic.train).sum()).all()
    assert table["pos_rate"].between(0, 1).all()


def test_flag_rare_skewed_picks_up_both_tails():
    table = pd.DataFrame({"label": LABELS, "n_pos": [0] * 12, "n_neg": [0] * 12,
                          "pos_rate": [0.01, 0.7, *([0.3] * 10)]})
    flagged = labels.flag_rare_skewed(table, lo=0.05, hi=0.60)
    assert flagged["label"].tolist() == ["ACL", "MCL"]
    assert "pos_rate < 5%" in flagged.loc[0, "reason"]
    assert "pos_rate > 60%" in flagged.loc[1, "reason"]


def test_cooccurrence_conditional_is_a_real_conditional_probability():
    n = 100
    train = pd.DataFrame({"StudyInstanceUID": [str(i) for i in range(n)],
                          **{c: np.zeros(n) for c in LABELS}})
    train.loc[:39, "ACL"] = 1.0          # 40 ACL positives
    train.loc[:19, "MCL"] = 1.0          # 20 MCL positives, all inside ACL
    out = labels.cooccurrence(train)

    # P(ACL | MCL) == 1.0, P(MCL | ACL) == 0.5, Jaccard == 20/40
    assert out["conditional"].loc["ACL", "MCL"] == pytest.approx(1.0)
    assert out["conditional"].loc["MCL", "ACL"] == pytest.approx(0.5)
    assert out["jaccard"].loc["ACL", "MCL"] == pytest.approx(0.5)
    assert out["jaccard"].index.tolist() == LABELS


# --- 1.2.2 report census -----------------------------------------------------------------
def test_language_detection_is_deterministic_and_finds_the_planted_languages(synthetic):
    first = reports.detect_languages(synthetic.train["Report"])
    second = reports.detect_languages(synthetic.train["Report"])
    pd.testing.assert_series_equal(first, second)
    assert {"en", "fr", "es"} <= set(first.unique())


def test_language_detection_survives_empty_and_junk_text():
    detected = reports.detect_languages(pd.Series(["", None, "   ", "123 456"]))
    assert (detected == "unknown").all()


def test_normalize_text_collapses_accents_case_and_numbers():
    a = reports.normalize_text("Épanchement articulaire de 12 mm.")
    b = reports.normalize_text("epanchement   articulaire de 7 mm")
    assert a == b == "epanchement articulaire de mm"


def test_duplicate_clusters_finds_planted_templates(synthetic):
    out = reports.duplicate_clusters(synthetic.train["Report"], synthetic.train["StudyInstanceUID"])
    exact = out["summary"].set_index("kind").loc["exact"]
    assert exact["n_clusters"] > 0
    assert exact["largest_cluster"] > 1
    # Near-duplicate clustering is a superset of exact.
    assert out["summary"].set_index("kind").loc["near", "n_studies_in_clusters"] >= exact["n_studies_in_clusters"]


def test_length_stats_reports_token_estimate_for_spec02_costing(synthetic):
    stats = reports.length_stats(synthetic.train["Report"]).set_index("stat")
    assert stats.loc["p50", "chars"] > 0
    assert stats.loc["p50", "approx_tokens"] == pytest.approx(stats.loc["p50", "chars"] / 4)
    assert stats.loc["p95", "chars"] >= stats.loc["p50", "chars"]


def test_section_headers_flag_languages_without_a_pattern():
    text = pd.Series(["FINDINGS: normal.", "blah blah"])
    langs = pd.Series(["en", "zz"])
    table = reports.section_header_counts(text, langs).set_index("language")
    assert table.loc["en", "n_with_header"] == 1
    assert "NO PATTERN" in table.loc["zz", "pattern"]


# --- 1.2.3 series census -----------------------------------------------------------------
def test_canonical_protocol_table_is_ranked_by_study_coverage(synthetic):
    table = series.canonical_protocol_table(synthetic.series, top_n=6)
    assert len(table) <= 6
    assert table["study_coverage"].is_monotonic_decreasing
    assert table["study_coverage"].between(0, 1).all()
    assert set(series.PROTOCOL_KEYS) <= set(table.columns)


def test_plane_coverage_separates_fluid_and_fatsat(synthetic):
    coverage = series.plane_coverage(synthetic.series).set_index("plane")
    for plane in coverage.index:
        row = coverage.loc[plane]
        assert row["coverage_fatsat_fluid"] <= row["coverage_fluid"] <= row["coverage_any"]


def test_dicom_census_guards_absent_tags():
    meta = pd.DataFrame({"n_files": [30, 28], "rows": [256, 256], "cols": [256, 256],
                         "Manufacturer": [None, None]})
    out = series.dicom_sample_census(meta)
    assert "<tag absent in sample>" in out["manufacturer"]["Manufacturer"].tolist()
    assert out["numeric"].set_index("field").loc["n_files", "n"] == 2


# --- 1.2.4 site clustering ---------------------------------------------------------------
def test_site_clusters_group_identical_fingerprints_and_pool_small_ones():
    meta = pd.DataFrame({
        "StudyInstanceUID": [f"s{i}" for i in range(25)],
        "Manufacturer": ["SIEMENS"] * 22 + ["RARE"] * 3,
        "ManufacturerModelName": ["Aera"] * 22 + ["Odd"] * 3,
        "MagneticFieldStrength": [1.5] * 25,
        "pixel_spacing_y": [0.312] * 25,
        "ImplementationVersionName": ["v1"] * 25,
    })
    clustered = sites.assign_site_clusters(sites.build_fingerprints(meta), min_cluster_size=20)
    counts = clustered["site_cluster"].value_counts()
    assert counts[0] == 22
    assert counts[sites.MISC_CLUSTER] == 3


def test_flag_site_confounds_triggers_above_the_ratio():
    prevalence = pd.DataFrame([
        {"site_cluster": 0, "n_studies": 50, **{c: 0.10 for c in LABELS}},
        {"site_cluster": 1, "n_studies": 50, **{c: 0.10 for c in LABELS}},
        {"site_cluster": "global", "n_studies": 100, **{c: 0.10 for c in LABELS}},
    ])
    prevalence.loc[0, "ACL"] = 0.30  # 3x global
    flagged = sites.flag_site_confounds(prevalence, ratio=2.0, min_studies=20)
    assert flagged["label"].tolist() == ["ACL"]
    assert flagged.loc[0, "ratio"] == pytest.approx(3.0)


def test_small_clusters_are_not_flagged_as_confounds():
    prevalence = pd.DataFrame([
        {"site_cluster": 0, "n_studies": 5, **{c: 0.9 for c in LABELS}},
        {"site_cluster": "global", "n_studies": 100, **{c: 0.10 for c in LABELS}},
    ])
    assert len(sites.flag_site_confounds(prevalence, ratio=2.0, min_studies=20)) == 0


def test_protocol_signature_fallback_covers_every_study(synthetic):
    fallback = sites.protocol_signature_clusters(synthetic.series, min_cluster_size=20)
    assert set(fallback["StudyInstanceUID"]) == set(synthetic.series["StudyInstanceUID"])
    assert (fallback["site_cluster"] >= sites.PROTOCOL_CLUSTER_OFFSET).any()


# --- rendering ---------------------------------------------------------------------------
def test_md_table_handles_empty_and_nan():
    assert md_table(pd.DataFrame()) == "_(empty)_"
    rendered = md_table(pd.DataFrame({"a": [np.nan], "b": [1.5]}))
    assert "—" in rendered and "1.5000" in rendered


# --- end-to-end --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def eda_result(synthetic_raw, tmp_path_factory):
    out = tmp_path_factory.mktemp("eda_out")
    cfg = load_config(overrides=[
        f"paths.raw_dir={synthetic_raw.raw_dir}",
        f"paths.docs_dir={out}",
        f"paths.figures_dir={out / 'figures'}",
        f"paths.study_meta_path={out / 'study_meta.parquet'}",
    ])
    return run_eda(cfg), cfg, out


def test_run_eda_produces_every_required_section(eda_result):
    result, _, _ = eda_result
    for heading in ["Label census", "Rare/skewed labels", "Report census", "Series census",
                    "Canonical protocol table", "Site clustering", "Site–label confounds",
                    "Top 5 risks observed"]:
        assert heading in result.report_md, f"missing section: {heading}"
    assert len(result.risks) >= 1
    assert result.scalars["n_labeled"] > 0
    assert result.scalars["n_sampled_series"] > 0, "DICOM sample should have been read"


def test_run_eda_writes_figures_and_study_meta(eda_result):
    result, cfg, out = eda_result
    paths = write_outputs(result, cfg)
    assert paths["report"].exists()
    for name in ["label_cooccurrence", "series_crosstab", "report_length", "series_per_study"]:
        assert result.figures[name].exists(), f"missing figure {name}"

    meta = pd.read_parquet(paths["study_meta"])
    assert set(meta.columns) == {"StudyInstanceUID", "language", "PatientID", "site_cluster",
                                 "site_cluster_source"}
    assert len(meta) == result.scalars["n_studies"]
    assert meta["site_cluster"].notna().all()
    # PatientID came from the small DICOM tree -> partial coverage, which folds must tolerate.
    assert meta["PatientID"].notna().any()


def test_run_eda_is_deterministic(eda_result):
    result, cfg, _ = eda_result
    assert run_eda(cfg).report_md == result.report_md


def test_study_meta_feeds_make_folds(eda_result):
    """The EDA output must be directly consumable by Task 1.3 without reshaping."""
    from src.utils.folds import assign_folds, build_study_frame

    result, cfg, _ = eda_result
    train = pd.read_csv(cfg.paths.raw_dir + "/train.csv")
    series_df = pd.read_csv(cfg.paths.raw_dir + "/train_series.csv")

    frame, report = build_study_frame(train, series_df, result.study_meta)
    folds = assign_folds(frame, train, n_folds=cfg.n_folds, seed=cfg.seed, report=report)
    assert len(folds) == len(train)
    assert report.proxy_available["language"] is True
    assert report.proxy_available["site_cluster"] is True


def test_protocol_incompleteness_risk_only_fires_when_coverage_is_thin():
    from src.eda.pipeline import EdaResult, _collect_risks

    def risk_text(coverage: float) -> str:
        res = EdaResult(
            scalars={"n_labeled": 10, "n_studies": 20, "pct_site_cluster_from_dicom": 1.0},
            tables={"plane_coverage": pd.DataFrame(
                [{"plane": "Axial", "coverage_fatsat_fluid": coverage}])},
        )
        return " ".join(_collect_risks(res))

    assert "Protocol incompleteness" not in risk_text(1.0)
    assert "Protocol incompleteness" in risk_text(0.42)


# --- header sweep (Colab/Drive path) ------------------------------------------------------
def test_header_sweep_covers_every_study_that_has_dicom(tmp_path, synthetic):
    from src.eda.pipeline import sweep_study_headers

    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    write_synthetic_dicom_tree(synthetic, raw, n_studies=12)
    out = tmp_path / "headers.parquet"

    swept = sweep_study_headers(raw, synthetic.series, out, flush_every=5)
    assert len(swept) == 12
    assert swept["PatientID"].notna().all()
    assert out.exists()


def test_header_sweep_resumes_instead_of_restarting(tmp_path, synthetic):
    """A Colab timeout mid-sweep must not throw away hours of Drive reads."""
    from src.eda.pipeline import sweep_study_headers

    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    write_synthetic_dicom_tree(synthetic, raw, n_studies=12)
    out = tmp_path / "headers.parquet"

    partial = sweep_study_headers(raw, synthetic.series.head(9), out)
    n_partial = len(partial)
    assert 0 < n_partial < 12

    full = sweep_study_headers(raw, synthetic.series, out)
    assert len(full) == 12
    assert full["StudyInstanceUID"].is_unique
    # Nothing was re-read: the already-known studies survived untouched.
    assert set(partial["StudyInstanceUID"]) <= set(full["StudyInstanceUID"])


def test_header_sweep_tolerates_studies_without_files(tmp_path, synthetic):
    from src.eda.pipeline import sweep_study_headers

    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    write_synthetic_dicom_tree(synthetic, raw, n_studies=3)

    swept = sweep_study_headers(raw, synthetic.series, tmp_path / "h.parquet")
    assert len(swept) == 3  # the other ~687 studies have no directory and are skipped


def test_header_sweep_returns_empty_without_a_series_root(tmp_path, synthetic):
    from src.eda.pipeline import sweep_study_headers

    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    assert sweep_study_headers(raw, synthetic.series, tmp_path / "h.parquet").empty


def test_sweep_gives_full_patient_id_coverage_so_folds_group_by_patient(tmp_path, synthetic):
    """The whole point of the sweep: patient-level grouping instead of study-level."""
    from src.eda.pipeline import run_eda
    from src.utils.folds import build_study_frame

    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    write_synthetic_dicom_tree(synthetic, raw, n_studies=len(synthetic.train))

    out = tmp_path / "out"
    cfg = load_config(overrides=[
        f"paths.raw_dir={raw}", f"paths.docs_dir={out}", f"paths.figures_dir={out / 'fig'}",
        f"paths.study_meta_path={out / 'meta.parquet'}",
        f"paths.study_headers_path={out / 'headers.parquet'}",
        "eda.patient_sweep=true",
    ])
    result = run_eda(cfg)
    assert result.scalars["header_source"] == "sweep"
    assert result.scalars["pct_patient_id"] == 1.0

    _, report = build_study_frame(synthetic.train, synthetic.series, result.study_meta)
    assert report.group_key == "PatientID"


def test_low_patient_id_coverage_is_flagged_as_a_risk(eda_result):
    """The shared fixture writes DICOM for only a handful of studies, so patient grouping
    degrades — the report must say so rather than let it pass silently."""
    result, _, _ = eda_result
    assert result.scalars["pct_patient_id"] < 0.95
    assert any("cannot group by patient" in risk for risk in result.risks)
