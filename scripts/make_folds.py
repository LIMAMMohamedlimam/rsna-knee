#!/usr/bin/env python3
"""Spec 01 Task 1.3 — generate artifacts/folds.parquet ONCE.

Folds are frozen (CLAUDE.md §3.2): if the output already exists this script logs its hash
and exits non-zero. Delete the file by hand if you truly mean to regenerate.

    make folds
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.config import cfg_path, load_config  # noqa: E402
from src.utils.folds import (  # noqa: E402
    FoldsFrozenError,
    assert_not_frozen,
    assign_folds,
    build_study_frame,
    fold_prevalence_table,
)
from src.utils.io import file_hash, resolve, write_parquet  # noqa: E402
from src.utils.logging import log_dataframe, setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--override", nargs="*", default=None)
    parser.add_argument("--log-level", default=None)
    args = parser.parse_args()

    cfg = load_config(base=args.config, overrides=args.override)

    with setup_logging(cfg, "make_folds", console_level=args.log_level) as ctx:
        set_seed(cfg.seed)

        out_path = resolve(cfg.paths.folds_path)
        try:
            assert_not_frozen(out_path)
        except FoldsFrozenError as exc:
            ctx.log.error("%s", exc)
            ctx.log_event("folds_frozen", path=str(out_path))
            ctx.close("refused")
            return 1

        raw_dir = cfg_path(cfg, "raw_dir")
        with ctx.step("load_raw", raw_dir=str(raw_dir)):
            train = pd.read_csv(raw_dir / "train.csv")
            series_df = pd.read_csv(raw_dir / "train_series.csv")
        log_dataframe(ctx.log, "train.csv", train)
        log_dataframe(ctx.log, "train_series.csv", series_df)

        meta_path = resolve(cfg.paths.study_meta_path)
        if meta_path.exists():
            study_meta = pd.read_parquet(meta_path)
            ctx.log.info("study_meta: %s", meta_path)
            log_dataframe(ctx.log, "study_meta", study_meta)
        else:
            study_meta = None
            ctx.log.warning("%s not found — run `make eda` first. Without it the proxy vector "
                            "loses language + site_cluster and grouping falls back to "
                            "StudyInstanceUID, which lets one patient span two folds.", meta_path)

        with ctx.step("assign_folds"):
            study_frame, report = build_study_frame(
                train,
                series_df,
                study_meta,
                group_by=cfg.folds.group_by,
                min_patient_id_coverage=cfg.folds.min_patient_id_coverage,
                series_count_buckets=list(cfg.folds.series_count_buckets),
            )
            folds = assign_folds(study_frame, train, n_folds=cfg.n_folds, seed=cfg.seed,
                                 report=report)

        write_parquet(folds, out_path)
        digest = file_hash(out_path)

        for line in report.to_lines():
            ctx.log.info("%s", line)
        ctx.log.info("wrote %s  sha256=%s", out_path, digest)

        prevalence = fold_prevalence_table(folds, train).round(4)
        ctx.log.info("per-label pos_rate by fold (GT subset):\n%s", prevalence.to_string())

        ctx.log_event("folds_written", path=str(out_path), sha256=digest,
                      group_key=report.group_key, n_studies=report.n_studies,
                      n_groups=report.n_groups, fold_sizes=report.fold_sizes,
                      proxy_available=report.proxy_available,
                      prevalence=prevalence.to_dict())
        ctx.log.info("FROZEN. Commit artifacts/folds.parquet.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
