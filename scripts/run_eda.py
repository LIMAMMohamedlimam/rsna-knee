#!/usr/bin/env python3
"""Spec 01 Task 1.2 — regenerate docs/eda_report.md + docs/figures/.

    make eda
    python scripts/run_eda.py --override eda.dicom_sample_series=500
    RSNA_LOG_LEVEL=DEBUG make eda        # verbose console
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eda.pipeline import run_eda, write_outputs  # noqa: E402
from src.utils.config import load_config  # noqa: E402
from src.utils.logging import setup_logging  # noqa: E402
from src.utils.seed import set_seed  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/base.yaml")
    parser.add_argument("--exp", default=None, help="optional experiment config merged on top")
    parser.add_argument("--override", nargs="*", default=None, help="dotlist, e.g. seed=7")
    parser.add_argument("--log-level", default=None, help="console level (default INFO)")
    args = parser.parse_args()

    cfg = load_config(exp=args.exp, base=args.config, overrides=args.override)

    with setup_logging(cfg, "run_eda", console_level=args.log_level) as ctx:
        set_seed(cfg.seed)

        with ctx.step("run_eda"):
            result = run_eda(cfg, ctx=ctx)
        with ctx.step("write_outputs"):
            outputs = write_outputs(result, cfg)

        ctx.log.info("studies=%d labeled=%d sampled_series=%d",
                     result.scalars["n_studies"], result.scalars["n_labeled"],
                     result.scalars["n_sampled_series"])
        for name, path in outputs.items():
            ctx.log.info("wrote %s: %s", name, path)
        for risk in result.risks:
            ctx.log.warning("risk: %s", risk.replace("\n", " "))

        ctx.log_event("eda_summary", **{k: v for k, v in result.scalars.items()},
                      n_risks=len(result.risks),
                      outputs={k: str(v) for k, v in outputs.items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
