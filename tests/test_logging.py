"""Contract tests for Task 1.0 — run logging.

Debugging infrastructure that fails silently is worse than none, so the contract is tested:
the manifest is complete, events are valid JSONL, failures are recorded before they re-raise,
and an unwritable log directory degrades instead of crashing the run.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys

import pandas as pd
import pytest

from src.utils.config import load_config
from src.utils.io import REPO_ROOT
from src.utils.logging import (
    ProgressLogger,
    RunContext,
    get_logger,
    log_dataframe,
    reset_logging,
    setup_logging,
)


@pytest.fixture(autouse=True)
def isolate_logging():
    """Tests must not inherit or leak project handlers."""
    reset_logging()
    yield
    reset_logging()


@pytest.fixture
def ctx(tmp_path) -> RunContext:
    cfg = load_config(overrides=[f"logging.dir={tmp_path / 'logs'}"])
    return setup_logging(cfg, "unit_test", log_dir=tmp_path / "logs")


def read_events(ctx: RunContext) -> list[dict]:
    lines = ctx.events_path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


def test_setup_creates_log_and_events_files(ctx):
    assert ctx.log_path.exists() and ctx.log_path.suffix == ".log"
    assert ctx.events_path.exists() and ctx.events_path.suffix == ".jsonl"
    assert ctx.run_id in ctx.log_path.name
    assert "unit_test" in ctx.log_path.name


def test_manifest_records_everything_needed_to_reproduce_a_run(ctx):
    manifest = ctx.manifest
    for key in ["run_id", "script", "git_sha", "argv", "cwd", "host", "python", "platform",
                "packages", "env", "config_hash", "seed"]:
        assert key in manifest, f"manifest is missing {key}"
    assert manifest["seed"] == 42
    assert manifest["packages"]["pandas"] != "not installed"
    # The manifest is the first event, so a truncated log still identifies the run.
    assert read_events(ctx)[0]["event"] == "run_start"


def test_log_event_writes_valid_jsonl(ctx):
    ctx.log_event("custom", n_studies=7, note="hello")
    record = read_events(ctx)[-1]
    assert record["event"] == "custom"
    assert record["n_studies"] == 7
    assert record["run_id"] == ctx.run_id
    assert "elapsed_s" in record and "ts" in record


def test_step_times_the_block_and_records_it(ctx):
    with ctx.step("decode", study="1.2.3"):
        pass
    assert "decode" in ctx.timings
    record = read_events(ctx)[-1]
    assert record["event"] == "step" and record["step"] == "decode"
    assert record["study"] == "1.2.3"
    assert record["duration_s"] >= 0


def test_step_records_a_failure_then_reraises(ctx):
    with pytest.raises(ValueError, match="boom"):
        with ctx.step("bad_step"):
            raise ValueError("boom")

    record = read_events(ctx)[-1]
    assert record["event"] == "step_failed"
    assert "ValueError: boom" in record["error"]
    assert "bad_step" in ctx.timings  # partial time is still attributed


def test_context_manager_logs_uncaught_exceptions_and_does_not_swallow(ctx):
    with pytest.raises(RuntimeError):
        with ctx:
            raise RuntimeError("kaboom")

    end = read_events(ctx)[-1]
    assert end["event"] == "run_end"
    assert end["status"] == "failed"
    assert end["error_type"] == "RuntimeError"
    assert "kaboom" in ctx.log_path.read_text(encoding="utf-8")
    assert "Traceback" in ctx.log_path.read_text(encoding="utf-8")


def test_log_exception_records_traceback_without_raising(ctx):
    """Spec 05 §5.2: a bad study is contained, but must stay reconstructible."""
    try:
        raise KeyError("missing tag")
    except KeyError as exc:
        ctx.log_exception("study failed", exc, study="1.2.3")

    record = read_events(ctx)[-1]
    assert record["event"] == "exception"
    assert record["error_type"] == "KeyError"
    assert record["study"] == "1.2.3"
    assert "KeyError" in record["traceback"]


def test_close_is_idempotent(ctx):
    ctx.close("ok")
    ctx.close("ok")
    assert sum(e["event"] == "run_end" for e in read_events(ctx)) == 1


def test_setup_twice_does_not_duplicate_console_output(tmp_path, capfd):
    cfg = load_config(overrides=[f"logging.dir={tmp_path / 'logs'}"])
    setup_logging(cfg, "first", log_dir=tmp_path / "logs")
    second = setup_logging(cfg, "second", log_dir=tmp_path / "logs")
    capfd.readouterr()

    second.log.info("only once please")
    assert capfd.readouterr().err.count("only once please") == 1


def test_console_goes_to_stderr_leaving_stdout_clean(ctx, capfd):
    ctx.log.info("diagnostic message")
    captured = capfd.readouterr()
    assert "diagnostic message" in captured.err
    assert "diagnostic message" not in captured.out


def test_unwritable_log_dir_degrades_to_console_only(tmp_path):
    blocked = tmp_path / "blocked"
    blocked.write_text("I am a file, not a directory")

    context = setup_logging(load_config(), "degraded", log_dir=blocked / "logs")
    assert context.log_path is None and context.events_path is None
    context.log_event("still works")  # must not raise
    context.close("ok")


def test_debug_level_can_be_raised_from_the_environment(tmp_path, monkeypatch, capfd):
    monkeypatch.setenv("RSNA_LOG_LEVEL", "DEBUG")
    context = setup_logging(load_config(), "verbose", log_dir=tmp_path / "logs")
    capfd.readouterr()
    context.log.debug("visible only at DEBUG")
    assert "visible only at DEBUG" in capfd.readouterr().err


def test_old_run_logs_are_pruned(tmp_path):
    log_dir = tmp_path / "logs"
    cfg = load_config(overrides=["logging.keep_runs=3"])
    for _ in range(6):
        setup_logging(cfg, "spam", log_dir=log_dir).close()
        reset_logging()
    # Pruning runs at startup, so the newest run plus keep_runs-1 survive.
    assert len(list(log_dir.glob("*.log"))) <= 3


def test_progress_logger_reports_rate_and_eta(ctx, capfd):
    progress = ProgressLogger(total=10, name="cache", ctx=ctx, every_s=0.0)
    capfd.readouterr()
    for _ in range(10):
        progress.update()
    progress.finish()

    assert "cache 10/10 (100.0%)" in capfd.readouterr().err
    done = read_events(ctx)[-1]
    assert done["event"] == "progress_done"
    assert done["n"] == 10 and done["total"] == 10
    assert done["rate_per_s"] > 0


def test_progress_logger_counts_errors(ctx):
    progress = ProgressLogger(total=2, name="extract", ctx=ctx, every_s=0.0)
    progress.update(errors=1)
    progress.finish()
    assert read_events(ctx)[-1]["errors"] == 1


def test_progress_logger_handles_zero_total(ctx):
    ProgressLogger(total=0, name="empty", ctx=ctx, every_s=0.0).finish()
    assert read_events(ctx)[-1]["eta_s"] is None


def test_log_dataframe_summarises_shape_and_nulls(ctx):
    df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
    log_dataframe(ctx.log, "frame", df)
    text = ctx.log_path.read_text(encoding="utf-8")
    assert "frame: shape=(3, 2)" in text
    assert "'a': 1" in text


def test_log_dataframe_never_raises_on_a_bad_object(ctx):
    log_dataframe(ctx.log, "not a frame", object())


def test_get_logger_namespaces_under_the_project_root():
    assert get_logger("src.eda.pipeline").name == "rsna.eda.pipeline"
    assert get_logger(__name__).name.startswith("rsna.")


def test_warnings_logger_shares_our_handlers(ctx):
    """`logging.captureWarnings` emits under "py.warnings", outside the rsna tree."""
    warnings_logger = logging.getLogger("py.warnings")
    assert warnings_logger.handlers == logging.getLogger("rsna").handlers
    assert warnings_logger.propagate is False


def test_warnings_really_land_in_the_log_file(tmp_path):
    """End-to-end in a subprocess: pytest's own warning capture would mask this in-process."""
    log_dir = tmp_path / "logs"
    script = (
        "import warnings, sys;"
        f"sys.path.insert(0, {str(REPO_ROOT)!r});"
        "from src.utils.logging import setup_logging;"
        f"ctx = setup_logging(None, 'warn_probe', log_dir={str(log_dir)!r});"
        "warnings.warn('a captured warning', UserWarning);"
        "ctx.close()"
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr

    log_file = next(log_dir.glob("warn_probe_*.log"))
    assert "a captured warning" in log_file.read_text(encoding="utf-8")


# --- entrypoint integration ---------------------------------------------------------------
def test_entrypoints_write_a_run_log(tmp_path, synthetic):
    raw_dir = tmp_path / "raw"
    synthetic.write_raw(raw_dir)
    log_dir = tmp_path / "logs"

    result = subprocess.run(
        [sys.executable, str(REPO_ROOT / "scripts" / "make_folds.py"), "--override",
         f"paths.raw_dir={raw_dir}", f"paths.folds_path={tmp_path / 'folds.parquet'}",
         f"paths.study_meta_path={tmp_path / 'meta.parquet'}", f"logging.dir={log_dir}"],
        capture_output=True, text=True, cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr

    logs = list(log_dir.glob("make_folds_*.log"))
    assert len(logs) == 1
    text = logs[0].read_text(encoding="utf-8")
    assert "FROZEN" in text
    assert "assign_folds" in text

    events = [json.loads(line) for line in
              logs[0].with_suffix(".jsonl").read_text(encoding="utf-8").splitlines()]
    kinds = [e["event"] for e in events]
    assert kinds[0] == "run_start" and kinds[-1] == "run_end"
    written = next(e for e in events if e["event"] == "folds_written")
    assert written["group_key"] == "StudyInstanceUID"  # no study_meta in this run
    assert len(written["sha256"]) == 64


def test_frozen_refusal_is_logged_and_still_exits_non_zero(tmp_path, synthetic):
    raw_dir = tmp_path / "raw"
    synthetic.write_raw(raw_dir)
    log_dir = tmp_path / "logs"
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "make_folds.py"), "--override",
           f"paths.raw_dir={raw_dir}", f"paths.folds_path={tmp_path / 'folds.parquet'}",
           f"paths.study_meta_path={tmp_path / 'meta.parquet'}", f"logging.dir={log_dir}"]

    assert subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT).returncode == 0
    second = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO_ROOT)
    assert second.returncode == 1
    assert "folds are frozen" in second.stderr

    refusal = sorted(log_dir.glob("make_folds_*.jsonl"))[-1].read_text(encoding="utf-8")
    assert "folds_frozen" in refusal
    assert '"status": "refused"' in refusal
