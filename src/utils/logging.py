"""Run logging (Task 1.0 — preliminary, used by every entrypoint from Spec 01 onward).

Design goals, driven by what actually goes wrong in this project:

* **Every artifact is traceable.** A run writes `artifacts/logs/{script}_{run_id}.log` whose
  first block is a manifest: config hash, git SHA, seed, argv, cwd, package versions, host.
  Months later, "which code and config produced this parquet?" is answerable.
* **Long jobs leave a trail.** Spec 02 extracts labels over every report, Spec 03 builds a
  570 GB cache, Spec 05 runs 1300 studies inside a 9h budget. `ProgressLogger` emits rate and
  ETA on a time-based cadence, so a job that dies at hour 6 shows where it was.
* **Failures are recorded, not swallowed.** `RunContext` is a context manager: an exception
  is logged with its traceback and the run is closed with `status=failed` before it re-raises.
  Spec 05 §5.2 must contain per-study exceptions and continue — `log_exception` does that
  without losing the stack.
* **Machine-readable events.** `log_event` appends JSONL next to the text log, so timings and
  metrics can be diffed across runs without parsing prose.
* **Never breaks the run.** An unwritable log dir (read-only Kaggle image, notebook sandbox)
  degrades to console-only with a warning.

Console output goes to **stderr**, so stdout stays clean for data a caller might pipe.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import socket
import sys
import time
import traceback
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

ROOT_LOGGER = "rsna"
# `logging.captureWarnings` emits under "py.warnings", outside our tree — attach the same
# handlers there so a DeprecationWarning during a 6h job lands in the run log.
_MANAGED_LOGGERS = (ROOT_LOGGER, "py.warnings")
CONSOLE_FORMAT = "%(asctime)s | %(levelname)-7s | %(message)s"
CONSOLE_DATEFMT = "%H:%M:%S"
FILE_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-28s | %(filename)s:%(lineno)d | %(message)s"
LEVEL_ENV_VAR = "RSNA_LOG_LEVEL"

# Reported in the manifest when installed; absence is itself useful debugging information.
TRACKED_PACKAGES = ("numpy", "pandas", "pyarrow", "scikit-learn", "pydicom", "torch", "timm")


class StderrHandler(logging.StreamHandler):
    """Console handler that resolves `sys.stderr` at emit time, not at construction.

    A handler holding the original stream object keeps writing there after a redirect —
    which silently loses console output inside a Kaggle notebook, under
    `contextlib.redirect_stderr`, or in a captured test.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def stream(self):  # type: ignore[override]
        return sys.stderr

    @stream.setter
    def stream(self, value) -> None:
        pass  # StreamHandler.__init__ assigns here; the property is the source of truth

    def close(self) -> None:
        logging.Handler.close(self)  # never close the real stderr


def get_logger(name: str) -> logging.Logger:
    """Project logger. Modules call `get_logger(__name__)`; handlers live on the root."""
    suffix = name.split(".", 1)[-1] if name.startswith("src.") else name
    return logging.getLogger(f"{ROOT_LOGGER}.{suffix}")


def _resolve_level(value: str | int | None, default: int) -> int:
    """Env var wins, so `RSNA_LOG_LEVEL=DEBUG make eda` needs no config edit."""
    override = os.environ.get(LEVEL_ENV_VAR)
    candidate = override if override else value
    if candidate is None:
        return default
    if isinstance(candidate, int):
        return candidate
    return logging.getLevelNamesMapping().get(str(candidate).upper(), default)


def make_run_id(git_sha: str = "nogit") -> str:
    """Sortable, unique-per-second run id: `20260809T154210Z_ab12cd`."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stamp}_{git_sha}"


@dataclass
class RunContext:
    """Handle on one run. Use as a context manager so failures are always recorded."""

    run_id: str
    script: str
    log_path: Path | None
    events_path: Path | None
    manifest: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)
    _started: float = field(default_factory=time.perf_counter)
    _closed: bool = False

    @property
    def log(self) -> logging.Logger:
        return logging.getLogger(f"{ROOT_LOGGER}.{self.script}")

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self._started

    def log_event(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one structured record to the JSONL sidecar (and DEBUG it to the text log)."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "script": self.script,
            "event": event,
            "elapsed_s": round(self.elapsed_s, 3),
            **fields,
        }
        if self.events_path is not None:
            try:
                with open(self.events_path, "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record, default=str) + "\n")
            except OSError as exc:  # a full or read-only disk must not kill the run
                self.log.warning("could not append event %r: %s", event, exc)
        self.log.debug("event %s %s", event, {k: v for k, v in fields.items()})
        return record

    @contextmanager
    def step(self, name: str, **fields: Any) -> Iterator[None]:
        """Time a block, log start/end, and record the duration for the Spec 05 budget table.

            with ctx.step("decode", study=uid):
                ...
        """
        self.log.info("▸ %s …", name)
        start = time.perf_counter()
        try:
            yield
        except Exception as exc:
            duration = time.perf_counter() - start
            self.timings[name] = self.timings.get(name, 0.0) + duration
            self.log.error("✗ %s failed after %.2fs: %s", name, duration, exc)
            self.log_event("step_failed", step=name, duration_s=round(duration, 3),
                           error=f"{type(exc).__name__}: {exc}", **fields)
            raise
        duration = time.perf_counter() - start
        self.timings[name] = self.timings.get(name, 0.0) + duration
        self.log.info("✓ %s (%.2fs)", name, duration)
        self.log_event("step", step=name, duration_s=round(duration, 3), **fields)

    def log_exception(self, message: str, exc: BaseException, **fields: Any) -> None:
        """Record a contained failure with its traceback, then let the caller continue.

        This is the Spec 05 §5.2 path: one bad study must not end a 1300-study notebook, but
        it must be reconstructible afterwards.
        """
        self.log.error("%s: %s: %s", message, type(exc).__name__, exc, exc_info=exc)
        self.log_event("exception", message=message, error_type=type(exc).__name__,
                       error=str(exc), traceback="".join(
                           traceback.format_exception(type(exc), exc, exc.__traceback__)), **fields)

    def close(self, status: str = "ok", **fields: Any) -> None:
        if self._closed:
            return
        self._closed = True
        self.log_event("run_end", status=status, duration_s=round(self.elapsed_s, 3),
                       timings={k: round(v, 3) for k, v in self.timings.items()}, **fields)
        self.log.info("run %s finished: status=%s in %.1fs", self.run_id, status, self.elapsed_s)
        if self.log_path is not None:
            self.log.info("log: %s", self.log_path)

    def __enter__(self) -> RunContext:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if exc is None:
            self.close("ok")
        elif isinstance(exc, SystemExit):
            self.close("ok" if not exc.code else "exit", exit_code=exc.code)
        else:
            self.log.critical("uncaught %s: %s", exc_type.__name__, exc, exc_info=(exc_type, exc, tb))
            self.close("failed", error_type=exc_type.__name__, error=str(exc))
        return False  # never swallow


def _package_versions() -> dict[str, str]:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in TRACKED_PACKAGES:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = "not installed"
    return out


def build_manifest(script: str, run_id: str, cfg: Any | None) -> dict[str, Any]:
    """Everything needed to reproduce a run, gathered once at startup."""
    from src.utils.io import git_sha

    manifest: dict[str, Any] = {
        "run_id": run_id,
        "script": script,
        "git_sha": git_sha(),
        "argv": sys.argv,
        "cwd": os.getcwd(),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
        "env": {k: os.environ.get(k) for k in ("RSNA_RAW", LEVEL_ENV_VAR, "CUDA_VISIBLE_DEVICES")},
    }
    if cfg is not None:
        from src.utils.config import config_hash

        manifest["config_hash"] = config_hash(cfg)
        manifest["seed"] = int(cfg.get("seed", -1))
    return manifest


def _prune_old_runs(log_dir: Path, keep: int) -> None:
    """Keep the newest `keep` runs so artifacts/logs cannot grow without bound."""
    if keep <= 0:
        return
    logs = sorted(log_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for stale in logs[keep:]:
        stale.unlink(missing_ok=True)
        stale.with_suffix(".jsonl").unlink(missing_ok=True)


def reset_logging() -> None:
    """Drop project handlers. Called by setup and by tests to stay isolated."""
    closed: set[int] = set()
    for name in _MANAGED_LOGGERS:
        logger = logging.getLogger(name)
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            if id(handler) not in closed:  # handlers are shared across the managed loggers
                closed.add(id(handler))
                handler.close()


def setup_logging(
    cfg: Any | None = None,
    script: str = "run",
    *,
    console_level: str | int | None = None,
    log_dir: str | Path | None = None,
) -> RunContext:
    """Install console + file + JSONL logging and open a run. Idempotent per process."""
    from src.utils.io import git_sha, resolve

    log_cfg = (cfg.get("logging", {}) if cfg is not None else {}) or {}
    run_id = make_run_id(git_sha())

    reset_logging()

    console = StderrHandler()
    console.setLevel(_resolve_level(console_level or log_cfg.get("console_level"), logging.INFO))
    console.setFormatter(logging.Formatter(CONSOLE_FORMAT, datefmt=CONSOLE_DATEFMT))
    handlers: list[logging.Handler] = [console]

    directory = Path(log_dir) if log_dir is not None else resolve(log_cfg.get("dir", "artifacts/logs"))
    log_path: Path | None = None
    events_path: Path | None = None
    setup_error: str | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        log_path = directory / f"{script}_{run_id}.log"
        events_path = log_path.with_suffix(".jsonl")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setLevel(_resolve_level(log_cfg.get("file_level"), logging.DEBUG))
        file_handler.setFormatter(logging.Formatter(FILE_FORMAT))
        handlers.append(file_handler)
        _prune_old_runs(directory, int(log_cfg.get("keep_runs", 50)))
    except OSError as exc:
        # Read-only checkout or Kaggle sandbox: console-only is still a usable run.
        log_path = events_path = None
        setup_error = f"{type(exc).__name__}: {exc}"

    logging.captureWarnings(True)  # route warnings.warn into the log file
    for name in _MANAGED_LOGGERS:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = False  # never double-print through the stdlib root logger
        for handler in handlers:
            logger.addHandler(handler)

    ctx = RunContext(run_id=run_id, script=script, log_path=log_path, events_path=events_path)
    ctx.manifest = build_manifest(script, run_id, cfg)
    if setup_error:
        ctx.log.warning("file logging disabled (%s) — console only", setup_error)

    ctx.log.info("run %s | script=%s | git=%s | config=%s | seed=%s", run_id, script,
                 ctx.manifest["git_sha"], ctx.manifest.get("config_hash", "-"),
                 ctx.manifest.get("seed", "-"))
    ctx.log.debug("manifest: %s", json.dumps(ctx.manifest, indent=2, default=str))
    ctx.log_event("run_start", **ctx.manifest)
    return ctx


def install_excepthook(ctx: RunContext) -> None:
    """Log uncaught exceptions for code paths that cannot use `with ctx:` (e.g. notebooks)."""
    previous = sys.excepthook

    def hook(exc_type, exc, tb):
        if not issubclass(exc_type, KeyboardInterrupt):
            ctx.log.critical("uncaught %s: %s", exc_type.__name__, exc, exc_info=(exc_type, exc, tb))
            ctx.close("failed", error_type=exc_type.__name__, error=str(exc))
        previous(exc_type, exc, tb)

    sys.excepthook = hook


class ProgressLogger:
    """Heartbeat for long loops: rate + ETA, on a time cadence rather than every iteration.

        progress = ProgressLogger(len(studies), "cache", ctx)
        for study in studies:
            ...
            progress.update()
        progress.finish()

    Time-based (not count-based) so a slow loop still reports and a fast one stays quiet.
    """

    def __init__(self, total: int, name: str, ctx: RunContext | None = None,
                 every_s: float = 30.0, logger: logging.Logger | None = None):
        self.total = max(int(total), 0)
        self.name = name
        self.ctx = ctx
        self.every_s = every_s
        self.log = logger or (ctx.log if ctx is not None else get_logger(__name__))
        self.n = 0
        self.n_errors = 0
        self._start = time.perf_counter()
        self._last = self._start

    def update(self, n: int = 1, errors: int = 0) -> None:
        self.n += n
        self.n_errors += errors
        now = time.perf_counter()
        if now - self._last >= self.every_s:
            self._last = now
            self._emit("progress")

    def _emit(self, event: str) -> None:
        elapsed = time.perf_counter() - self._start
        rate = self.n / elapsed if elapsed > 0 else 0.0
        remaining = (self.total - self.n) / rate if rate > 0 and self.total else float("nan")
        self.log.info("%s %d/%d (%.1f%%) | %.1f it/s | elapsed %.0fs | eta %.0fs | errors %d",
                      self.name, self.n, self.total,
                      100 * self.n / self.total if self.total else 0.0,
                      rate, elapsed, remaining, self.n_errors)
        if self.ctx is not None:
            self.ctx.log_event(event, name=self.name, n=self.n, total=self.total,
                               rate_per_s=round(rate, 3), elapsed_s=round(elapsed, 1),
                               eta_s=None if remaining != remaining else round(remaining, 1),
                               errors=self.n_errors)

    def finish(self) -> None:
        self._emit("progress_done")


def log_dataframe(logger: logging.Logger, name: str, df: Any, max_columns: int = 12) -> None:
    """Shape, dtypes and null counts — the first thing you want when a table looks wrong."""
    try:
        nulls = df.isna().sum()
        columns = list(df.columns)[:max_columns]
        logger.debug("%s: shape=%s columns=%s%s", name, df.shape, columns,
                     " …" if len(df.columns) > max_columns else "")
        logger.debug("%s: nulls=%s", name, {c: int(nulls[c]) for c in columns if nulls[c]})
    except Exception as exc:  # diagnostics must never be the thing that breaks a run
        logger.debug("%s: could not summarise (%s)", name, exc)
