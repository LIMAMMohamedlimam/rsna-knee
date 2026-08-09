"""Path, hashing and parquet/csv helpers shared by every entrypoint."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    """Resolve a config path: absolute stays put, relative is anchored at the repo root.

    Keeps `cache_dir: artifacts/cache` working regardless of the caller's cwd, and keeps us
    off hardcoded /mnt/ or /kaggle paths (CLAUDE.md §6).
    """
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p)


def file_hash(path: str | Path, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_sha(short: bool = True) -> str:
    """Current git SHA, or 'nogit' outside a repo. Logged with every run (CLAUDE.md §3.1)."""
    cmd = ["git", "rev-parse", "--short" if short else "--verify", "HEAD"]
    try:
        out = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "nogit"
    return out.stdout.strip() if out.returncode == 0 else "nogit"


def write_parquet(df: pd.DataFrame, path: str | Path, **kwargs) -> Path:
    """Write parquet, creating parent dirs. Index is never written (schemas are explicit)."""
    p = resolve(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(p, index=False, **kwargs)
    return p


def read_parquet(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_parquet(resolve(path), **kwargs)


def read_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    return pd.read_csv(resolve(path), **kwargs)
