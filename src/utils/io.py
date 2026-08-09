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


RAW_FILES = ("train.csv", "train_series.csv")


class RawDataNotFound(FileNotFoundError):
    """Raised when RSNA_RAW does not point at the competition data."""


def find_data_dirs(root: Path, max_depth: int = 3) -> list[Path]:
    """Directories at or under `root` that contain train.csv.

    Only used to build a helpful error message. Pointing RSNA_RAW one level too high is the
    easiest mistake to make — on Kaggle the data sits under `/kaggle/input/<slug>` or
    `/kaggle/input/competitions/<slug>` depending on how it was attached.
    """
    if not root.exists():
        return []
    found = [root] if (root / "train.csv").exists() else []
    for depth in range(1, max_depth + 1):
        found += [p.parent for p in root.glob("/".join(["*"] * depth) + "/train.csv")]
    return sorted(set(found))


def load_raw(raw_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read train.csv + train_series.csv, failing with an actionable message if RSNA_RAW is
    wrong rather than a bare FileNotFoundError from deep inside pandas."""
    from src.utils.constants import LABELS

    raw_dir = Path(raw_dir)
    missing_files = [name for name in RAW_FILES if not (raw_dir / name).exists()]
    if missing_files:
        candidates = find_data_dirs(raw_dir)
        hint = (f"\n  did you mean: {[str(c) for c in candidates]}" if candidates else
                f"\n  nothing under {raw_dir} contains train.csv either — check RSNA_RAW")
        listing = sorted(p.name for p in raw_dir.iterdir())[:20] if raw_dir.exists() else []
        raise RawDataNotFound(
            f"RSNA_RAW does not point at the competition data.\n"
            f"  RSNA_RAW : {raw_dir}\n"
            f"  missing  : {missing_files}\n"
            f"  contains : {listing}{hint}"
        )

    train = pd.read_csv(raw_dir / "train.csv")
    series_df = pd.read_csv(raw_dir / "train_series.csv")
    absent = [c for c in LABELS if c not in train.columns]
    if absent:
        raise ValueError(f"train.csv is missing label columns: {absent}")
    return train, series_df
