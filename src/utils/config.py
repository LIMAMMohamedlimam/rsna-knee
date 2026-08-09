"""Config loading. Every run is driven by YAML + logged config hash (CLAUDE.md §3.1)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.utils.constants import LABELS
from src.utils.io import resolve

BASE_CONFIG = "configs/base.yaml"


def load_config(
    exp: str | Path | None = None,
    base: str | Path = BASE_CONFIG,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Load base.yaml, merge an experiment config over it, then dotlist overrides.

    Validates that cfg.labels matches the canonical order — a silent reorder here would
    scramble every OOF file and the submission.
    """
    cfg = OmegaConf.load(resolve(base))
    if exp is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(resolve(exp)))
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))

    if list(cfg.labels) != LABELS:
        raise ValueError(
            "cfg.labels does not match src.utils.constants.LABELS (canonical order).\n"
            f"  config: {list(cfg.labels)}\n  canonical: {LABELS}"
        )
    return cfg


def config_hash(cfg: DictConfig, length: int = 12) -> str:
    """Stable hash of the fully-resolved config, logged with every run."""
    payload = OmegaConf.to_yaml(cfg, resolve=True, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:length]


def cfg_path(cfg: DictConfig, key: str) -> Path:
    """Resolve `paths.<key>` to an absolute Path."""
    value: Any = cfg.paths[key]
    return resolve(str(value))
