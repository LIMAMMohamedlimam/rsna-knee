#!/usr/bin/env python3
"""Prepare a Colab (or any bare) session: mount Drive, install deps, verify the environment.

    python scripts/colab_bootstrap.py            # mount + install + check
    python scripts/colab_bootstrap.py --check    # check only, install nothing

This is the one file allowed to use `print` (CLAUDE.md §3.11): it runs *before* the
dependencies that `src/utils/logging.py` needs are installed, so it cannot import them.
It also must not import anything from `src/` beyond the standard library for the same reason.
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# Distribution name -> module name, where they differ.
IMPORT_NAMES = {
    "pyyaml": "yaml",
    "scikit-learn": "sklearn",
    "iterative-stratification": "iterstrat",
    "pylibjpeg-libjpeg": "libjpeg",
    "pylibjpeg-openjpeg": "openjpeg",
}
REQUIRED_DATA_FILES = ("train.csv", "train_series.csv")
DRIVE_MOUNT = Path("/content/drive")


def print_header(title: str) -> None:
    print(f"\n{'=' * 68}\n{title}\n{'=' * 68}")


def has_module(name: str) -> bool:
    """find_spec raises (not returns None) when a parent package is missing, and can raise
    ValueError on partially-initialised packages — neither should abort the bootstrap."""
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def in_colab() -> bool:
    return has_module("google.colab")


def dist_to_module(name: str) -> str:
    return IMPORT_NAMES.get(name, name.replace("-", "_"))


def parse_requirements(pyproject: Path, extras: list[str] | None = None) -> list[str]:
    """Core dependencies from pyproject.toml, plus any requested extras.

    Read from the file rather than duplicated here, so the Colab path can never drift from
    the pinned set the rest of the project uses.
    """
    with open(pyproject, "rb") as fh:
        data = tomllib.load(fh)
    project = data.get("project", {})
    requirements = list(project.get("dependencies", []))
    optional = project.get("optional-dependencies", {})
    for extra in extras or []:
        requirements += list(optional.get(extra, []))
    return requirements


def requirement_name(requirement: str) -> str:
    """'numpy>=1.26,<3' -> 'numpy'."""
    for separator in (">=", "<=", "==", "!=", "~=", ">", "<", "[", ";", " "):
        requirement = requirement.split(separator)[0]
    return requirement.strip()


def missing_requirements(requirements: list[str]) -> list[str]:
    """Only what is genuinely absent — reinstalling satisfied pins risks breaking Colab's
    preinstalled numpy/torch build."""
    missing = []
    for requirement in requirements:
        if not has_module(dist_to_module(requirement_name(requirement))):
            missing.append(requirement)
    return missing


def install(requirements: list[str]) -> int:
    if not requirements:
        print("all dependencies already present")
        return 0
    print(f"installing {len(requirements)} package(s): {', '.join(requirements)}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", *requirements],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"pip failed:\n{result.stdout}\n{result.stderr}")
    else:
        print("install ok")
    return result.returncode


def mount_drive() -> Path | None:
    """Mount Google Drive if we are on Colab and it is not mounted yet."""
    if not in_colab():
        print("not running on Colab — skipping Drive mount")
        return None
    if (DRIVE_MOUNT / "MyDrive").exists():
        print(f"Drive already mounted at {DRIVE_MOUNT}")
        return DRIVE_MOUNT
    from google.colab import drive  # noqa: PLC0415  (only importable on Colab)

    drive.mount(str(DRIVE_MOUNT))
    return DRIVE_MOUNT


def find_data_dirs(root: Path, max_depth: int = 3) -> list[Path]:
    """Directories at or under `root` holding train.csv — bounded-depth globs only, never a
    full walk. Duplicated from src/utils/io.py on purpose: this script must stay importable
    before the project's dependencies exist."""
    if not root.exists():
        return []
    found = [root] if (root / "train.csv").exists() else []
    for depth in range(1, max_depth + 1):
        found += [p.parent for p in root.glob("/".join(["*"] * depth) + "/train.csv")]
    return sorted(set(found))


def check_data_dir(raw_dir: Path) -> list[str]:
    """Validate RSNA_RAW without walking the whole tree (that would be one Drive round trip
    per file). Checks only the two CSVs and that train_series/ exists."""
    problems = []
    if not raw_dir.exists():
        return [f"RSNA_RAW does not exist: {raw_dir}"]

    for name in REQUIRED_DATA_FILES:
        if not (raw_dir / name).exists():
            problems.append(f"missing {name} in {raw_dir}")

    if problems:
        # Pointing one level too high is the usual cause (e.g. /kaggle/input/competitions).
        candidates = [str(c) for c in find_data_dirs(raw_dir) if c != raw_dir]
        problems.append(f"set RSNA_RAW to one of: {candidates}" if candidates else
                        f"no train.csv anywhere under {raw_dir} — is the dataset attached?")
        return problems

    if not (raw_dir / "train_series").exists():
        problems.append(f"missing train_series/ in {raw_dir} "
                        f"(fine for Spec 01/02 label work; required from Spec 03)")
    return problems


def describe_environment() -> dict[str, str]:
    info = {
        "python": sys.version.split()[0],
        "colab": str(in_colab()),
        "cwd": os.getcwd(),
        "repo": str(REPO_ROOT),
        "RSNA_RAW": os.environ.get("RSNA_RAW", "<unset>"),
        "RSNA_ARTIFACTS": os.environ.get("RSNA_ARTIFACTS", "<unset — defaults to ./artifacts>"),
    }
    usage = shutil.disk_usage(REPO_ROOT)
    info["disk_free"] = f"{usage.free / 1e9:.0f} GB free of {usage.total / 1e9:.0f} GB"
    try:
        import torch

        info["torch"] = torch.__version__
        info["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        info["torch"] = "not installed"
    return info


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify only, install nothing")
    parser.add_argument("--no-mount", action="store_true", help="skip the Drive mount")
    parser.add_argument("--extras", nargs="*", default=[], help="extras to add, e.g. dicom train")
    args = parser.parse_args()

    print_header("1. Google Drive")
    if not args.no_mount:
        mount_drive()
    else:
        print("skipped (--no-mount)")

    print_header("2. Dependencies")
    requirements = parse_requirements(REPO_ROOT / "pyproject.toml", args.extras)
    missing = missing_requirements(requirements)
    if args.check:
        print(f"missing: {missing or 'none'}")
    elif install(missing) != 0:
        return 1

    print_header("3. Environment")
    for key, value in describe_environment().items():
        print(f"{key:16s}: {value}")

    print_header("4. Data")
    raw = os.environ.get("RSNA_RAW")
    if not raw:
        print("RSNA_RAW is unset. In a Colab cell:\n"
              "    %env RSNA_RAW=/content/drive/MyDrive/rsna-data\n"
              "(`!export` does NOT work — it dies with its subshell.)")
        problems = ["RSNA_RAW unset"]
    else:
        problems = check_data_dir(Path(raw))
        print("\n".join(f"  ✗ {p}" for p in problems) if problems else f"  ✓ {raw} looks usable")

    print_header("Next steps")
    if problems:
        print("Fix the data problems above, then rerun this script.")
    else:
        print("make eda   PY=python     # header sweep + report (resumable)")
        print("make folds PY=python     # frozen CV split — run AFTER make eda")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
