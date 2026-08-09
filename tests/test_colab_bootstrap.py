"""Contract tests for scripts/colab_bootstrap.py.

The Colab path cannot be exercised here, so what is tested is everything that decides
whether the bootstrap does the right thing once it gets there: requirement parsing stays in
sync with pyproject.toml, probing never raises, and data validation catches a wrong RSNA_RAW
before a long job starts.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest

from src.utils.io import REPO_ROOT

spec = importlib.util.spec_from_file_location(
    "colab_bootstrap", REPO_ROOT / "scripts" / "colab_bootstrap.py"
)
bootstrap = importlib.util.module_from_spec(spec)
sys.modules["colab_bootstrap"] = bootstrap
spec.loader.exec_module(bootstrap)


@pytest.mark.parametrize(
    ("requirement", "expected"),
    [
        ("numpy>=1.26,<3", "numpy"),
        ("scikit-learn>=1.4,<2", "scikit-learn"),
        ("torch", "torch"),
        ("pandas>=2.2,<3 ; python_version>'3.10'", "pandas"),
        ("albumentations[imgaug]>=1.4", "albumentations"),
    ],
)
def test_requirement_name_strips_specifiers(requirement, expected):
    assert bootstrap.requirement_name(requirement) == expected


@pytest.mark.parametrize(
    ("dist", "module"),
    [
        ("pyyaml", "yaml"),
        ("scikit-learn", "sklearn"),
        ("iterative-stratification", "iterstrat"),
        ("numpy", "numpy"),
        ("some-new-package", "some_new_package"),
    ],
)
def test_dist_to_module_mapping(dist, module):
    assert bootstrap.dist_to_module(dist) == module


def test_requirements_are_read_from_pyproject_not_duplicated():
    """A hardcoded list here would drift from the pins the rest of the project uses."""
    requirements = bootstrap.parse_requirements(REPO_ROOT / "pyproject.toml")
    names = {bootstrap.requirement_name(r) for r in requirements}
    assert {"numpy", "pandas", "pydicom", "omegaconf", "langdetect",
            "iterative-stratification"} <= names
    assert "torch" not in names  # heavy stack lives in the `train` extra


def test_parse_requirements_can_add_extras():
    core = bootstrap.parse_requirements(REPO_ROOT / "pyproject.toml")
    with_train = bootstrap.parse_requirements(REPO_ROOT / "pyproject.toml", ["train"])
    assert "torch" in {bootstrap.requirement_name(r) for r in with_train}
    assert len(with_train) > len(core)


def test_every_core_requirement_resolves_to_an_importable_module():
    """Guards the IMPORT_NAMES map: a wrong entry would make the bootstrap reinstall a
    package on every run, or worse, believe a missing one is present."""
    requirements = bootstrap.parse_requirements(REPO_ROOT / "pyproject.toml")
    unresolved = [r for r in requirements
                  if not bootstrap.has_module(bootstrap.dist_to_module(bootstrap.requirement_name(r)))]
    assert unresolved == [], f"IMPORT_NAMES needs an entry for: {unresolved}"


def test_has_module_returns_false_instead_of_raising_on_a_missing_parent():
    # find_spec("google.colab") raises ModuleNotFoundError when `google` itself is absent.
    assert bootstrap.has_module("definitely.not.a.package") is False
    assert bootstrap.has_module("json") is True


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("KAGGLE_KERNEL_RUN_TYPE", "KAGGLE_URL_BASE", "COLAB_RELEASE_TAG", "COLAB_GPU"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_colab_is_detected_from_its_environment_markers(clean_env):
    clean_env.setenv("COLAB_RELEASE_TAG", "release-colab-20260801")
    assert bootstrap.in_colab() is True
    assert bootstrap.in_kaggle() is False


def test_kaggle_is_detected_and_is_never_mistaken_for_colab(clean_env):
    """The Kaggle image ships an importable `google.colab` shim, so a module check alone
    claims Colab there and sends the bootstrap hunting for a Drive that does not exist."""
    clean_env.setenv("KAGGLE_KERNEL_RUN_TYPE", "Interactive")
    clean_env.setattr(bootstrap, "has_module", lambda name: True)

    assert bootstrap.in_kaggle() is True
    assert bootstrap.in_colab() is False


def test_kaggle_env_wins_over_colab_markers(clean_env):
    clean_env.setenv("KAGGLE_URL_BASE", "https://www.kaggle.com")
    clean_env.setenv("COLAB_RELEASE_TAG", "release-colab-20260801")
    assert bootstrap.in_colab() is False


def test_plain_machine_is_neither(clean_env):
    clean_env.setattr(bootstrap, "has_module", lambda name: False)
    assert bootstrap.in_colab() is False
    assert bootstrap.in_kaggle() is False


def test_check_data_dir_accepts_a_valid_layout(tmp_path, synthetic):
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    (raw / "train_series").mkdir()
    assert bootstrap.check_data_dir(raw) == []


def test_check_data_dir_reports_a_missing_directory(tmp_path):
    problems = bootstrap.check_data_dir(tmp_path / "nope")
    assert len(problems) == 1 and "does not exist" in problems[0]


def test_missing_csvs_short_circuit_the_train_series_check(tmp_path, synthetic):
    """A missing CSV means RSNA_RAW is wrong, so also complaining about train_series/ would
    bury the one message that matters."""
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    (raw / "train.csv").unlink()

    problems = bootstrap.check_data_dir(raw)
    assert any("train.csv" in p for p in problems)
    assert not any("train_series/" in p for p in problems)


def test_missing_train_series_is_reported_when_the_csvs_are_present(tmp_path, synthetic):
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)  # no train_series/ directory
    assert any("train_series/" in p for p in bootstrap.check_data_dir(raw))


def test_check_data_dir_does_not_walk_the_tree(tmp_path, synthetic, monkeypatch):
    """Validation must stay O(1) in Drive round trips — a walk over 570 GB would hang."""
    raw = tmp_path / "raw"
    synthetic.write_raw(raw)
    (raw / "train_series").mkdir()

    def explode(*args, **kwargs):
        raise AssertionError("check_data_dir must not enumerate the data tree")

    monkeypatch.setattr("pathlib.Path.rglob", explode)
    monkeypatch.setattr("pathlib.Path.iterdir", explode)
    assert bootstrap.check_data_dir(raw) == []


def test_describe_environment_reports_the_key_facts():
    info = bootstrap.describe_environment()
    for key in ["python", "colab", "repo", "RSNA_RAW", "RSNA_ARTIFACTS", "disk_free"]:
        assert key in info


def test_missing_requirements_flags_only_absent_packages():
    assert bootstrap.missing_requirements(["numpy>=1.26"]) == []
    assert bootstrap.missing_requirements(["a-package-that-does-not-exist>=1"]) == [
        "a-package-that-does-not-exist>=1"
    ]


def test_check_data_dir_suggests_the_right_path_when_set_one_level_too_high(tmp_path, synthetic):
    """The exact Kaggle failure: RSNA_RAW=/kaggle/input/competitions instead of .../<slug>."""
    synthetic.write_raw(tmp_path / "competitions" / "rsna-knee-2026")

    problems = bootstrap.check_data_dir(tmp_path / "competitions")
    suggestion = [p for p in problems if p.startswith("set RSNA_RAW")]
    assert suggestion, problems
    assert str(tmp_path / "competitions" / "rsna-knee-2026") in suggestion[0]


def test_check_data_dir_says_when_nothing_is_attached(tmp_path):
    empty = tmp_path / "input"
    empty.mkdir()
    assert any("is the dataset attached?" in p for p in bootstrap.check_data_dir(empty))
