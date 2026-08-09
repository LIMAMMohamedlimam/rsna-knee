"""The shipped notebooks are hand-written JSON, so their structure is tested.

A malformed cell only shows up when someone opens the notebook on Kaggle — by which point
they have already lost the session. These checks are cheap and catch it here instead.
"""

from __future__ import annotations

import json

import pytest

from src.utils.io import REPO_ROOT

NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def cell_lines(cell) -> list[str]:
    """nbformat allows `source` as either a string or a list of lines; tools emit both."""
    source = cell["source"]
    return source.splitlines() if isinstance(source, str) else [line.rstrip("\n") for line in source]


def code_source(notebook) -> str:
    return "\n".join(
        "\n".join(cell_lines(cell)) for cell in notebook["cells"] if cell["cell_type"] == "code"
    )


def test_notebooks_exist():
    names = {p.name for p in NOTEBOOKS}
    assert {"kaggle_spec01.ipynb", "colab_setup.ipynb"} <= names


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_is_valid_json_with_a_usable_structure(path):
    notebook = load(path)
    assert notebook["nbformat"] == 4
    assert notebook["cells"], "notebook has no cells"

    for i, cell in enumerate(notebook["cells"]):
        assert cell["cell_type"] in {"code", "markdown"}, f"cell {i}"
        assert isinstance(cell["source"], (list, str)), f"cell {i} has an invalid source"
        assert cell_lines(cell), f"cell {i} is empty"
        if cell["cell_type"] == "code":
            assert "outputs" in cell and "execution_count" in cell, f"cell {i}"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebook_code_cells_parse_as_python(path):
    """Catches unbalanced quotes and stray characters from hand-editing the JSON.

    IPython magics (`%env`, `%cd`) and shell escapes (`!cmd`) are not valid Python, so they
    are blanked out before parsing.
    """
    import ast

    for cell in load(path)["cells"]:
        if cell["cell_type"] != "code":
            continue
        lines = ["" if line.lstrip().startswith(("!", "%")) else line
                 for line in cell_lines(cell)]
        ast.parse("\n".join(lines))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_do_not_use_export_for_environment_variables(path):
    """`!export` dies with its subshell — a silent no-op that costs a whole run."""
    assert "!export " not in code_source(load(path))


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_notebooks_set_the_environment_before_running_anything(path):
    """Setting RSNA_RAW after the bootstrap would skip its data validation."""
    source = code_source(load(path))
    if "RSNA_RAW" not in source:
        pytest.skip("notebook does not configure the data path")
    assert source.index("RSNA_RAW") < source.index("colab_bootstrap")


def test_kaggle_notebook_runs_the_spec01_entrypoints_in_order():
    source = code_source(load(REPO_ROOT / "notebooks" / "kaggle_spec01.ipynb"))
    assert source.index("run_eda.py") < source.index("make_folds.py")
    # The data directory is discovered, not typed — that mistake already cost one run.
    assert "glob" in source and "train.csv" in source


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_captured_subprocesses_check_their_exit_code(path):
    """A `git pull` whose stderr and returncode are discarded reports success while leaving
    stale code in place — that cost a full debugging cycle, so it is guarded here."""
    source = code_source(load(path))
    if "capture_output" not in source:
        pytest.skip("notebook runs no captured subprocess")
    assert "returncode" in source, "captured subprocess output without checking returncode"
    assert "stderr" in source, "captured subprocess output without surfacing stderr"
