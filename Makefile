.PHONY: setup setup-train test eda folds colab clean-eda help

# Runner indirection: uv locally, plain python where uv is absent (Colab, Kaggle).
#   make eda PY=python
PY     ?= uv run python
PYTEST ?= uv run pytest
UV     ?= uv

help:
	@echo "make setup        install core deps (Spec 01-03)"
	@echo "make setup-train  add the torch/timm stack (Spec 04+)"
	@echo "make test         run the test suite"
	@echo "make eda          regenerate docs/eda_report.md + docs/figures/"
	@echo "make folds        generate artifacts/folds.parquet (refuses if it exists)"
	@echo "make colab        mount Drive, install deps, verify the environment"
	@echo ""
	@echo "On Colab/Kaggle add PY=python PYTEST=pytest, e.g. 'make eda PY=python'."

setup:
	$(UV) sync

setup-train:
	$(UV) sync --extra train --extra dicom

test:
	$(PYTEST) -q

eda:
	$(PY) scripts/run_eda.py

folds:
	$(PY) scripts/make_folds.py

colab:
	$(PY) scripts/colab_bootstrap.py

clean-eda:
	rm -f docs/eda_report.md
	rm -f docs/figures/*.png
