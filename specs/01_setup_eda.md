# Spec 01 — Setup, EDA, Frozen Folds

**Goal:** a reproducible repo skeleton, a quantitative EDA report, and frozen 5-fold CV assignments. Nothing in later phases may start until the exit criteria here are green.

## Task 1.1 — Repo + config skeleton

Create the layout from CLAUDE.md §2. Then:

- `configs/base.yaml` with:
  ```yaml
  paths:
    raw_dir: ${oc.env:RSNA_RAW,/data/rsna}          # contains train.csv etc.
    cache_dir: artifacts/cache
    labels_dir: artifacts/labels
    weights_dir: artifacts/weights
    oof_dir: artifacts/oof
  seed: 42
  n_folds: 5
  labels: [ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture]
  ```
- `src/utils/constants.py` exporting `LABELS` (canonical order) and `N_LABELS = 12`.
- `pyproject.toml` (uv or pip-tools): pin pydicom, pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, numpy, pandas, pyarrow, torch, timm, albumentations, scikit-learn, iterative-stratification, matplotlib, pytest, pyyaml.
- `Makefile` targets: `make test`, `make eda`, `make folds`.

**Acceptance:** `pytest -q` runs (even with 1 trivial test); `python -c "from src.utils.constants import LABELS; assert len(LABELS)==12"`.

## Task 1.2 — EDA script (`scripts/run_eda.py` → `docs/eda_report.md` + `docs/figures/`)

Operate on `train.csv` + `train_series.csv` + a **sample** of ≤200 DICOM series (do not scan all 570 GB). Produce, in this order:

### 1.2.1 Label census (labeled subset only)
- Count of studies with vs without ground-truth labels (`train.csv` label columns non-NaN).
- Per-label prevalence table (n_pos, n_neg, pos_rate) — labeled subset only.
- 12×12 co-occurrence matrix (Jaccard + conditional P(A|B)), saved as heatmap `figures/label_cooccurrence.png`.
- Flag every label with pos_rate < 5% or > 60% in the report under "**Rare/skewed labels**".

### 1.2.2 Report census
- Language detection over ALL reports (`langdetect` or `lingua`; add to deps). Table: language → count → %.
- Report length distribution (chars, tokens ~ chars/4): p5/p50/p95/max. This feeds LLM cost estimation in Spec 02.
- Detect structural sections: regex-count occurrences of common headers per language (English: `FINDINGS|IMPRESSION|CONCLUSION`; French: `RÉSULTAT|CONCLUSION|TECHNIQUE`; add per detected language).
- Exact-duplicate and near-duplicate reports: exact via hash; near via MinHash or normalized-text hash. Report duplicate cluster sizes (template reports are common and matter for leakage).

### 1.2.3 Series census
- Series per study: histogram + p5/p50/p95.
- Cross-tab: Anatomical_Plane × Fluid_Sensitive × Fat_Suppression, with study coverage (% of studies having ≥1 series of that type). Save as `figures/series_crosstab.png`.
- From the 200-series DICOM sample: slice counts, rows×cols distribution, PixelSpacing distribution, TransferSyntaxUID counts, Manufacturer / MagneticFieldStrength value counts (guard missing tags).
- **Canonical protocol table:** the top-6 (plane, fluid, fatsat) combos by study coverage. This becomes the routing input in Spec 03.

### 1.2.4 Site clustering
- Build a per-study "site fingerprint": tuple of (Manufacturer, ManufacturerModelName, MagneticFieldStrength, rounded PixelSpacing, ImplementationVersionName if present) from the sampled DICOMs; for unsampled studies, extend the sample or cluster on series-count patterns as a fallback.
- KMeans or simple groupby on fingerprints → `site_cluster` id per sampled study. Report cluster sizes and per-cluster label prevalence (labeled subset) — flag any cluster whose prevalence deviates >2× from global on any label under "**Site–label confounds**".

### 1.2.5 Risks section
End the report with a "Top 5 risks observed" section (auto-fill with the flags above + free-text placeholders).

**Acceptance:** `make eda` regenerates `docs/eda_report.md` deterministically; report contains all tables above; figures exist.

## Task 1.3 — Frozen folds (`scripts/make_folds.py` → `artifacts/folds.parquet`)

- Universe: ALL training studies (labeled and unlabeled — unlabeled ones get LLM labels later and must already have folds).
- Grouping: if any patient-level tag is available in DICOM metadata (PatientID among the 86 tags — verify), group by patient; else group by StudyInstanceUID.
- Stratification: `MultilabelStratifiedKFold` (iterative-stratification) on the 12 labels where known; for unlabeled studies, stratify on a proxy vector: (PatientSex, report language, site_cluster, series-count bucket). Implement as: two-stage assignment — stratify labeled studies first, then assign unlabeled studies to folds balancing the proxy vector greedily.
- Output schema: `StudyInstanceUID (str), fold (int8), site_cluster (int16), has_gt_labels (bool)`.
- Guard: if `artifacts/folds.parquet` already exists, print its hash and **exit non-zero** with message "folds are frozen; delete manually to regenerate".

**Acceptance tests (`tests/test_folds.py`):**
1. Every study appears exactly once; folds are 0..4.
2. On labeled subset, per-label pos_rate per fold is within ±25% relative of global (looser for labels with <50 positives — assert only that every fold has ≥1 positive).
3. Re-running the script against an existing file exits non-zero.
4. No group (patient) spans two folds.

## Exit criteria for Spec 01
- [ ] `docs/eda_report.md` complete with canonical protocol table and risk flags.
- [ ] `artifacts/folds.parquet` frozen, tests green.
- [ ] `docs/LOG.md` entry written.
