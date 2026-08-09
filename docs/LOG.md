# LOG

Append-only. One 5-line entry per task: date, what, config, result, next (CLAUDE.md §5).

---

**2026-08-09 — Fix: multi-valued DICOM tags broke the parquet checkpoint**
- What: `SeriesMeta.to_row()` now flattens every value to a single scalar type — `_as_text` joins VM>1 tags on `\` (DICOM's own separator), `_as_float_pair` copes with PixelSpacing arriving as a pair, a scalar or absent, and all fingerprint columns are text (`MagneticFieldStrength` is numeric on most scanners and multi-valued on some).
- Trigger: first real sweep died at the 500-study checkpoint — `ArrowInvalid: cannot mix list and non-list values` on `SoftwareVersions`. Progress up to the previous successful flush is preserved by design; here it failed on the first one, so nothing was lost but time.
- Result: 8 new tests, including a regression that writes rows from scanners with VM>1, VM=1 and an absent tag into one parquet, plus an invariant test that no `to_row()` value is ever a list.
- Next: watch for the same class of surprise when Spec 03 reads pixel-level tags.

**2026-08-09 — Fix: Kaggle was detected as Colab**
- What: `in_colab()` no longer decides on module importability alone — the Kaggle image ships an importable `google.colab` shim. Detection is now `in_kaggle()` (KAGGLE_* env) first, then Colab env markers (COLAB_RELEASE_TAG / COLAB_GPU), and the module probe only as a last resort. `mount_drive` says so explicitly on Kaggle.
- Trigger: `test_in_colab_is_false_here` failed on Kaggle. The test was wrong to assert an ambient environment fact, but it exposed a real latent bug: without `--no-mount`, the bootstrap would have hunted for a Drive that does not exist there.
- Result: that test replaced by 4 that monkeypatch the environment and check the decision logic, including Kaggle winning over Colab markers. 133 passing.
- Next: none.

**2026-08-09 — Notebooks: ready-to-run Kaggle and Colab entry points**
- What: `notebooks/kaggle_spec01.ipynb` (runs all of Spec 01 where the data is already local — auto-discovers `RSNA_RAW` by globbing for train.csv, so the container-directory mistake cannot recur; packages the outputs into one downloadable zip) and a corrected `notebooks/colab_setup.ipynb`.
- Result: 6 tests in `tests/test_notebooks.py` — valid JSON, code cells parse as Python (magics/shell escapes stripped), no `!export` (a silent no-op in notebooks), env set before the bootstrap so its validation actually runs, and entrypoints in the right order.
- Next: Spec 02 needs the model/endpoint/budget decisions before any code can run.

**2026-08-09 — Fix: unhelpful failure when RSNA_RAW points one level too high**
- What: `load_raw` moved to `src/utils/io.py` (so `make_folds` shares it) and now raises `RawDataNotFound` naming the directory that *does* contain train.csv, instead of a bare pandas `FileNotFoundError`. `colab_bootstrap.check_data_dir` makes the same suggestion, and short-circuits the `train_series/` complaint when the CSVs are missing. The EDA report header now records the resolved `RSNA_RAW`.
- Trigger: a first Kaggle run with `RSNA_RAW=/kaggle/input/competitions` — a container directory, not the competition folder.
- Result: 11 new tests in `tests/test_io.py` + 3 in `tests/test_colab_bootstrap.py`, including the exact `<root>/competitions/<slug>` layout. Verified by reproducing the failure locally.
- Next: none.
- Also removed `docs/eda_report.md` and 4 figures from git — they had been generated from the synthetic fixture during development and committed by mistake. `artifacts/folds.parquet` was correctly never committed.

**2026-08-09 — Task 1.0b (added): Colab + Google Drive support**
- What: `PY`/`PYTEST` indirection in the Makefile (`make eda PY=python` works without uv); `paths.artifacts_root` driven by `RSNA_ARTIFACTS` so every output can move to Drive or to fast local disk without editing configs; `scripts/colab_bootstrap.py` (mounts Drive, installs the pins read straight from `pyproject.toml`, validates `RSNA_RAW` in O(1) Drive round trips); `notebooks/colab_setup.ipynb`; and a **resumable header sweep** in `src/eda/pipeline.py` (`eda.patient_sweep`) that reads ONE header per study to recover `PatientID` for every study.
- Config: `paths.artifacts_root`, `paths.study_headers_path`, `eda.patient_sweep=true`, `eda.sweep_flush_every=500`.
- Result: 24 new tests. Verified end-to-end on 230 synthetic studies — the sweep lifts PatientID coverage to 100%, so `make folds` reports `group_key: PatientID` instead of falling back to study level; interrupting the sweep at 74/230 and rerunning resumes at 74 with no re-reads; `RSNA_ARTIFACTS` redirects logs, parquets and folds out of the repo.
- Next: Spec 03 must not read raw DICOM off Drive FUSE (millions of small files ≫ a 12h Colab session) — build the cache on Kaggle, or in resumable batches, then train off local disk.
- Why: without the sweep, only the ~200 sampled studies have a PatientID, coverage falls under the 95% threshold, and folds silently group by study — letting one patient span two folds and inflating CV for the whole competition.

**2026-08-09 — Task 1.0 (added, preliminary): run logging**
- What: `src/utils/logging.py` — `setup_logging()` opens a run and writes `artifacts/logs/{script}_{run_id}.log` + a `.jsonl` event sidecar, headed by a manifest (config hash, git SHA, seed, argv, cwd, host, package versions). `RunContext` is a context manager (failures logged with traceback, then re-raised), `ctx.step()` times phases, `ctx.log_exception()` records contained per-item failures, `ProgressLogger` emits rate/ETA on a time cadence, `log_dataframe()` dumps shape/dtypes/nulls. Console → stderr, so stdout stays clean. Both Spec 01 entrypoints rewired off `print`. New rule **CLAUDE.md §3.11** makes this binding for later phases.
- Config: `logging.{dir,console_level,file_level,keep_runs}` in `base.yaml`; `RSNA_LOG_LEVEL=DEBUG` overrides the console level without editing config.
- Result: 23 tests in `tests/test_logging.py` green — manifest completeness, JSONL validity, failure capture, idempotent setup, log pruning, unwritable-dir degradation, and both entrypoints verified end-to-end via subprocess.
- Next: use `ProgressLogger` in the Spec 02 extraction loop and the Spec 03 cache builder; feed `ctx.timings` into the Spec 05 §5.5 budget table.
- Why now: Spec 02 runs an LLM over every report, Spec 03 builds a 570 GB cache, Spec 05 has a 9h wall clock. Each is a long unattended job where a failure at hour 6 is unreconstructible without a run log.

**2026-08-09 — Spec 01 Task 1.1: repo + config skeleton**
- What: created the CLAUDE.md §2 layout, `configs/base.yaml`, `src/utils/{constants,config,seed,io}.py`, `pyproject.toml` (uv), `Makefile` (`test`/`eda`/`folds`), `.gitignore` that keeps `artifacts/folds.parquet` tracked and everything else out.
- Config: `configs/base.yaml`, hash `2eb9d907…` on the demo run; seed 42; n_folds 5.
- Result: `uv run pytest -q` → **44 passed**. `python -c "from src.utils.constants import LABELS; assert len(LABELS)==12"` green.
- Next: nothing — feeds Task 1.2/1.3.

**2026-08-09 — Spec 01 Task 1.2: EDA script**
- What: `scripts/run_eda.py` (thin) over `src/eda/{labels,reports,series,sites,figures,render,pipeline}.py`. Covers §1.2.1 label census + 12×12 Jaccard/conditional co-occurrence, §1.2.2 language + length + section headers + exact/near duplicate clusters, §1.2.3 series census + **canonical protocol table** + plane coverage + sampled-DICOM header census, §1.2.4 site fingerprints → clusters → confound flags, §1.2.5 auto-filled top-5 risks.
- Config: `eda.dicom_sample_series=200`, `rare_label_pos_rate ∈ [0.05, 0.60]`, `protocol_table_top_n=6`, `site_confound_ratio=2.0`.
- Result: end-to-end on a 690-study synthetic stand-in — report + 4 figures + `artifacts/eda/study_meta.parquet` written; **rerun byte-identical** (no timestamps in the report). 4 risks auto-flagged.
- Next: rerun on real data once `RSNA_RAW` is set; review the canonical protocol table and freeze it as routing constants in Spec 03 Task 3.3.

**2026-08-09 — Spec 01 Task 1.3: frozen folds**
- What: `scripts/make_folds.py` over `src/utils/folds.py`. Two-stage group-level assignment: `MultilabelStratifiedKFold` on the 12 labels for labeled groups, then deterministic greedy balancing of unlabeled groups on the proxy vector (sex, language, site_cluster, series-count bucket). Freeze guard prints the sha256 and exits non-zero.
- Config: `folds.group_by=auto` (PatientID when ≥95% coverage, else StudyInstanceUID), `series_count_buckets=[1,3,5,8]`, seed 42.
- Result: on the synthetic stand-in — fold sizes 137–139, every per-label fold prevalence within ~2% relative of global (tolerance is 25%), proxy marginals within 0.7–1.3× fair share, rerun exits 1. All four spec acceptance tests green in `tests/test_folds.py`.
- Next: **do not run on real data until `make eda` has produced `study_meta.parquet`** — without it, grouping silently falls back to study level and patients can leak across folds.

---

## Deviations from the letter of the specs (all deliberate)

1. **Extra modules beyond the CLAUDE.md §2 tree**: `src/eda/`, `src/utils/config.py`, `src/utils/constants.py`. The §2 tree is not exhaustive — Task 1.1 itself mandates `constants.py`, which the tree omits — and §2 also requires `scripts/` to be *thin* entrypoints, so the EDA logic needs a home that is importable and testable.
2. **`src/data/dicom_reader.py` created in Spec 01** with metadata-only helpers (`read_series_meta`, `list_series_files`). Spec 01 §1.2.3 needs DICOM headers, and §3.3 says raw DICOM is touched by exactly one non-inference module — so the EDA reads through that module rather than opening DICOM itself. Spec 03 extends the same file with `read_series` (pixels, ordering, rescale).
3. **Dependencies**: added `omegaconf` (base.yaml uses `${oc.env:...}` resolvers) and `tqdm`. The torch/timm/albumentations stack is pinned in the `train` extra and the pylibjpeg family in the `dicom` extra, so Spec 01–02 installs without the GPU wheels — `make setup-train` pulls them for Spec 03+. Everything stays pinnable for the offline Kaggle image.
4. **Near-duplicate reports via normalized-text hash**, not MinHash (§1.2.2 permits either). Accent/case/digit-stripped SHA-1 catches templates that differ only in measurements, and adds no dependency.
5. **Site clusters via exact-fingerprint groupby**, not KMeans (§1.2.4 permits either): fingerprints are categorical, so an exact match is deterministic and interpretable. Clusters below `eda.min_cluster_size` pool into `-1`. Studies outside the DICOM sample use the spec's fallback — protocol-signature clustering, ids offset by 1000, recorded in `site_cluster_source`.
6. **Tests run on a synthetic dataset** (`tests/conftest.py:make_synthetic`) reproducing the CLAUDE.md §4 structure: labels on a subset only (NaN elsewhere), multilingual templated reports with planted duplicates, repeat-visit patients, three scanner fingerprints. The real 570 GB data is not present, and CI must stay green without it.

## Known gaps to close on real data

- `eda.dicom_sample_series=200` gives DICOM fingerprints for ~200 studies only; every other study falls back to the protocol-signature proxy. Raise the budget before trusting the site-confound analysis (the report prints the coverage % and self-flags below 50%).
- Whether `PatientID` is inside the 86-tag allowlist is **unverified** — the code handles both outcomes and logs which grouping it used. Check the printed `group_key` on the first real run.
- `SECTION_PATTERNS` in `src/eda/reports.py` covers 7 languages; the report flags any detected language with no pattern as `NO PATTERN` so the list can be extended from real data.
