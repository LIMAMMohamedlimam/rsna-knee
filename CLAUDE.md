# CLAUDE.md — RSNA Knee Abnormality Detection (Kaggle 2026)

You are the coding agent for this Kaggle competition project. This file is your permanent contract. Phase-specific specs live in `specs/` — read the relevant spec **in full** before writing any code for that phase.

## 1. Project one-liner

Predict per-study probabilities for 12 knee abnormalities (ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture) from multi-series knee MRI DICOMs. Metric: **macro-averaged AUC-ROC**. Train data has radiology reports (multilingual, only a subset has ground-truth labels); **test data has NO reports**. Inference runs in an offline Kaggle notebook, ≤9h GPU.

## 2. Repository layout (create and maintain exactly this)

```
rsna-knee/
├── CLAUDE.md
├── specs/                      # phase specs (read-only for you)
├── configs/                    # YAML configs; one per experiment
│   ├── base.yaml
│   └── exp/
├── src/
│   ├── data/
│   │   ├── dicom_reader.py     # DICOM → numpy, all transfer syntaxes
│   │   ├── preprocess.py       # series → cached uint8 volumes
│   │   ├── routing.py          # label → series-type routing logic
│   │   ├── dataset.py          # torch Dataset/DataLoader
│   │   └── augment.py          # incl. laterality-aware flip
│   ├── labels/
│   │   ├── extract.py          # LLM report → JSON labels
│   │   ├── prompts.py          # versioned prompt templates
│   │   ├── calibrate.py        # agreement vs ground truth
│   │   └── merge.py            # ensemble labelers → labels_final
│   ├── models/
│   │   ├── encoder2p5d.py      # stage-A slice encoder
│   │   ├── aggregator.py       # stage-B study transformer
│   │   ├── heads.py            # 12-label + auxiliary heads
│   │   └── distill.py          # text-teacher + efficiency distillation
│   ├── train/
│   │   ├── loop.py             # fit/validate, AMP, checkpointing
│   │   ├── losses.py           # weighted BCE, soft-label KD, CLIP aux
│   │   └── metrics.py          # per-label AUC, macro-AUC
│   ├── infer/
│   │   ├── predict.py          # study → 12 probs
│   │   └── kaggle_notebook.py  # generates the submission notebook
│   ├── eda/                    # EDA analysis modules (Spec 01); scripts/ stays thin
│   └── utils/
│       ├── constants.py        # LABELS (canonical order), N_LABELS
│       ├── config.py           # YAML load, merge, config hash
│       ├── folds.py            # frozen CV assignment
│       ├── io.py               # npz/parquet helpers
│       ├── logging.py          # run manifest, JSONL events, step timings
│       └── seed.py
├── scripts/                    # thin CLI entrypoints, one per task
├── tests/                      # pytest; mirrors src/
├── artifacts/                  # gitignored: caches, weights, labels
│   ├── cache/                  # preprocessed volumes
│   ├── labels/                 # labels_v*.parquet
│   ├── weights/
│   └── oof/                    # out-of-fold predictions
└── notebooks/                  # EDA + Kaggle submission notebooks only
```

## 3. Non-negotiable engineering rules

1. **Reproducibility:** every run is driven by a YAML in `configs/exp/`, named `e{NNN}_{slug}.yaml`. Log the config hash, git SHA, and seed. `src/utils/seed.py:set_seed(cfg.seed)` at every entrypoint. Never hardcode hyperparameters in code.
2. **Frozen folds:** `artifacts/folds.parquet` is generated ONCE by `scripts/make_folds.py` and committed (it's small). If it exists, refuse to regenerate — raise instead. All training reads folds from this file.
3. **No raw DICOM in training.** Training/validation only ever read from `artifacts/cache/`. Raw DICOM is touched by exactly two modules: `dicom_reader.py` and the Kaggle inference path.
4. **Determinism of preprocessing:** same input series → byte-identical cache file. No RNG in preprocessing.
5. **Laterality invariant:** any horizontal flip MUST go through `augment.py:laterality_flip(volume, labels)` which swaps label pairs (Medial Meniscus↔Lateral Meniscus, Medial OA↔Lateral OA) and is unit-tested. Never call a generic hflip on knee images.
6. **Label columns order (canonical everywhere, incl. submission):**
   `["ACL","MCL","Medial Meniscus","Lateral Meniscus","Medial OA","Lateral OA","PF OA","Effusion","Synovitis","Baker's","Contusion","Fracture"]`
   Define once in `src/utils/constants.py:LABELS` and import it. Never retype this list.
7. **Metrics:** `metrics.py` computes per-label AUC and macro-AUC on out-of-fold predictions. Every training run writes `artifacts/oof/e{NNN}_fold{k}.parquet` (StudyInstanceUID, 12 probs, 12 targets, fold).
8. **Tests before merge:** every module in `src/data/` and `src/labels/` needs pytest coverage of its contract (see per-spec acceptance tests). Run `pytest -q` before declaring any task done.
9. **Kaggle constraints are production constraints:** no internet at inference, 9h wall clock, weights + wheels shipped as Kaggle Datasets. Any dependency you add must have a pinnable wheel. Prefer: pydicom, pylibjpeg(+libjpeg,+openjpeg), numpy, pandas, pyarrow, torch, timm, albumentations, scikit-learn, iterative-stratification.
10. **Compute discipline:** default to the smallest config that tests the hypothesis (fold 0 only, ConvNeXt-tiny, 224², 10 epochs). Full 5-fold sweeps only after a fold-0 win.
11. **Every run is logged.** No `print` in `src/` or `scripts/` — use `src/utils/logging.py`. Each entrypoint opens `with setup_logging(cfg, "<script>") as ctx:`, which writes `artifacts/logs/{script}_{run_id}.log` plus a `.jsonl` event sidecar, starting with a manifest (config hash, git SHA, seed, argv, package versions). Wrap phases in `ctx.step(name)` so timings are recorded (they feed the Spec 05 budget table), use `ProgressLogger` for any loop over studies/series/reports, and `ctx.log_exception(...)` for contained per-item failures (Spec 05 §5.2). Console goes to stderr; stdout stays clean. The single exception is `scripts/colab_bootstrap.py`, which runs before the dependencies logging needs are installed.

## 4. Data facts you must respect

- `train.csv`: StudyInstanceUID, PatientSex, Report (free text, multilingual), 12 binary labels — **labels present only for a subset**; missing = NaN, not 0.
- `train_series.csv`: StudyInstanceUID, SeriesInstanceUID, Fluid_Sensitive (0/1), Fat_Suppression (0/1), Anatomical_Plane (Sagittal/Coronal/Axial).
- DICOM layout: `train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`, 20–45 slices typical (median 30, tail to hundreds). Transfer syntaxes: Explicit VR LE, Implicit VR LE, JPEG Lossless, JPEG 2000. 86 allowlisted metadata tags only — do not assume any tag exists; guard every access.
- Test: ~1300 studies, same layout, **no Report column**.
- Prevalence differs across train/public/private by organizer statement → model selection is CV-first, never public-LB-first.

## 5. Definition of done (global)

A task is done when: (a) code + tests pass, (b) the spec's acceptance criteria are demonstrated with a command I can rerun, (c) outputs land in the exact paths named by the spec, (d) a 5-line summary is appended to `docs/LOG.md` (date, what, config, result, next).

## 6. Things you must NOT do

- Do not regenerate folds, relabel data, or modify `artifacts/labels/labels_final.parquet` without an explicit instruction.
- Do not add heavy dependencies (mmcv, detectron, monai) without asking — wheel availability on Kaggle offline is the constraint.
- Do not "fix" class imbalance by resampling inside the Dataset silently — sampling strategy lives in the config.
- Do not train on studies whose label confidence weight is 0 for a given head (mask the loss instead of dropping the study).
- Do not write to `/mnt/` paths or assume Kaggle paths locally — all paths come from `configs/base.yaml:paths`.

## 7. Phase order

Work through specs strictly in order unless told otherwise:
`specs/01_setup_eda.md` → `specs/02_label_engine.md` → `specs/03_data_pipeline.md` → `specs/04_modeling.md` → `specs/05_inference_kaggle.md` → `specs/06_ensemble_efficiency.md`

`specs/07_domain_priors.md` is a cross-cutting amendment file: read it TOGETHER with specs 02, 03, 04, and 06 — it modifies the LLM extraction schema, routing constraints, head pooling, auxiliary targets, stacking features, and pretraining program based on clinical evidence. Its §7.6 expected-gain map is the default prioritization when choosing the next experiment.
