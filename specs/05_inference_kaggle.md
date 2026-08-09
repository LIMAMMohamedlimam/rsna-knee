# Spec 05 — Kaggle Inference Plumbing (do the dry-run EARLY)

**Goal:** a green submission with a dummy model **before serious training finishes** (target: within 2 weeks of Spec 03 completion), then a hardened, profiled inference path reused by every later submission. Plumbing failures discovered in the final week are the #1 way strong teams lose this competition.

## Constraints (verbatim from rules)
- Notebook submission, internet OFF, ≤9h GPU (or CPU) runtime.
- Output: `submission.csv` with header `StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture`.
- ~1300 test studies; test DICOMs replace the placeholder at scoring time — code must discover studies from `test.csv` / directory listing, never hardcode counts.

## Task 5.1 — Dependency packaging
- `scripts/build_kaggle_wheels.py`: download pinned wheels (pylibjpeg, pylibjpeg-libjpeg, pylibjpeg-openjpeg, pydicom, timm, iterative-stratification not needed at inference — keep the inference dep set MINIMAL) into `kaggle_assets/wheels/`; upload as Kaggle Dataset `rsna-knee-wheels`. Notebook installs with `pip install --no-index --find-links=/kaggle/input/rsna-knee-wheels ...`.
- Weights: `kaggle_assets/weights/` mirrored as Kaggle Dataset `rsna-knee-weights-vN` (version per ensemble freeze). Include `routing_constants.py` values and any normalization stats INSIDE the weights dataset so the notebook has zero repo dependencies beyond a single `src` snapshot.
- `src` snapshot: `scripts/export_src.py` copies the minimal inference subset of `src/` (dicom_reader, preprocess, routing, models, infer) into `kaggle_assets/src/` — no training code, no test code. This snapshot is what the notebook imports.

## Task 5.2 — Inference entrypoint (`src/infer/predict.py`)
- `predict_study(study_dir, series_meta_df, models: list, cfg) -> np.ndarray (12,)`:
  1. Read + preprocess series with the EXACT Spec-03 code path (same functions, imported — never reimplemented).
  2. Route with the same routing constants.
  3. Forward through each model (fp16, `torch.inference_mode()`), average logits, sigmoid.
  4. TTA (config flag): laterality flip → flip input, forward, swap the 4 medial/lateral output indices back, average. This is the only TTA.
- Failure containment: ANY exception on a study → log, emit per-label train prevalence as the prediction, continue. A crashed notebook scores nothing; a prevalence-filled study costs ~nothing.
- Streaming: process studies one at a time (or small batches); never materialize all test volumes in RAM/disk simultaneously (Kaggle disk ~20 GB working space beyond inputs).

## Task 5.3 — Notebook generator (`src/infer/kaggle_notebook.py`)
- Emits `notebooks/kaggle_submit.ipynb` programmatically (nbformat) so the notebook is versioned as code: cells = [install wheels] [add src to path] [load models manifest] [loop studies with tqdm + running ETA] [write submission.csv] [print head + describe()].
- ETA guard cell: after the first 50 studies, extrapolate total runtime; if projection >8.0h, automatically switch to a config-declared degraded mode (drop TTA → drop lowest-weight ensemble member → reduce slices per series) and log which mode ran.

## Task 5.4 — Dry run (DO THIS FIRST, before any real model exists)
- Submit with a randomly initialized B1-size model. Success = green submission, any score.
- Record in `docs/kaggle_runbook.md`: exact steps to update weights dataset, re-run, submit; observed runtime breakdown (decode vs model) on the 3 example test studies extrapolated to 1300.

## Task 5.5 — Profiling & budget
- `scripts/profile_inference.py` on 50 local studies: seconds/study split into {decode, preprocess, forward, overhead}. Budget table to maintain in the runbook:
  - decode+preprocess ≤ 8 s/study → ~2.9h at 1300
  - model forward (full ensemble) ≤ 8 s/study → ~2.9h
  - target total ≤ 7h leaving 2h buffer.
- If decode is the bottleneck (likely): ensure pylibjpeg C-paths active, parallelize decode with a thread pool (I/O bound), decode-once-reuse-across-models.

## Acceptance
- [ ] Green dummy submission recorded (screenshot/score in runbook) — CALENDAR GATE, do not defer.
- [ ] Runbook complete; ETA guard tested by artificially inflating per-study time in a local run.
- [ ] Full-ensemble local profile ≤7h projected.
- [ ] `submission.csv` schema test: exact header string match, all probs in [0,1], row count == test.csv rows.
