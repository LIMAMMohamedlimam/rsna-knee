# Spec 03 — Data Pipeline (DICOM → cache → Dataset)

**Goal:** raw 570 GB DICOM converted ONCE into a deterministic, compact cache (~≤80 GB) that trains at >500 slices/s, plus routing logic and a laterality-safe augmentation stack. The identical preprocessing code path must be reusable inside the offline Kaggle notebook.

## Task 3.1 — DICOM reader (`src/data/dicom_reader.py`)

- `read_series(series_dir: Path) -> SeriesVolume` where `SeriesVolume` is a dataclass: `pixels: np.ndarray (S,H,W) float32`, `spacing: (dz,dy,dx)`, `orientation: str`, `meta: dict`.
- Requirements:
  - Handle all 4 transfer syntaxes via pydicom + pylibjpeg family. On decode failure of a single slice: log, skip slice, continue; on >20% slice failures: raise `SeriesUnreadable`.
  - **Slice ordering:** sort by projection of ImagePositionPatient onto the slice-normal (cross product of ImageOrientationPatient rows). Fall back to InstanceNumber only if position tags missing; record `order_source` in meta.
  - Apply RescaleSlope/Intercept if present. Guard EVERY tag access (`ds.get(...)`) — only 86 allowlisted tags exist.
  - Handle mixed per-slice shapes within a series: resize minority slices to the modal shape; log.
- **Unit tests first:** commit 4 tiny fixture DICOMs (one per transfer syntax — generate synthetic ones with pydicom if real ones can't be committed) and test decode + ordering on shuffled filenames.

## Task 3.2 — Cache builder (`src/data/preprocess.py` + `scripts/build_cache.py`)

Per series:
1. `read_series` → float32 volume.
2. Intensity: clip to [p0.5, p99.5] computed per-series → scale to uint8 [0,255].
3. In-plane resize to **256×256** (bilinear, preserve aspect via pad-to-square with zeros first). Also store center-crop 384×384 variant ONLY for series whose native size ≥384 (config flag `cache.hires: false` initially — build later if needed).
4. Slice handling: keep native slice count up to 64; if >64, uniformly subsample to 64. Store native count in index.
5. Write `artifacts/cache/{StudyUID}/{SeriesUID}.npz` with keys `vol (uint8)`, `spacing (float32[3])`.
6. Append row to global index parquet `artifacts/cache/index.parquet`: StudyInstanceUID, SeriesInstanceUID, plane, fluid, fatsat (joined from train_series.csv), n_slices_native, n_slices_cached, H, W, spacing, order_source, site_cluster, fold, bytes.

Runner requirements: multiprocessing (n_workers config), resumable (skip existing npz), progress bar, final integrity pass (every series in train_series.csv is cached or listed in `cache/failures.csv` with reason). Determinism: re-running on an already-cached series produces a byte-identical file (test this).

**QA task:** `scripts/qa_cache.py --n 50` renders 50 random series as PNG contact sheets (middle 9 slices) into `docs/figures/cache_qa/` for your visual review; automatically flag series with >90% zero pixels or all-constant volumes.

## Task 3.3 — Routing (`src/data/routing.py`)

- Input: the canonical protocol table from EDA (read `docs/eda_report.md` values, then hardcode as reviewed constants).
- `route_study(study_series: pd.DataFrame, cfg) -> list[SeriesInstanceUID]` returning an **ordered** list of series to feed the model, by priority slots:
  1. Sagittal fluid-sensitive (fat-sat preferred) — ACL, menisci, contusion, effusion.
  2. Coronal fluid-sensitive — MCL, medial/lateral OA, menisci.
  3. Axial fluid-sensitive — PF OA, synovitis, Baker's, effusion.
  4. Sagittal non-fluid (T1) — chronic OA, fracture line.
- Fallback rules when a slot is empty: substitute nearest available (e.g., no ax-fluid → ax any → sag-fluid duplicate). Always return exactly `cfg.data.n_series` entries (default 4; duplication allowed, log fallback usage rate).
- Test: synthetic studies with missing planes route without error and hit documented fallbacks.

## Task 3.4 — Dataset & augmentation (`src/data/dataset.py`, `src/data/augment.py`)

- `KneeStudyDataset(index, labels_final, folds, cfg, split)` returns:
  ```python
  {"volumes": float32 tensor (n_series, n_slices, H, W),   # n_slices = cfg (default 32, center window or uniform sample)
   "meta_tokens": int tensor (n_series, 3),                # plane id, fluid, fatsat
   "targets": float32 (12,), "weights": float32 (12,),
   "study_id": str}
  ```
- Normalization: uint8 → float /255, per-slice z-score optional via config.
- **Augmentations (train only), all config-driven:**
  - `laterality_flip(vol, targets, weights)` p=0.5: horizontal flip AND swap index pairs (Medial Meniscus↔Lateral Meniscus, Medial OA↔Lateral OA) in BOTH targets and weights. Implemented once here; the ONLY flip allowed in the codebase.
  - Rotate ±10°, scale 0.9–1.1, translate ±5% (applied consistently across slices of a series).
  - Intensity: brightness/contrast ±0.2, gamma 0.8–1.2.
  - Slice dropout: drop each slice p=0.1 (replace with zeros) — robustness to short series.
  - Coarse dropout: 1–4 holes ≤32px.
- **Tests:** (a) laterality_flip twice == identity for both pixels and labels; (b) flip on an asymmetric synthetic volume moves a marker from left to right; (c) dataset smoke test over 5 studies returns correct shapes and masks weights where labels are NaN→(target 0.5, weight 0).

## Task 3.5 — Throughput benchmark

`scripts/bench_loader.py`: measure slices/s with cfg workers on cached data. Gate: ≥500 slices/s on 8 workers. If below: profile (npz decode vs augmentation), consider `np.load(mmap_mode)` or moving resize into cache.

## Exit criteria for Spec 03
- [ ] Full cache built; `failures.csv` reviewed (<0.5% series failed, or failures explained).
- [ ] QA contact sheets visually clean; auto-flags resolved.
- [ ] Routing fallback usage report printed (<10% of studies use fallback for slot 1).
- [ ] All tests green incl. determinism + laterality; loader ≥500 slices/s.
- [ ] `docs/LOG.md` entry.
