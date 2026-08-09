# Spec 04 — Modeling

**Goal:** climb a strict baseline ladder to the main two-stage architecture with text-teacher training, reaching CV macro-AUC ≥0.92 with per-label floor ≥0.80. Every experiment = one config file + one OOF file + one LOG entry.

## Experiment protocol (applies to everything below)

- Config naming: `configs/exp/e{NNN}_{slug}.yaml`, NNN monotonically increasing.
- Fast loop: hypothesis-testing runs train **fold 0 only**. Promote to 5-fold only if fold-0 macro-AUC improves ≥0.003 over current best.
- Adoption rule for 5-fold runs: adopt iff macro-AUC improves on ≥3/5 folds AND mean improves.
- Every run writes: `artifacts/oof/e{NNN}_fold{k}.parquet`, `artifacts/weights/e{NNN}_fold{k}.pt`, and metrics to the tracker (W&B project `rsna-knee`, run name = config name).
- `src/train/metrics.py`: per-label AUC (skip labels with <2 classes in fold — warn), macro-AUC, and per-label AUC computed **only on GT-labeled studies** as a secondary "clean" metric (LLM-label noise can inflate the primary).

## Task 4.1 — Training loop (`src/train/loop.py`)

Standard but exact:
- AMP (bf16 if available), grad clip 1.0, AdamW, cosine schedule with 3-epoch warmup, EMA of weights (decay 0.999) — evaluate EMA weights.
- Loss (in `losses.py`): masked weighted BCE-with-logits:
  `loss = Σ_labels w_i * BCE(logit_i, soft_target_i) / Σ w_i` — soft targets from labels_final; weights `{label}_w`; NaN target → w=0.
- Checkpointing: save best-by-fold-macro-AUC + last; resume support.
- Early stop: patience 5 evals. Eval every epoch.
- Batch construction: study-level batches (default bs=8 studies × 4 series × 32 slices at 224–256² — verify memory on first run, gradient-accumulate if needed).
- Optional per-label positive sampling boost via config `sampler.pos_boost: {Fracture: 3.0, Contusion: 2.0}` implemented with WeightedRandomSampler (explicit in config only, per CLAUDE.md §6).

## Task 4.2 — Baseline ladder

### B0 — prevalence sanity (`e001`)
Predict per-label train prevalence for every study. Assert OOF macro-AUC ≈ 0.5. Purpose: metrics + OOF plumbing verified.

### B1 — single-series 2.5D (`e002`)
- Route slot-1 series only (sag fluid). Model: timm `convnext_tiny` (pretrained, in_chans=3) over slice windows: input slices grouped as overlapping 3-slice channels; per-window embeddings mean-pooled; linear 12-head.
- 224², 15 epochs, lr 1e-4 (head 1e-3), fold 0.
- **Expected shape of result:** ACL/menisci AUC high (≥0.90), axial-dependent labels (PF OA, Baker's, synovitis) mediocre. If ACL <0.85 something is broken (routing, ordering, or labels) — STOP and debug before proceeding; check 10 highest-loss studies visually.
- Promote to 5 folds when sane. **Gate to continue: 5-fold macro-AUC ≥0.85.**

### B2 — multi-series + attention (`e00x`)
- All 4 routed series. Per-series: B1 encoder → slice-attention pooling (single-head attention with learned query) → series embedding (+ learned meta-token embedding: plane/fluid/fatsat added). Study: attention pooling over 4 series embeddings → 12 heads.
- **Gate: 5-fold macro-AUC ≥0.89.** Submit B2 to Kaggle for a public-LB reference point (see Spec 05 plumbing — must already be green).

## Task 4.3 — Main architecture (`src/models/encoder2p5d.py` + `aggregator.py`)

Two-stage, the RSNA-winning family:

### Stage A — sequence-aware series encoder
- Backbone per 3-slice window (timm, config: `convnext_small.fb_in22k_ft_in1k` default) → per-window embedding (dim d=768 proj to 512).
- **Slice-axis model:** 2-layer bidirectional transformer encoder (or BiGRU config-switchable) over the window sequence with learned positional encoding scaled by native slice count. Output: attended series embedding + per-slice logits (deep supervision head, weight 0.2, targets = study labels broadcast — standard MIL trick).

### Stage B — study aggregator
- Inputs: 4 series embeddings + meta tokens + PatientSex token.
- 2-layer transformer (d=512, 8 heads) with a learned [CLS] token → 12 main logits + auxiliary heads (`heads.py`): prior_surgery (BCE, w=0.1), per-series plane prediction (CE, w=0.05 — regularizer sanity check).
- Train end-to-end with grad checkpointing; if OOM, two-stage: cache Stage-A embeddings per epoch-0 model, train B, then joint finetune 3 epochs.

### Text-teacher distillation (the differentiator — implement carefully)
- Precompute report embeddings once: multilingual encoder (e.g., `intfloat/multilingual-e5-large` via sentence-transformers) over full report → `artifacts/labels/report_emb.parquet` (StudyUID, 1024-d).
- Add to the loss (train time only, never inference):
  `L = L_bce + λ_clip * L_infoNCE(study_emb_proj, report_emb_proj)` with λ_clip=0.2, temperature learned, in-batch negatives; both sides projected to 256-d.
- Ablate explicitly: e{N} without vs e{N+1} with λ_clip. Keep only if adoption rule passes. Also ablate soft-vs-hard labels the same way.

**Gate: 5-fold macro-AUC ≥0.91 before starting per-label program.**

## Task 4.4 — Per-label program (run weekly once main model trains)

`scripts/label_report.py`: from best OOF — per-label AUC (all + GT-only), ROC curves, 10 worst false negatives + false positives per label with their routed series listed. For the current 3 worst labels open one micro-experiment each. Menu of interventions (pick per failure pattern seen in errors):
- **Routing fix:** label mostly missed when slot-N fell back → adjust routing/fallbacks.
- **Resolution:** small findings (meniscal root, fracture line) → 384² hires cache for the relevant slot.
- **Anatomy crop:** PF OA / Baker's → axial patella-centered / posterior crop head (simple fixed-fraction crops first; learned localizer only if that fails).
- **Loss:** rare labels → pos_boost sampling or focal loss (γ=2) for that head only.
- **Label audit:** if GT-only AUC ≫ all-study AUC for a label → LLM labels noisy there → tighten Spec-02 prompt for that condition and re-merge (new labels version, retrain).
- **External data (allowed by rules):** MRNet (ACL/meniscus/abnormal) for pretraining Stage A on sagittal knees; SKM-TEA / fastMRI knee for self-supervised pretraining (MAE-style) if time permits. Track licenses in `docs/external_data.md` for winner obligations.

## Task 4.5 — 3D probe (single experiment, timeboxed)
One run: 3D ResNet-50 (or X3D-M) on 160×160×32 volumes, slot-1+2 series, fold 0. Purpose: check if true 3D context helps Contusion/Fracture specifically. Adopt only for those heads (per-head ensemble later) if ≥+0.02 on either.

## Exit criteria for Spec 04
- [ ] Ladder gates hit: B1 ≥0.85, B2 ≥0.89, main ≥0.91, after per-label program ≥0.92 CV macro-AUC (5-fold), per-label floor ≥0.80.
- [ ] Ablation table in `docs/ablations.md`: soft labels, text-teacher, aux heads, 3D probe — each with fold-0 and (if promoted) 5-fold deltas.
- [ ] ≥3 diverse promoted configs retained as ensemble candidates (different backbone or resolution or routing).
- [ ] `docs/LOG.md` entries for every experiment.
