# Spec 06 — Ensemble, Efficiency Track, Final Selection

**Goal:** (1) a full ensemble maximizing CV macro-AUC within the 9h budget; (2) a distilled single model targeting top-3 on the Efficiency track; (3) two final submissions selected by the pre-committed CV rule.

## Task 6.1 — Ensemble construction (`scripts/build_ensemble.py`)

- Candidates: every promoted 5-fold config from Spec 04 (expect 3–5). Ensembling operates on **logits** (mean), per label.
- Greedy forward selection on OOF: start from best single; add the member that maximizes OOF macro-AUC; stop when improvement <0.001 or runtime budget exceeded (each member's per-study forward time comes from the Spec-05 profile — the script must check the projected total against 7h and refuse over-budget ensembles).
- Also evaluate **per-label member weighting**: constrained weights (simplex, per label) fit on OOF via scipy minimize; accept over uniform mean only if OOF macro-AUC +≥0.0015 AND the GT-only metric agrees (guards against fitting LLM-label noise).
- Stacking (optional, timeboxed 1 evening): per-label logistic regression over member logits with fold-consistent OOF; accept under the same double-agreement rule. Expect to reject it.
- Output: `artifacts/ensemble/manifest_vN.json` — list of (config, fold, weight path, per-label weights), OOF macro-AUC, projected runtime. The Kaggle weights dataset is built FROM a manifest, never assembled by hand.

## Task 6.2 — Efficiency distillation (`src/models/distill.py` + `scripts/train_distill.py`)

- Teacher: final ensemble's OOF predictions (out-of-fold, so no leakage) + its full-train predictions from fold models on their own folds — build `artifacts/ensemble/teacher_soft.parquet` (StudyUID × 12 soft probs).
- Student: single 2.5D model, `efficientnetv2_s` or `convnext_tiny`, 224², **2 routed series only** (slot 1 + slot 3 — verify with an ablation that dropping coronal costs least; if MCL/OA collapse, use slots 1+2+3 at 24 slices).
- Loss: `0.7 * KLDiv(student, teacher_soft) + 0.3 * weighted BCE(labels_final)`. Train on ALL studies (teacher covers unlabeled ones), 5 folds → pick single best fold model OR average-weights (SWA-style soup) of the 5 — evaluate both on OOF.
- Runtime engineering, in order of effort/return: fp16 → channels_last → `torch.compile` → ONNXRuntime (only if wheels available offline) → reduce slices. Target: **<45 min total notebook runtime** at 1300 studies (~2 s/study all-in).
- Track the daily efficiency leaderboard notebook; log our projected standing weekly in `docs/LOG.md`.
- Gate: distilled model within **0.010** OOF macro-AUC of the full ensemble. If gap >0.015, distill from a 2-member sub-ensemble teacher and/or increase student to 288².

## Task 6.3 — Final selection (pre-committed rule — do not renegotiate in the last week)

- Selection metric: 5-fold OOF macro-AUC, with GT-only macro-AUC as tiebreaker. Public LB is used ONLY as a plumbing sanity check (a submission scoring wildly below CV means a bug, not a modeling signal).
- Submission 1: best manifest ensemble (runtime-verified).
- Submission 2: distilled efficiency model IF its projected efficiency rank ≤5 on the tracked leaderboard; ELSE second-best diverse ensemble.
- Freeze calendar: **Oct 19** code freeze → Oct 19–21 re-run both notebooks end-to-end on fresh notebook versions → Oct 21 select → buffer day Oct 22.

## Task 6.4 — Winner obligations package (build as you go, finalize only if in the money)

`docs/obligations/`:
- `SOLUTION.md`: method description (architecture diagram, label engine description, ablation table, external data + licenses).
- Training code path: the repo already is it — add `REPRODUCE.md` with exact commands from raw data to final weights.
- Open-source license file (check rules for required license; default Apache-2.0 if unconstrained).
- Model publication checklist: weights → public Kaggle Model (format per the example linked in rules), forum post template with links.
- Video outline (3–5 min): problem → label engine → architecture → results → efficiency model. Slides only if placing.

## Acceptance
- [ ] `manifest_vFinal.json` with OOF macro-AUC ≥ best single +0.004 and projected runtime ≤7h.
- [ ] Distilled model: OOF gap ≤0.010, runtime <45 min projected, submitted at least once.
- [ ] Both final submissions re-run green during freeze window; selection recorded with justification in `docs/LOG.md`.
- [ ] Obligations skeleton exists.
