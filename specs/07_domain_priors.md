# Spec 07 — Clinical Domain Priors (evidence-backed modelization)

**Goal:** encode radiological domain knowledge into the pipeline as concrete components: extra LLM-extracted auxiliary targets, routing constraints, head-level architecture changes, a label co-occurrence prior for stacking, and external pretraining assets. This spec AMENDS specs 02, 03, 04, and 06 — apply each amendment where marked. Nothing here replaces the gates in those specs; interventions are adopted under the same fold-0 → 5-fold adoption rule.

Sources are peer-reviewed radiology/epidemiology literature and prior RSNA challenge winners; each prior below states its evidence in one line so future-you can audit it.

---

## 7.1 Per-condition prior table (the master reference)

| Label | Primary series (routing slot) | Anatomical zone (for crops) | Key imaging signature | Discriminative pitfalls | Approx. performance ceiling / base rates |
|---|---|---|---|---|---|
| ACL | Sagittal fluid-sensitive (slot 1) | Intercondylar notch | Fiber discontinuity; empty notch (chronic) | Intact graft post-reconstruction reads as normal; mucoid degeneration mimics tear | MRNet AUC 0.965 internal, 0.824 external → expect ~0.10 domain-shift drop risk |
| MCL | Coronal fluid-sensitive (slot 2) | Medial joint line, femoral epicondyle → tibial insertion | Acute: bright fluid superficial to ligament; chronic: thickening WITHOUT fluid | Acute-vs-chronic differ by sequence signature (T2 fluid vs T1 thickening) | Grade III MCL → 78% have ACL tear |
| Medial Meniscus | Sag + Cor fluid (slots 1+2, cross-plane) | Posterior horn dominant | Surfacing signal on ≥2 slices (two-slice-touch); root tear = 1 slice suffices | Intrasubstance degeneration ≠ tear; post-meniscectomy | Hardest label class: MRNet meniscus AUC 0.847; medial ≫ lateral (63.5% vs 26.1% of tears); tears skew male (~70%) |
| Lateral Meniscus | Sag + Cor fluid | Posterior horn; uncovering sign when ACL torn | Same as medial | In acute ACL context, lateral involvement is MORE common than medial (unhappy-triad revision) | — |
| Medial OA | Coronal (slot 2) + T1 (slot 4) | Medial tibiofemoral compartment | Compartment cartilage loss + osteophytes | OA subchondral edema must NOT trigger Contusion head | OA skews female, older |
| Lateral OA | Coronal + T1 | Lateral tibiofemoral compartment | Same | Rarest OA compartment → rare-label program | — |
| PF OA | Axial (slot 3) | Patellofemoral joint (patella-centered crop) | Patellar/trochlear cartilage loss, osteophytes | Invisible without axial/patellar coverage | — |
| Effusion | Axial + Sagittal fluid | Suprapatellar recess | Fluid volume above trace | "Trace physiologic" = 0 by our labeling rule | ~30% baseline in OA cohorts (MOST); bridges acute & degenerative clusters |
| Synovitis | Axial fluid ± (Hoffa region on sagittal) | Suprapatellar + Hoffa's fat pad | Synovial thickening; Hoffa hyperintensity | Non-contrast MRI under-detects → labels intrinsically noisy → expect lower agreement ceiling in Spec 02 calibration | Hoffa-synovitis ~37% in OA cohorts; effusion-synovitis volume correlates r=0.74 with MOAKS |
| Baker's | Axial + Sagittal (slot 3+1) | Popliteal fossa, posteromedial (94% medial side) | Fluid between medial gastrocnemius & semimembranosus | Near-deterministic from other labels — see §7.2 | Adult prevalence 10–41%; 62% co-occur with posterior-horn medial meniscus tear |
| Contusion | **Fat-sat fluid-sensitive REQUIRED** (slot 1) | Lateral femoral condyle + posterolateral tibia (pivot-shift); variable | Bright marrow on fat-sat T2/PD, dark on T1 | Essentially invisible without fat suppression; OA subchondral change is NOT contusion | Present in 70–80% of acute complete ACL tears (pivot-shift pattern) |
| Fracture | Fat-sat fluid (edema) + T1 (fracture line, slot 4) | Variable; Segond = lateral tibial rim; tibial spine; deep lateral femoral notch | Line + surrounding edema | Segond & deep-notch fractures are subtle AND high-specificity ACL markers | Rare label → rare-label program |

---

## 7.2 Co-occurrence prior (AMENDS Spec 06, Task 6.1)

Quantified dependency structure, from the literature:

- P(Baker's | any one of {Effusion, Meniscal tear, Degenerative arthropathy}) ≈ 0.08–0.10; given any two ≈ 0.19–0.21; given all three ≈ 0.38. Associations independent of each other. **No association** with ACL or MCL.
- P(ACL | MCL grade III) ≈ 0.78.
- Pivot-shift contusion pattern ⇒ strong evidence for ACL (secondary signs >80% specificity: pivot-shift bruise, Segond fracture).
- Two clusters + bridge: ACUTE {ACL, MCL, Contusion, Fracture, Lateral Meniscus} vs DEGENERATIVE {3×OA, Synovitis, Baker's, medial meniscus (degenerative)} with Effusion in both. Synovitis mediates meniscus→OA.

**Implementation (Task 6.1 amendment):**
1. Add to the stacking evaluation a per-label GBM/logistic layer whose features are: the other 11 OOF logits + PatientSex + the auxiliary-head outputs of §7.3. Primary expected beneficiaries: **Baker's, MCL, Synovitis**. Evaluate under the standard double-agreement rule (OOF + GT-only both improve).
2. Add a diagnostic to `scripts/label_report.py`: model's implied conditional P(Baker's | effusion∧meniscus∧OA predictions binarized at Youden) vs the literature ladder above; large deviation = the model isn't exploiting the dependency → stacker likely to help.
3. Calibration check: per-sex predicted prevalence ratios should qualitatively reproduce meniscal-tear-skews-male / OA-skews-female. Add to the weekly label report.

## 7.3 Additional LLM auxiliary targets (AMENDS Spec 02, Tasks 2.1/2.4)

Extend the extraction JSON schema (new prompt version, re-run calibration gate only for the 12 core labels; aux fields are best-effort):

```json
{
  ...12 core labels...,
  "acute_vs_degenerative": "acute|degenerative|mixed|unknown",
  "pivot_shift_contusion": "present|absent|unknown",   // LFC + posterolateral tibia bruise pattern
  "effusion_size": "none|trace|small|moderate|large|unknown",
  "meniscus_tear_location": {"medial": "none|posterior_horn|root|body|anterior|complex|unknown",
                              "lateral": "..."},
  "segond_or_avulsion": "present|absent|unknown",
  "prior_surgery": true,
  "laterality": "left|right|both|unknown"
}
```

`merge.py`: carry these into `labels_final.parquet` as additional columns (majority vote, unknown-tolerant). They become auxiliary heads (§7.4) and stacker features (§7.2). Note for calibration expectations: synovitis on non-contrast MRI is intrinsically under-reported — if synovitis is one of the ≤2 labels failing the 95% gate, document and move on rather than burn iterations.

## 7.4 Architecture amendments (AMENDS Spec 04)

1. **Meniscus heads — two-slice-touch aggregation (Task 4.2/4.3):** for the 2 meniscus heads (and optionally Fracture), replace attention-mean slice pooling with **top-k mean over per-slice logits (k=2–4)** or noisy-OR, mirroring the clinical rule that a tear = abnormality on ≥2 slices. Keep attention pooling for diffuse findings (OA, effusion, synovitis). Config: `heads.pooling: {default: attention, Medial Meniscus: topk2, Lateral Meniscus: topk2, Fracture: topk2, Contusion: topk3}`. Ablate as one experiment.
2. **Cross-plane fusion for meniscus/MCL:** the two-slice-touch rule explicitly allows the 2 confirming slices to come from different planes at the same location. Ensure the Stage-B aggregator lets sagittal and coronal series tokens interact BEFORE the meniscus/MCL heads (they already do via the transformer; add a check that these heads read the [CLS] token, not a single-series token). MRNet's weak meniscus AUC (0.847, sagittal-dominant) is the cautionary tale.
3. **Secondary-sign coupling for ACL:** add the §7.3 aux heads (`pivot_shift_contusion`, `segond_or_avulsion`, `acute_vs_degenerative`, `effusion_size` ordinal) as auxiliary outputs of Stage B (weight 0.1 each). Rationale: pivot-shift bruise pattern is present in 70–80% of acute complete ACL tears and >80% specific — forcing the representation to encode it is free ACL signal, and it doubles as the Contusion head's spatial prior.
4. **OA vs Contusion disambiguation:** both produce marrow signal. The `acute_vs_degenerative` aux head is the disambiguator; additionally, log in error analysis whether Contusion false positives concentrate in high-OA-probability studies (if yes, add an explicit interaction feature in the stacker).
5. **Protocol-completeness feature:** from §7.1, Contusion/Fracture are near-undetectable without fat-sat fluid sequences. Routing already prefers them (Spec 03); ADD: a per-study binary meta-token `has_fatsat_fluid` fed to Stage B, and report Contusion/Fracture AUC stratified by this flag in the weekly label report. If the no-fat-sat stratum is large and bad, consider excluding those studies from Contusion/Fracture LOSS (mask) since their LLM labels came from reports written off sequences we may not be routing.
6. **Anatomy crops (Task 4.3.3, now prioritized):** literature-backed crop targets — patella-centered axial crop (PF OA), popliteal posteromedial crop (Baker's), intercondylar notch sagittal crop (ACL), lateral tibial rim (Segond/Fracture). The RSNA-2024 winner's localize-then-classify pipeline is the template; start with fixed fractional crops (no learned localizer) — knee FOV is well standardized.
7. **Winner tricks to add to the augmentation/TTA menu (from RSNA 2022–2024 winners):** slice-sequence order flip, manifold mixup on embeddings, max-pool over slice logits (2023 winner), pseudo-labeling of low-confidence studies with the current best model in later rounds, rotation TTA (evaluate cost under 9h budget before adopting).

## 7.5 External data & pretraining program (AMENDS Spec 04, Task 4.3.2)

All public/free → allowed by competition rules. Priority order:

| Asset | What it gives | Use | Effort |
|---|---|---|---|
| **MRNet** (Stanford, 1,370 exams) | Study-level ACL / meniscus / abnormal labels, 3 planes | Supervised pretraining of Stage A+B at our exact task shape; also a free external-validation set for ACL/meniscus heads | Low — do first |
| **fastMRI+** (knee) | 16,154 bounding boxes, 13 study-level labels, 22 pathology categories | Pretrain the detection-style auxiliary / anatomy-crop verification; box supervision for effusion & meniscus | Medium |
| **SKM-TEA** (155 pts, ~25k slices) | 4-tissue segmentations + 16 pathology boxes, sagittal qDESS | Cartilage segmentation pretraining → OA heads; segmentation aux head à la 2023 winner | Medium |
| **OAI** | Massive longitudinal knee MRI, OA-rich, MOAKS subset | **DINO/MAE self-supervised pretraining of Stage-A backbone** (published evidence: OAI-DINO significantly beats from-scratch and beats supervised pretraining on classification) — combine with our own unlabeled cache | High (weekend GPU job) — schedule only if Phase-4 gates are met early |

Log every asset + license in `docs/external_data.md` (winner obligations).

## 7.6 Expected-gain map (prioritization guidance for the agent)

When choosing the next micro-experiment, prefer by expected macro-AUC impact:
1. Stacker with co-occurrence features → Baker's, MCL, Synovitis (§7.2) — cheap, high-confidence.
2. Top-k pooling + cross-plane check → both meniscus labels (§7.4.1–2) — cheap.
3. Axial patella / popliteal crops → PF OA, Baker's (§7.4.6) — medium.
4. Secondary-sign aux heads → ACL, Contusion (§7.4.3) — medium.
5. MRNet pretraining → ACL, menisci + free external validation (§7.5) — medium.
6. OAI SSL → everything, especially OA heads (§7.5) — expensive, do last.

## Acceptance
- [ ] Prompt vN+1 with §7.3 schema shipped; core-label calibration re-passed; labels_final regenerated with aux columns.
- [ ] `has_fatsat_fluid` token + stratified Contusion/Fracture reporting live.
- [ ] Top-k pooling and stacker experiments run under the standard adoption rule; results in `docs/ablations.md`.
- [ ] Per-sex prevalence and Baker's-conditional diagnostics added to the weekly label report.
- [ ] `docs/external_data.md` created with licenses.
