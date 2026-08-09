# Spec 02 — Label Engine (LLM report → 12 labels)

**Goal:** `artifacts/labels/labels_final.parquet` giving every training study a soft label + confidence weight for each of the 12 conditions, validated at ≥95% agreement against the ground-truth subset for ≥10/12 labels.

**Why this exists:** only a subset of studies carries ground-truth labels; all carry reports; the test set has no reports. Label quality here upper-bounds everything downstream. Treat this phase with the same rigor as modeling.

## Task 2.1 — Prompt + schema (`src/labels/prompts.py`)

- Prompt templates are **versioned constants**: `PROMPT_V1`, `PROMPT_V2`, ... Never edit in place; add a version.
- The extraction prompt must:
  1. Present the 12 conditions with explicit clinical inclusion/exclusion rules. Encode these initial rules (refine during calibration, Task 2.3):
     - **ACL / MCL:** any tear (partial or complete), acute or chronic, counts as 1. "Sprain grade I" without tear → uncertain. Intact graft after reconstruction → 0; re-tear of graft → 1. "Mucoid degeneration" alone → uncertain.
     - **Menisci:** tear of any type (horizontal, radial, complex, root, bucket-handle) → 1. "Degeneration/signal without surfacing tear" → 0. Post-meniscectomy without new tear → uncertain.
     - **OA (3 compartments):** requires compartment-specific degenerative change: cartilage loss AND/OR osteophytes explicitly in that compartment. "Chondropathy grade ≥2" in the compartment → 1. Diffuse "mild degenerative change" without compartment → uncertain for all three.
     - **Effusion:** more than trace/physiologic fluid → 1; "trace/small physiologic" → 0.
     - **Synovitis:** explicit mention of synovitis/synovial thickening/proliferation → 1.
     - **Baker's:** popliteal/Baker's cyst any size → 1.
     - **Contusion:** bone marrow edema attributed to contusion/trauma → 1; edema attributed to OA subchondral change → 0 (that's OA, not contusion).
     - **Fracture:** any acute/subacute fracture incl. subchondral insufficiency fracture → 1; "old healed fracture" → uncertain.
  2. Demand JSON only, schema:
     ```json
     {"ACL": "present|absent|uncertain", ..., "Fracture": "...",
      "laterality": "left|right|both|unknown",
      "prior_surgery": true|false,
      "language": "iso639-1"}
     ```
  3. Include ≥2 few-shot examples per major language found in EDA (write them from real report *styles* but with synthetic content; note in code comment).
  4. Instruct: base answers ONLY on the report text; negations and "no evidence of X" → absent; unmentioned → absent for high-prevalence-of-mention findings (effusion) but **uncertain** for findings radiologists omit when normal is ambiguous — encode a per-condition unmentioned→{absent|uncertain} default table in the prompt (initial: unmentioned → absent for ACL/MCL/menisci/effusion/Baker's/fracture; uncertain for OA compartments/synovitis/contusion; refine in calibration).

## Task 2.2 — Extraction runner (`src/labels/extract.py` + `scripts/extract_labels.py`)

- Interface: `extract(reports: pd.DataFrame, model: str, prompt_version: str, out_path) -> parquet` with columns: StudyInstanceUID, model, prompt_version, per-label str values, laterality, prior_surgery, language, raw_json, error.
- Requirements:
  - Async batched calls, exponential backoff, resume-from-partial (skip already-processed UIDs by reading existing out file).
  - Strict JSON parsing with one repair-retry (re-ask with "return only valid JSON"). Persist failures with `error` filled; never crash the batch.
  - Cost guard: before running, print estimated tokens (Σ report_chars/4 + prompt overhead) and require `--yes` flag.
  - Support ≥2 backends behind one interface: an API model and a local open model (vLLM/ollama endpoint URL from env). Config chooses.
- Run matrix (the labeler ensemble): {model_A, model_B} × {PROMPT_Vlatest, PROMPT_Vlatest_paraphrase} = 4 passes over all reports. Store each as `artifacts/labels/raw_{model}_{prompt}.parquet`.

## Task 2.3 — Calibration loop (`src/labels/calibrate.py` + `scripts/calibrate_labels.py`)

- Input: one raw extraction file + ground-truth subset from `train.csv`.
- Mapping for scoring: present→1, absent→0, uncertain→excluded from agreement denominator (reported separately as uncertain_rate).
- Outputs per label: agreement, precision, recall, n_uncertain, and a **disagreement dump**: `docs/label_disagreements_{label}.md` listing up to 30 cases with (UID, GT, LLM, the report sentence(s) matched — extract the sentence containing the condition's keyword list for human review).
- Console summary table + hard gate line: `PASS` if agreement ≥0.95, else `FAIL`.
- **Iteration protocol (you, the agent, follow this loop):**
  1. Run calibration on the labeled subset.
  2. For each FAIL label, read the disagreement dump, classify the top error pattern, and propose a prompt rule change as a new `PROMPT_V{n+1}` (never mutate old versions).
  3. Re-extract **labeled subset only** (cheap) with the new prompt; re-calibrate.
  4. Repeat until ≥10/12 labels PASS or 5 iterations reached — then stop and report which labels are stuck and your hypothesis (likely: ground-truth noise vs genuine ambiguity — support with examples).
  5. Only after the gate: run the full 4-pass matrix on ALL reports.

## Task 2.4 — Merge (`src/labels/merge.py` → `artifacts/labels/labels_final.parquet`)

Combine 4 passes + ground truth into the final training-label table. Schema (one row per study):

| column | type | rule |
|---|---|---|
| StudyInstanceUID | str | |
| `{label}` ×12 | float32 | **soft label**: mean over passes of {present=1, absent=0, uncertain=0.5}; **overridden by ground truth (0/1) when present** |
| `{label}_w` ×12 | float32 | confidence weight for the loss: GT present → 1.0; else agreement-based: 4/4 agree → 1.0; 3/4 → 0.7; 2/2 split → 0.3; any-uncertain-majority → 0.15 |
| source | str | "gt" / "llm" |
| prior_surgery | bool | majority vote |
| laterality | str | majority vote; log conflicts |

- Version the file (`labels_v1.parquet`, symlink/copy the accepted one to `labels_final.parquet`).
- Sanity report auto-printed: per-label soft-prevalence on LLM-labeled studies vs GT prevalence on labeled subset — flag any label deviating >30% relative (suggests systematic extraction bias or genuine subset shift; investigate before accepting).

## Acceptance tests (`tests/test_labels.py`)
1. `merge` on a synthetic fixture of 6 studies × known pass outputs produces exactly the expected soft labels and weights (hand-computed).
2. GT always overrides LLM in the fixture.
3. `extract` resume logic: given a partial output file, only missing UIDs are queued (mock the API).
4. JSON repair path covered with a malformed-response mock.

## Exit criteria for Spec 02
- [ ] ≥10/12 labels at ≥95% agreement on GT subset, documented in `docs/label_calibration_report.md` (final table + iteration history).
- [ ] `labels_final.parquet` exists, sanity report clean or deviations explained in the report.
- [ ] All 4 raw pass files archived; prompts versioned in code.
- [ ] `docs/LOG.md` entry.
