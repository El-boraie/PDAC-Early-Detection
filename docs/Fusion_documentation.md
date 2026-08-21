# Fusion — Real Inference-Time Score Combiner

**Source:** `src/fusion/fusion.ipynb`.
**Purpose:** the actual inference-time combiner the dashboard will call — loads both
branches' finished artifacts from `checkpoints/{clinical,imaging}/final/` only (never
retrains, never fits anything on synthetic/simulated paired data), applies two fixed,
hand-set rules (cross-modality weighting, and imaging slice-vs-volume aggregation), and
writes this stage's own `checkpoints/fusion/final/model_card.json`. **Not** the evaluation
notebook (`fusion_evaluation.ipynb`, documented separately in
`docs/Fusion_Evaluation_documentation.md`).
**Run date:** 2026-07-19.

---

## Two Fixed Rules, Both Judgment Calls, Neither Fitted

This notebook makes exactly two hand-set decisions. Both are stated as named constants
first, then backed by traceable supporting numbers — same pattern `imaging_evaluation.ipynb`
and `clinical_final_fit.ipynb` use for `WINNER_MODEL_NAME`.

### Rule 1 — Cross-modality combination weight

```
fused = 0.4 * imaging_calibrated_proba + 0.6 * tabular_calibrated_proba
```

`W_IMAGING = 0.4`, `W_TABULAR = 0.6` (asserted to sum to 1.0). **A deliberate, documented
judgment call, not an empirically fitted parameter** — there is no paired ground truth to
fit or tune weights against (see the pair-evaluation notebook's central constraint).
Imaging is weighted *below* clinical because it carries a disclosed, unresolved
confound-check finding (see below) — its detection confidence is not yet confirmed to
reflect genuine tumour-specific reasoning, so it is deliberately not allowed to dominate a
disagreement between the two branches. The discount is kept modest (0.4, not lower) because
imaging's own calibration is strong and, post-calibration, it is slightly under-confident,
so a larger discount would over-correct.

### Rule 2 — Imaging slice-vs-volume aggregation

```
patient_score = mean(per-slice calibrated probabilities across the whole scan)
```

`IMAGING_SLICE_AGGREGATION = "mean"`. Applies **only** to whole-volume input — the imaging
model's native single-slice path (how it was trained and evaluated throughout this project)
remains available and unaffected. Added because a real CT scan *is* a volume (128–526 slices
in this dataset); requiring a user to pre-select the one slice showing the tumour would push
the model's own detection job onto them. This is also a **fixed, hand-set inference-time
rule, not a fitted parameter** — the imaging branch's own training/evaluation notebooks are
not retrofitted with it.

---

## Measurement Behind Rule 2

Measured, not asserted: imaging's real 5-fold OOF slice predictions
(`results/imaging/oof_predictions.csv`), calibrated with the branch's own saved
`calibrator.pkl`, aggregated per patient several different ways, scored against the
patients' real labels.

**Legitimacy check performed first:** every patient's slices sit entirely within one CV
fold (verified — zero patients span folds) and every patient has one consistent label across
all its slices (verified — zero inconsistencies), so patient-level aggregation of OOF scores
never mixes a fold's held-out predictions with its training predictions, and the
patient-level label is well-defined.

| Granularity | Rule | n | ROC-AUC | Precision | Recall | F1 | Brier | TN/FP/FN/TP |
|---|---|---|---|---|---|---|---|---|
| Slice (baseline) | every slice scored independently | 90,693 | 0.9920 | 0.9887 | 0.9686 | 0.9786 | 0.0282 | 17,821/795/2,262/69,815 |
| Volume (patient) | **mean** | 361 | 0.9973 | 0.9928 | 0.9858 | **0.9893** | **0.0142** | 78/2/4/277 |
| Volume (patient) | median | 361 | 0.9975 | 0.9928 | 0.9822 | 0.9875 | 0.0146 | 78/2/5/276 |
| Volume (patient) | max | 361 | 0.9913 | 0.9272 | 0.9964 | 0.9605 | 0.0577 | 58/22/1/280 |
| Volume (patient) | top-5%-slice-mean | 361 | 0.9974 | 0.9428 | 0.9964 | 0.9689 | 0.0449 | 63/17/1/280 |
| Volume (patient) | top-10-slice-mean | 361 | 0.9976 | 0.9396 | 0.9964 | 0.9672 | 0.0458 | 62/18/1/280 |
| Volume (patient) | 90th-percentile | 361 | 0.9973 | 0.9654 | 0.9929 | 0.9789 | 0.0306 | 70/10/2/279 |

**`mean` wins on F1 and Brier.** `max`/top-k buy marginally higher recall (0.996 vs. 0.986)
but collapse precision — a ~27% false-positive rate on healthy patients (22 of 80) — not a
worthwhile trade given the disclosed confound below. Saved to
`results/fusion/imaging_slice_vs_volume_aggregation.csv`.

**Read this table as imaging-only, not a joint benchmark.** `dataset` perfectly predicts
`class` in this project (every MSD patient is cancer, every NIH patient is healthy), so
these patient-level numbers inherit the same scanner-confound the slice-level ones already
carry — they are not cleaner evidence, just the same signal at a coarser grain.

### The honest catch: why `mean` wins is itself diagnostic

A within-patient spread check, run on the same data:

| | Cancer patients (n=281) | Healthy patients (n=80) |
|---|---|---|
| Mean of per-patient MEAN slice prob | 0.973 | 0.132 |
| Mean of per-patient MIN slice prob | 0.424 | — |
| Mean of per-patient MAX slice prob | — | 0.348 |
| Mean of per-patient STD | 0.053 | — |
| Patients whose lowest slice still scores >0.5 | 35.6% | — |
| Patients with *any* slice scoring >0.5 | — | 27.5% |

Only a minority of a cancer patient's ~250 slices actually contain visible tumour, yet every
slice carries that patient's cancer label, and the model's per-slice scores barely move
(std ≈ 0.05; even the *lowest*-scoring slice in a cancer patient averages 0.42). If the model
genuinely localized tumours, within-patient variance would be high and `max`/top-k would win.
It doesn't, and they don't. **`mean` outperforming the alternatives is itself a fingerprint
of the same confound** — the "cancer signal" reads as global to the volume, not localized to
tumour-bearing slices. Volume aggregation therefore improves the numbers *without resolving*,
and can visually obscure, the tumour-under-reliance finding. This reasoning is recorded
verbatim in the fusion-level model card (see below), not just in this document.

---

## Loading Both Branches (`checkpoints/*/final/` only, never retrained)

**Clinical:** `model.pkl`, `ca19_9_imputer.pkl`, `calibrator.pkl` loaded as-is.

**Real gotcha hit and documented:** `ca19_9_imputer.pkl` was pickled by `joblib.dump` from a
`MICE_CA19_9Imputer` class defined inline in `clinical_final_fit.ipynb`'s own kernel
`__main__` namespace. Unpickling it in a *different* notebook's kernel fails
(`AttributeError: Can't get attribute 'MICE_CA19_9Imputer' on <module '__main__'>`) unless
an identical class is redefined in the new kernel's `__main__` first — confirmed by actually
hitting the error before fixing it. `fusion.ipynb` copies the class verbatim from
`clinical_final_fit.ipynb`, same pattern that notebook itself used when copying it forward
from `clinical_imputer_benchmark.ipynb`, purely so `joblib.load` can resolve the class, not
to refit or redefine its logic.

**Imaging:** `ResNet50UNet` reconstructed from the real, importable `imaging/models.py`
module (same pattern `imaging_evaluation.ipynb`/`imaging_confound_check.ipynb` use), weights
loaded from `model.pt` with a `torch.compile` prefix-leakage check, `calibrator.pkl` loaded,
`box_size=320` read from `pod_training_run.json` rather than hardcoded a second time.
`known_limitations.confound_check_summary` is loaded once into `IMAGING_CAVEAT` and reused
verbatim everywhere downstream.

---

## Interface

```python
run_tabular_branch(patient_features: dict) -> float
```
Impute → XGBoost → Platt calibrator. `patient_features` keyed by
`["creatinine", "LYVE1", "REG1B", "TFF1", "plasma_CA19_9", "age", "sex"]`;
`plasma_CA19_9` may be `NaN`.

```python
run_imaging_branch(image_tensor: torch.Tensor) -> float          # (3, 320, 320)
run_imaging_branch_volume(slice_stack: torch.Tensor) -> dict     # (N, 3, 320, 320)
```
Single-slice path is the model's native granularity (unchanged, always available).
Volume path batches at 32 (matching training batch size — a ~250-slice scan never hits the
GPU in one go), returns `{"patient_score", "per_slice_proba", "n_slices", "aggregation"}` —
the full per-slice vector is returned deliberately, since a dashboard will want to render
*where* along the scan the model reacted, and it's also the honest way to see how flat the
model's response actually is (per the diagnostic above).

```python
fuse(imaging_input=None, tabular_input=None) -> dict
```
`imaging_input` may be a single slice `(3, BOX, BOX)` or a whole volume `(N, 3, BOX, BOX)` —
dispatched on tensor rank, and which one was used is recorded explicitly in the result's
`imaging_granularity` field, never left implicit. Returns:

```python
{
  "imaging_calibrated_proba": float | None,
  "imaging_granularity": "single-slice" | "volume (mean over slices)" | None,
  "imaging_n_slices": int | None,
  "tabular_calibrated_proba": float | None,
  "fused_score": float,
  "mode": "fused (both modalities)" | "single-modality (imaging only)" | "single-modality (tabular only)",
}
```

**If only one modality is supplied, `fused_score` is that branch's own calibrated score,
returned untouched** — never silently degraded or combined with a fabricated placeholder for
the missing branch. Raises `ValueError` if neither is supplied, rather than returning
something meaningless.

---

## Smoke Test — 4 Real Input Paths

Real data throughout, no placeholders: patient `S10` (real, `plasma_CA19_9` missing —
exercises the imputer path), MSD patient `pancreas_001` slice 0 (single-slice path), and
`pancreas_001`'s complete 275-slice scan in `slice_index` order (volume path).

| Path | `fused_score` | `mode` | `imaging_granularity` |
|---|---|---|---|
| Imaging (single slice) | 0.9947 | single-modality (imaging only) | single-slice |
| Imaging (whole volume) | 0.9913 | single-modality (imaging only) | volume (mean over slices), n=275 |
| Tabular only | 0.1301 | single-modality (tabular only) | n/a |
| Volume + tabular | 0.4746 | fused (both modalities) | volume (mean over slices) |
| Neither | — | — | correctly raised `ValueError` |

Per-slice profile for `pancreas_001`'s full scan: min=0.163, mean=0.991, max=0.995,
std=0.050 — a live instance of the near-flat within-patient response the diagnostic
describes above.

**The tabular and imaging samples in the "volume + tabular" row are independently real but
arbitrarily paired** — this does not imply they're the same patient (no patient in this
project has both a CT scan and a urine sample). Same fabricated-pairing framing used in
`fusion_evaluation.ipynb`'s worked examples: exercises the code path, reports no accuracy.

All assertions passed: every probability in `[0, 1]`; single-modality paths pass the
branch's own score through untouched; the fused score matches `W_IMAGING`/`W_TABULAR`
exactly; granularity and slice count are recorded correctly.

---

## `checkpoints/fusion/final/model_card.json`

This stage's own model card, separate from each branch's `model_card.json`. Carries no
trained weights of its own — the combination and aggregation rules are fixed/hand-set, not
fitted, so this directory holds only the JSON, no `model.pt`/`model.pkl`. Fields:

- `combination_rule` — formula, weights, and the reasoning above.
- `imaging_slice_aggregation_rule` — the rule, its reasoning (including the confound
  fingerprint interpretation), a pointer to the measurement CSV, and the full measurement
  table embedded inline (7 rows).
- `single_modality_behavior` — states the pass-through guarantee explicitly.
- `no_joint_benchmark_metric` — restates why no joint metric exists, and explicitly flags
  that the aggregation measurement is single-branch (imaging-only) and must not be read as a
  joint/fused benchmark.
- `branch_model_cards` — pointers to `checkpoints/{clinical,imaging}/final/model_card.json`.
- `imaging_known_limitations_caveat` — the confound-check finding, quoted verbatim (see
  `docs/Fusion_Evaluation_documentation.md` for the full text), so it can't get lost one
  level up the pipeline from where it was originally disclosed.
- `smoke_test` — the real results from all 4 paths above, embedded so the card is a live
  artifact of an actual run, not a hand-written claim.

Verified after writing: read back from disk (not the in-memory dict) and diffed against the
in-notebook `IMAGING_CAVEAT`/weights/measurement — all matched exactly.

---

## Config Changes

`src/utils/config.py` gained two new constants (this notebook was the first consumer of
either):

- `CHECKPOINTS_FUSION_FINAL_DIR = checkpoints/fusion/final/` — commented to explain why
  there's no sibling `model.pt`/`model.pkl` constant (fixed rule, not a fitted artifact).
- `FUSION_IMAGING_AGGREGATION_PATH = results/fusion/imaging_slice_vs_volume_aggregation.csv`
  — commented to explain why it lives under `results/fusion/`, not `results/imaging/` (a
  fusion/inference-time decision measured from imaging's OOF file, not a retrofit of the
  imaging branch's own training/evaluation notebooks).

Both added to `ensure_dirs()`.

---

## Artifacts

- `checkpoints/fusion/final/model_card.json`
- `results/fusion/imaging_slice_vs_volume_aggregation.csv`

## What This Notebook Does Not Do

- Does not retrain or refit anything — loads `checkpoints/*/final/` only.
- Does not fit weights or the aggregation rule against paired data — no paired ground truth
  exists in this project, and both rules are stated as fixed, hand-set judgment calls.
- Does not resolve the imaging confound — volume aggregation improves the measured numbers
  but is shown above to be partly a re-expression of the same finding, not a fix for it.
- Does not decide how the dashboard should surface these scores — only exposes the interface
  (`fuse()`, `run_imaging_branch_volume()`) it will call.
