# Imaging Branch — Evaluation, Promotion, and Calibration

**Source:** `src/imaging/imaging_evaluation.ipynb`.
**Purpose:** compare `resnet50_unet` vs. `baseline_cnn` on the real 5-fold CV results,
verify + calibrate the already-promoted final model, and disclose the confound-check
finding directly in the artifact that gets deployed — not just in a separate doc that a
future reader might not see.
**Run date:** 2026-07-19.

---

## Candidate Comparison

Built from each candidate's `checkpoints/imaging/candidates/<name>/model_card.json` —
**not recomputed from raw predictions.** Those per-fold metrics were already produced by
the same `utils.metrics.detection_metrics`/`dice_score`/`iou_score` functions during
training; recomputing them here would risk a subtly different number for no reason.

| scheme | model | n_folds | precision | recall | specificity | f1 | roc_auc | dice | iou |
|---|---|---|---|---|---|---|---|---|---|
| 5-Fold CV | **resnet50_unet** | 5 | 0.9920 ± 0.0091 | 0.9628 ± 0.0372 | 0.9696 ± 0.0333 | 0.9768 ± 0.0164 | 0.9934 ± 0.0069 | 0.396 ± 0.204 | 0.283 ± 0.153 |
| 5-Fold CV | baseline_cnn | 5 | 0.9647 ± 0.0453 | 0.9073 ± 0.0789 | 0.8584 ± 0.1913 | 0.9320 ± 0.0293 | 0.9840 ± 0.0076 | NaN | NaN |

Pooled confusion matrix — `resnet50_unet`: TN=18,031 FP=585 FN=2,679 TP=69,398.
`baseline_cnn`: TN=15,982 FP=2,634 FN=6,829 TP=65,248. `NaN` dice/iou for `baseline_cnn` is
correct by design (no segmentation head), not missing data.

Written to `results/imaging/model_comparison.csv`; ROC/PR/reliability figures (pooled OOF,
**uncalibrated**) saved to `outputs/eval/imaging/candidate_comparison.png`.

**Winner: `resnet50_unet`** — human-adjudicated, same pattern as `clinical_final_fit.ipynb`'s
XGBoost pick, not re-derived by formula. Higher and far more fold-stable recall (the
clinically prioritised metric — a missed cancer slice is costlier than a false alarm) than
`baseline_cnn`, whose recall/specificity swing dramatically fold-to-fold (see
`docs/Imaging_5Fold_Training_Results_documentation.md` for the full per-fold breakdown).
Only `resnet50_unet` produces the segmentation output the project's derived risk-score
(tumour/gland pixel ratio) structurally depends on.

---

## Promotion: `checkpoints/imaging/final/`

`model.pt` was **not** produced by this notebook — it's the all-data, no-folds final fit
done on the RunPod pod, mirroring `clinical_final_fit.ipynb`'s "what's the single best
version of it, using every patient we have?" philosophy, adapted for a deep model that
needs GPU compute and has no natural "refit trivially" path.

### Final-fit provenance (`checkpoints/imaging/final/pod_training_run.json`)

| Field | Value |
|---|---|
| Fit type | `final_all_data_no_folds` |
| Patients / rows | 361 / 90,693 (all of them — no held-out fold) |
| Epochs | **6** (fixed count, not early-stopped — see reasoning below) |
| Batch size | 32 |
| `augment` | `True` (the Round 5 padding-shortcut fix — see confound check section) |
| `slice_stride` | 3 |
| Optimizer | AdamW, lr=1e-4 |
| `torch.compile` | enabled |
| Box size | 320 |
| Params | 71,876,484 |

**Why a fixed epoch count instead of early stopping:** a no-folds fit has no held-out
validation set, so the patience-based early stopping used everywhere else in this project
isn't available. Instead: `checkpoints/imaging/candidates/resnet50_unet/model_card.json`'s
per-fold `best_epoch` values were `[4, 0, 2, 12, 8]`, mean 5.2 → `round(5.2) = 5` (0-indexed)
→ **6 total epochs** (indices 0–5). Per-epoch train losses were logged as a sanity check
that training progressed normally (steadily decreasing det/seg loss across all 6 epochs) —
explicitly **not** a performance estimate, since there's no held-out data in that run to
evaluate against.

### Verification performed in `imaging_evaluation.ipynb`

- **Checkpoint integrity:** loads cleanly into a fresh `ResNet50UNet()` — 71,876,484
  params, zero `torch.compile` `_orig_mod.`-prefix leakage (same check every CV checkpoint
  already passed).
- **In-sample sanity check** (64 random rows — NOT a performance estimate, since this final
  model was fit on all data and there is no true out-of-sample set left to score it on):
  mean P(cancer | true=cancer) = **0.9963**, mean P(cancer | true=healthy) = **0.0159**, all
  finite. Sane, well-separated, non-degenerate output.
- Real performance is only knowable via the 5-fold CV run backing this
  architecture/hyperparameters (the candidate comparison table above).

### Calibration

**Platt scaling** (`LogisticRegression` on raw sigmoid probability → calibrated
probability), fit on all **90,693** `resnet50_unet` OOF rows (361 unique patients) from
`results/imaging/oof_predictions.csv` — predictions the final model never trained on, never
the final model's own in-sample output, which would be circular. Isotonic regression was
not used: it typically needs thousands of samples *per class* to avoid staircase
overfitting, and while there are 90,693 rows, the effective unit of novel information is
closer to 361 unique patients — favouring the simpler, more stable parametric option, same
reasoning as the clinical branch. Saved as `checkpoints/imaging/final/calibrator.pkl`.

### `model_card.json`

Written to `checkpoints/imaging/final/model_card.json`, recording: architecture, training
provenance, the comparison metrics that justified the choice, calibrator choice + reasoning,
the in-sample sanity check (explicitly labeled as not a performance estimate), and —
critically — the confound-check finding as a disclosed `known_limitations` field (see next
section). All 4 expected files confirmed present: `model.pt` (287,854,371 bytes),
`calibrator.pkl` (831 bytes), `model_card.json` (4,579 bytes), `pod_training_run.json`
(3,413 bytes).

---

## Confound Check — Summary and Disclosure

**Not re-run in this notebook.** The full investigation (6 attribution methods, then
causal occlusion testing, across 6 rounds) was already done in
`imaging/imaging_confound_check.ipynb` — see `docs/Imaging_Confound_Check_documentation.md`
and `docs/Imaging_Session_Summary_2026-07-18.md` for the complete methodology and history.
This notebook's confound-check cell instead **recomputes its summary numbers live** from
`results/imaging/confound_check_occlusion_sensitivity.csv` every time it runs, so the
summary can't silently drift out of sync with the actual result file. As executed:

| | Recomputed live | Matches documented Round 5 result |
|---|---|---|
| Tumor / control ratio | 0.62x | ✓ exact match |
| Padding / control ratio | 1.08x | ✓ exact match |

**Verdict, disclosed directly in `model_card.json`'s `known_limitations`:**
- **Padding shortcut** (model relied *more* on synthetic BOX=320 padding than the real
  tumour): **fixed and verified** (Round 5) — 2.03x control (p=0.044) → 1.08x
  (p=0.917, statistically indistinguishable from the control).
- **Tumour under-reliance** (model relies *less* on the real tumour than a random patch of
  ordinary tissue): **still open, unresolved**. Two remediation attempts were made — the
  crop augmentation fixed the padding shortcut but not this; a second attempt
  (mask-preserving random erase, Round 6) also did not help (0.58x control, p=0.0148,
  statistically unchanged from Round 5's 0.62x) and was reverted. This is not a new finding
  introduced by promotion — it is the same open issue documented in Round 5/6, carried
  forward explicitly into the promoted model's own card rather than left to be discovered
  only by reading a separate document.

**Practical consequence:** `resnet50_unet`'s detection metrics (ROC-AUC 0.993, recall 0.963)
are strong and reproducible, but are **not yet confirmed evidence of tumour-specific
pixel-level reasoning**. Any use of this model or its detection metrics in the project
write-up should carry this caveat explicitly, not present the numbers alone.

---

## Artifacts

- `results/imaging/model_comparison.csv`
- `outputs/eval/imaging/candidate_comparison.png`
- `checkpoints/imaging/final/model.pt`, `calibrator.pkl`, `model_card.json`,
  `pod_training_run.json`
