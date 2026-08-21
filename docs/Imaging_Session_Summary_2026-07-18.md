# Imaging Model Review & Round 6 Remediation — Session Summary

**Date:** 2026-07-18
**Scope:** Full record of one working session — reviewing the promoted imaging model's results, surfacing an unresolved confound-check finding, attempting a fix, and the decision taken based on the outcome. This doc exists so the whole session is traceable from one place; the deep technical history of Rounds 1–6 lives in `Imaging_Confound_Check_documentation.md`, which this doc cross-references rather than duplicates in full.

---

## 1. Starting point: the promoted final model

`checkpoints/imaging/final/model.pt` — a `resnet50_unet` (71,876,484 params), fit type `final_all_data_no_folds`: trained on all 361 patients / 90,693 rows with **no held-out validation set**, so it has no standalone eval of its own. Its epoch count (6) was derived from the mean `best_epoch` across the 5-fold CV candidate run it was based on.

Because the final fit has no eval data, its performance is only knowable via the 5-fold cross-validation run for the same architecture/hyperparameters — `checkpoints/imaging/candidates/resnet50_unet/model_card.json` — which is what's reported below.

## 2. Cross-validation performance (backing the live/promoted final model)

This is the **Round 5** state (padding-shortcut fix applied, verified) — see §4 for why it isn't the raw/unaugmented numbers.

| Metric | Mean across 5 folds |
|---|---|
| ROC AUC | 0.993 |
| Precision | 0.992 |
| Recall | 0.963 |
| Specificity | 0.970 |
| F1 | 0.977 |
| Dice (segmentation) | 0.396 |
| IoU (segmentation) | 0.283 |

| Fold | AUC | Precision | Recall | F1 | Dice | IoU |
|---|---|---|---|---|---|---|
| 0 | 0.998 | 0.978 | 0.995 | 0.986 | 0.522 | 0.373 |
| 1 | 0.982 | 0.997 | 0.930 | 0.962 | 0.061 | 0.032 |
| 2 | 0.992 | 0.999 | 0.916 | 0.956 | 0.340 | 0.242 |
| 3 | 0.997 | 0.998 | 0.979 | 0.989 | 0.520 | 0.374 |
| 4 | 0.998 | 0.988 | 0.995 | 0.991 | 0.539 | 0.396 |

**Read:** detection (classification) performance is strong and consistent (AUC 0.98–1.0, F1 0.96–0.99). Segmentation (Dice/IoU) is weaker and highly variable — fold 1 collapses to Dice 0.06 while others reach ~0.52–0.54.

## 3. The confound-check finding that prompted this session

Pulled from `results/imaging/confound_check_occlusion_sensitivity.csv` and `confound_check_all_methods_metrics.csv` (the state as of session start, i.e. Round 5's numbers):

**Occlusion sensitivity** (65 slices, 5 folds — blank a region, measure the change in detection logit):

| Region occluded | Mean Δ logit | % of slices where confidence dropped |
|---|---|---|
| Tumor | 0.066 | 55.4% |
| Padding | −0.041 (n=58) | — |
| Control (random tissue) | 0.151 | 75.4% |

Occluding the tumor moved confidence *less* than occluding random tissue — the opposite of what a model genuinely reasoning about the tumor should show.

**Saliency-overlap sweep** (6 attribution methods, 65 slices each, vs. the real tumor mask):

| Method | Overlap precision | Overlap recall | Enrichment vs. chance |
|---|---|---|---|
| Integrated Gradients | 1.5% | 31.0% | 1.55× |
| EigenCAM | 0.08% | 4.2% | 0.09× |
| GradCAM | 0.0% | 0.0% | 0× |
| GradCAM++ | 0.0% | 0.0% | 0× |
| HiResCAM | 0.0% | 0.0% | 0× |
| XGradCAM | 0.0% | 0.0% | 0× |

Every CAM-family method showed **zero** overlap with the tumor mask.

## 4. Discovery: this was already a known, actively-investigated issue

`docs/Imaging_Confound_Check_documentation.md` documented a prior investigation (Rounds 1–5, same day, earlier in the project timeline):

- **Root concern:** `dataset` (MSD vs. NIH) and `class` (cancer vs. healthy) are perfectly confounded — the model could be learning "which scanner produced this image" instead of real pathology.
- **Round 4 (occlusion, direct causal test):** found the model relied significantly **more** on BOX=320 packing's synthetic zero-padding than on random tissue (2.03× control, p=0.044), and significantly **less** on the real tumor than on random tissue (0.47× control, p=0.0075).
- **Round 5 (fix + verify):** added random-resized-crop augmentation (`SliceCacheDataset(..., augment=True)`) so the padding fraction varies per sample/epoch. Retrained, re-ran the occlusion test:

| Condition | Before (Round 4) | After (Round 5) |
|---|---|---|
| Tumor occlusion vs. control | 0.47×, p=0.0075 | 0.62×, p=0.0065 (still significant) |
| Padding occlusion vs. control | 2.03×, p=0.044 (significant) | **1.08×, p=0.917 (fixed)** |

**Padding shortcut: fixed and verified.** **Tumor under-reliance: improved numerically, but still statistically significant and unresolved** — an open problem, not a new finding, at the point this session started.

## 5. This session: Round 6 remediation attempt

You asked what could be done about the remaining tumor under-reliance, and then asked to go ahead and implement the doc's top-ranked "if resuming" option, with full archiving/documentation at each step.

### Fix implemented
Added **mask-preserving random erase** to `imaging/slice_cache_dataset.py`: for MSD rows only (`has_mask=True`), with probability `MASK_ERASE_PROB=0.4`, randomly blank a 5–15%-area patch of tissue that never overlaps the real tumor mask, filled with the image's own median intensity. Goal: make every region except the tumor an unreliable, randomly-vanishing training cue, so the tumor becomes the one signal that's always present.

- Unit-tested in isolation (200 trials — tumor mask never touched) and through the full dataset `__getitem__` path before touching the training pipeline.
- Archived the pre-change state first: `checkpoints/imaging/candidates_5fold_padding_fix_archive/`, `checkpoints/imaging/final_padding_fix_archive/`, `results/imaging/archive/confound_check_padding_fix_archive/`, `results/imaging/archive/oof_predictions_padding_fix_archive.csv`, `outputs/eval/imaging/archive_padding_fix/`, `imaging/imaging_confound_check_padding_fix_archive.ipynb`.
- Retrained `resnet50_unet`, all 5 folds (`baseline_cnn` untouched, not the winning candidate) — ~90 min on the RTX 6000 Ada pod.
- Re-ran the identical occlusion confound check against the new checkpoints.

### Expected result
Tumor-occlusion sensitivity should rise from 0.62× toward/above 1× control, while padding sensitivity should stay near the fixed 1.08× level.

### Actual result

**Detection/segmentation metrics, Round 5 vs. Round 6 (mean across 5 folds):**

| Metric | Round 5 (padding-fix only) | Round 6 (+ mask-erase) |
|---|---|---|
| ROC-AUC | 0.9934 | 0.9928 |
| Recall | 0.9628 | 0.9790 |
| Precision | 0.9920 | 0.9832 |
| Specificity | 0.9696 | 0.9359 |
| F1 | 0.9768 | 0.9811 |
| Dice / IoU | 0.396 / 0.283 | 0.369 / 0.262 |
| Mean stopped epoch | 8.0 | 5.4 |

Roughly a wash — no meaningful change.

**Occlusion sensitivity, Round 5 vs. Round 6 (same 59 MSD slices, same methodology):**

| Condition | Round 5 (after padding fix) | Round 6 (after mask-erase) |
|---|---|---|
| Tumor occlusion vs. control | 0.62×, p=0.0065 (significant) | **0.58×, p=0.0148 (still significant — unchanged)** |
| Padding occlusion vs. control | 1.08×, p=0.9168 (fixed) | **2.86×, p=0.1353 (not significant, but the point estimate is now larger than the original 2.03× problem)** |

**Six-method attribution sweep** (re-run for completeness, same 59 slices):

| Method | Round 5 enrichment | Round 6 enrichment |
|---|---|---|
| Integrated Gradients | 1.55× | 1.76× |
| EigenCAM | 0.09× | 0.17× |
| GradCAM / GradCAM++ / HiResCAM / XGradCAM | ~0.00× | ~0.00–0.008× |

Same mixed-bag pattern as before — not informative for this specific question.

### Verdict

**The fix did not achieve its goal, and its effect on the already-fixed padding issue is genuinely ambiguous rather than clearly safe:**

1. Tumor under-reliance is statistically unchanged (0.58× vs. 0.62×, both significantly below 1×) — the augmentation did not make the model lean on the tumor more.
2. Padding sensitivity's point estimate rose substantially (1.08× → 2.86×) without reaching significance at this sample size (n=52) — not proof the padding shortcut returned, but not proof it didn't either.

## 6. Decision and current state

**Decision (yours, after reviewing the above):** revert. Given no demonstrated improvement and a not-fully-ruled-out risk on the padding axis, `checkpoints/imaging/final/` and `checkpoints/imaging/candidates/` were restored byte-for-byte to the verified Round 5 state.

- **Round 6 was not discarded** — fully archived at `checkpoints/imaging/candidates_5fold_mask_erase_archive/`, `results/imaging/archive/confound_check_mask_erase_archive/`, `results/imaging/archive/oof_predictions_mask_erase_archive.csv`, `outputs/eval/imaging/archive_mask_erase/`, `imaging/imaging_confound_check_mask_erase_archive.ipynb`.
- **The new code stays in the codebase** — `_apply_mask_preserving_erase` in `imaging/slice_cache_dataset.py`, with its docstring recording this outcome, available for a future attempt with different hyperparameters or combined with a different approach.
- **`docs/Imaging_Confound_Check_documentation.md`** now has a full "Round 6" section (this same material, in the project's established Round-by-Round format) and an updated "if resuming" priority list — **attention-supervision auxiliary loss is now the top-ranked next step**, since two independent augmentation-only attempts (crop, then erase) have both failed to move the tumor-reliance number.

**Net effect on the promoted model: none.** `checkpoints/imaging/final/model.pt` is exactly the same Round 5 model as before this session started. What changed is that a plausible next fix was tried, honestly evaluated, found not to work, and the entire trail — code, checkpoints, results, docs — is preserved rather than lost.

## 7. Where everything lives (quick index)

| What | Path |
|---|---|
| Live/promoted final model | `checkpoints/imaging/final/model.pt` |
| Live 5-fold CV backing it | `checkpoints/imaging/candidates/resnet50_unet/` |
| Full confound-check history (Rounds 1–6, deep detail) | `docs/Imaging_Confound_Check_documentation.md` |
| This session's summary | `docs/Imaging_Session_Summary_2026-07-18.md` (this file) |
| Round 6 archived checkpoints | `checkpoints/imaging/candidates_5fold_mask_erase_archive/` |
| Round 6 archived results | `results/imaging/archive/*_mask_erase_archive*` |
| Round 6 archived notebook | `imaging/imaging_confound_check_mask_erase_archive.ipynb` |
| Mask-preserving erase code (unused by live model) | `imaging/slice_cache_dataset.py` (`_apply_mask_preserving_erase`) |
