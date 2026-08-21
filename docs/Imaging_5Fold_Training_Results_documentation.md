# Imaging Candidate Training — 5-Fold Results (Superseded for resnet50_unet — see note)

**Run date:** 2026-07-17
**Status:** Supersedes the 3-fold run archived in `Imaging_3Fold_Training_Results_documentation.md`. **`resnet50_unet`'s numbers below are themselves superseded** by a subsequent augmented retrain — see `Imaging_Confound_Check_documentation.md`, "Round 5", for why (a real, causally-confirmed shortcut was found and partially fixed) and for the current live detection metrics. `baseline_cnn`'s numbers below remain current and unchanged. The checkpoints/OOF predictions this doc describes are archived at `checkpoints/imaging/candidates_5fold_no_augmentation_archive/` and `results/imaging/archive/oof_predictions_5fold_no_augmentation.csv`.

## Why 5 folds instead of 3

The original `FOLDS_TO_RUN = [0, 1, 2]` cap was a schedule constraint from the local RTX 3050 laptop era, not a methodological choice. On the rented RTX 6000 Ada pod the 2 extra folds cost ~27 extra minutes and well under $1, so the cap was lifted. The 3-fold run's OOF predictions covered only 60.7% of patients (folds 0–2) — fold 3 and 4 patients were never held out and evaluated. This run gives every one of the 361 patients exactly one out-of-fold prediction (181,386 OOF rows total, both candidates × all 5 folds).

## Setup

Same as the 3-fold run except `FOLDS_TO_RUN`: RunPod RTX 6000 Ada, BOX=320 cache, `SLICE_STRIDE=3`, `MAX_EPOCHS=15`, `PATIENCE=3`, `BATCH_SIZE=32`, `torch.compile` enabled for `resnet50_unet` only. See `Imaging_3Fold_Training_Results_documentation.md` for the pre-flight sweep that produced these settings — unchanged here, only re-validated.

**Total training wall-clock:** resnet50_unet 46.2 min across 5 folds (avg 9.23 min/fold, avg 5.4 epochs to early-stop); baseline_cnn 24.2 min across 5 folds (avg 4.85 min/fold, avg 11.2 epochs to early-stop). Total training ≈ 70.4 min; total run wall-clock (including the redundant pre-flight re-validation baked into the notebook) ≈ 78 min ≈ **$1.00** at $0.77/hr.

## Results

### resnet50_unet (71,876,484 params)

| Fold | Stopped/Best Epoch | ROC-AUC | Recall | Precision | Specificity | F1 | Dice | IoU |
|---|---|---|---|---|---|---|---|---|
| 0 | 3 / 0 | 0.9963 | 0.9967 | 0.9457 | 0.8000 | 0.9705 | 0.323 | 0.238 |
| 1 | 3 / 0 | 0.9886 | 0.9588 | 0.9800 | 0.9206 | 0.9693 | 0.191 | 0.118 |
| 2 | 9 / 6 | 0.9878 | 0.9706 | 0.9545 | 0.8269 | 0.9625 | 0.424 | 0.306 |
| 3 | 5 / 2 | 0.9971 | 0.9953 | 0.9507 | 0.7965 | 0.9725 | 0.539 | 0.385 |
| 4 | 7 / 4 | 0.9993 | 0.9949 | 0.9912 | 0.9631 | 0.9931 | 0.497 | 0.366 |
| **Mean** | **5.4 / 2.4** | **0.9938** | **0.9833** | **0.9644** | **0.8614** | **0.9736** | **0.395** | **0.282** |

### baseline_cnn (97,761 params)

| Fold | Stopped/Best Epoch | ROC-AUC | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|---|
| 0 | 14 / 12 | 0.9921 | 0.9947 | 0.9313 | 0.7433 | 0.9619 |
| 1 | 7 / 4 | 0.9623 | 0.8744 | 0.9725 | 0.8992 | 0.9208 |
| 2 | 14 / 11 | 0.9861 | 0.9966 | 0.8705 | 0.4458 | 0.9293 |
| 3 | 10 / 7 | 0.9799 | 0.6890 | 0.9986 | 0.9963 | 0.8154 |
| 4 | 11 / 8 | 0.9890 | 0.8681 | 0.9961 | 0.9858 | 0.9277 |
| **Mean** | **11.2 / 8.4** | **0.9819** | **0.8845** | **0.9538** | **0.8141** | **0.9110** |

## What changed vs. the 3-fold run

- **resnet50_unet's segmentation performance improved** (mean Dice 0.323 → 0.395, mean IoU 0.222 → 0.282), driven by strong Dice on folds 3 and 4 (0.539, 0.497) — these patient subsets segment noticeably better than folds 0–2.
- **resnet50_unet's recall lead over baseline_cnn narrowed slightly but stayed large and, more importantly, far more stable**: 98.3% vs 88.5% mean (a 9.9-point gap, vs 13.3 points at 3 folds). resnet50_unet's recall stays in a tight 95.9–99.7% band across all 5 folds; baseline_cnn's swings from 68.9% (fold 3) to 99.7% (fold 2).
- **baseline_cnn's fold-to-fold instability is now much more visible with 5 folds.** Fold 2: recall 0.997, but specificity collapses to 0.446 (TN=1,669 vs FP=2,075 — it flags almost everything as cancer). Fold 3: precision 0.999, specificity 0.996, but recall collapses to 0.689 (misses ~31% of actual cancer slices). These are not small variations — baseline_cnn's operating point (its precision/recall trade-off) swings dramatically depending on which patients are held out, while resnet50_unet's stays consistent across all 5 folds (recall 95.9–99.7%, precision 94.6–99.1%). This instability, not just the mean numbers, is itself an argument against baseline_cnn for a clinical-facing pipeline.
- **resnet50_unet plateaued immediately (epoch 0) in 2 of 5 folds** (folds 0 and 1: `best_epoch=0`), i.e. the pretrained ImageNet encoder needed essentially no fine-tuning to reach its best validation AUC on those folds, then started overfitting. Not concerning on its own, but worth noting in the write-up as a sign the pretrained features are doing most of the work.

## Winner: resnet50_unet (unchanged from the 3-fold run, now on stronger evidence)

Same reasoning as the 3-fold run, reinforced by full patient coverage: recall (98.3% vs 88.5%) and its stability across folds are the deciding factors — a missed cancer slice is clinically costlier than a false alarm, and resnet50_unet is both more sensitive and far more consistent fold-to-fold than baseline_cnn. baseline_cnn wins on precision/specificity but at the cost of wild fold-to-fold swings in its operating point, which is a liability, not just a tradeoff, for a model meant to generalize to unseen patients. resnet50_unet remains the only candidate producing the segmentation output the project's risk-score calculation depends on.

**Fold 1 remains the hardest fold for resnet50_unet** (lowest ROC-AUC, 0.9886) and **fold 3 is baseline_cnn's worst fold** (recall collapse to 0.689) — together with fold 2's specificity collapse, baseline_cnn does not have one "hard fold," it has different failure modes in different folds, which is itself informative about its lack of robustness.

## Live artifacts (this 5-fold run)

- `checkpoints/imaging/candidates/resnet50_unet/{fold_0..fold_4}.pt` + `model_card.json`
- `checkpoints/imaging/candidates/baseline_cnn/{fold_0..fold_4}.pt` + `model_card.json`
- `results/imaging/oof_predictions.csv` (181,386 rows — both candidates × 5 folds × all 361 patients)
- `imaging/train_segmentation_detection.ipynb` (current live notebook, `FOLDS_TO_RUN = list(range(5))`)

All 10 checkpoints re-verified clean (no `_orig_mod.` compile-prefix leakage) after this run.

Prior 3-fold artifacts remain archived and untouched at `checkpoints/imaging/candidates_3fold_archive/`, `results/imaging/archive/oof_predictions_3fold.csv`, and `imaging/train_segmentation_detection_3fold_archive.ipynb`.
