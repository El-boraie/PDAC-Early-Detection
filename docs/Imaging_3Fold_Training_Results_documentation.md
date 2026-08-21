# Imaging Candidate Training — 3-Fold Results (Archived)

**Run date:** 2026-07-17
**Status:** Superseded by the 5-fold run (see `Imaging_5Fold_Training_Results_documentation.md` once that lands). Kept here as the historical record of the first full run on the rented GPU.

## Setup

- **Hardware:** RunPod RTX 6000 Ada, 48GB VRAM (moved from a local RTX 3050 laptop, 4GB VRAM — the laptop measured 37.81 min/epoch and was too slow for the deadline).
- **Data:** packed slice cache at `data/processed/cache/`, BOX=320, 90,693 slices / 361 patients, manifest's existing `split` column (5 pre-computed folds: fold0–fold4).
- **Folds trained:** `FOLDS_TO_RUN = [0, 1, 2]` — 3 of the 5 available folds, per the original schedule-driven cap. This means the OOF predictions from this run cover only folds 0–2 (~60.7% of patients, 55,056 / 90,693 rows) — folds 3–4 patients were never held out and evaluated in this run.
- **Config:** `SLICE_STRIDE=3` (rotating offset), `MAX_EPOCHS=15`, `PATIENCE=3` (on fixed-subset val AUC), `BATCH_SIZE=32`, `torch.compile` enabled for `resnet50_unet` only (validated in pre-flight: +6.2% throughput at batch=32, no measurable benefit assumed for `baseline_cnn` since it was untested there).
- **Pre-flight (batch-size sweep, resnet50_unet, one real epoch each):** batch=16 → 190.9 img/s (2.11 min/epoch); batch=32 → 201.7 img/s (1.99 min/epoch); batch=64 → 193.0 img/s (2.08 min/epoch); batch=32+compile → 214.2 img/s (1.88 min/epoch). batch=32+compile selected. Peak VRAM never exceeded ~28GB of the 48GB available.
- **Total training wall-clock (this run):** resnet50_unet 27.1 min across 3 folds (avg 9.03 min/fold, avg 5.0 epochs to early-stop); baseline_cnn 13.8 min across 3 folds (avg 4.59 min/fold, avg 10.7 epochs to early-stop). Total run cost ≈ $0.69 at $0.77/hr (RunPod Secure Cloud on-demand estimate).

## Results

### resnet50_unet (71,876,484 params, pretrained ImageNet ResNet-50 encoder + U-Net decoder + detection head)

| Fold | Stopped/Best Epoch | ROC-AUC | Recall | Precision | Specificity | F1 | Dice | IoU |
|---|---|---|---|---|---|---|---|---|
| 0 | 4 / 1 | 0.9983 | 0.9965 | 0.9698 | 0.8914 | 0.9830 | 0.348 | 0.267 |
| 1 | 6 / 3 | 0.9773 | 0.9617 | 0.9547 | 0.8144 | 0.9582 | 0.239 | 0.147 |
| 2 | 5 / 2 | 0.9897 | 0.9690 | 0.9629 | 0.8606 | 0.9660 | 0.383 | 0.252 |
| **Mean** | **5.0 / 2.0** | **0.9884** | **0.9757** | **0.9625** | **0.8555** | **0.9691** | **0.323** | **0.222** |

### baseline_cnn (97,761 params, from-scratch CNN, detection head only, no segmentation)

| Fold | Stopped/Best Epoch | ROC-AUC | Recall | Precision | Specificity | F1 |
|---|---|---|---|---|---|---|
| 0 | 14 / 12 | 0.9912 | 0.8987 | 0.9956 | 0.9862 | 0.9447 |
| 1 | 7 / 4 | 0.9652 | 0.7186 | 0.9918 | 0.9757 | 0.8333 |
| 2 | 11 / 8 | 0.9805 | 0.9124 | 0.9823 | 0.9386 | 0.9461 |
| **Mean** | **10.7 / 8.0** | **0.9790** | **0.8432** | **0.9899** | **0.9668** | **0.9080** |

`segmentation_metrics: null` for every baseline_cnn fold — correct by design (no segmentation head), not missing data.

## Winner: resnet50_unet

The ROC-AUC gap (0.0095) alone is below the notebook's stated "too close to call" threshold (0.02), but recall diverges sharply: **97.6% vs 84.3% mean, a 13.3-point gap**, and resnet50_unet's recall is far more stable across folds (96.2–99.7% band) than baseline_cnn's (71.9–91.2%). In a cancer-detection context, recall/sensitivity is the clinically load-bearing metric — a missed cancer slice is far costlier than a false alarm, and baseline_cnn misses roughly 1 in 6 actual cancer slices where resnet50_unet misses about 1 in 40.

baseline_cnn wins cleanly on precision (0.9899 vs 0.9625) and specificity (0.9668 vs 0.8555) — a real, honestly-stated tradeoff: baseline_cnn is the more conservative model, resnet50_unet the more sensitive one.

Beyond the detection metrics: only resnet50_unet produces segmentation output, which the project's derived risk-score (tumour/gland pixel ratio) structurally depends on. baseline_cnn cannot support that downstream feature regardless of its detection performance.

**Fold 1 was the hardest fold for both candidates** — lowest ROC-AUC for each (0.9773 resnet50_unet, 0.9652 baseline_cnn), and where baseline_cnn's recall cratered to 71.9%. Worth a closer look at what's different about that patient subset if time allows.

## Archived artifacts (this 3-fold run)

- `checkpoints/imaging/candidates_3fold_archive/resnet50_unet/{fold_0,fold_1,fold_2}.pt` + `model_card.json`
- `checkpoints/imaging/candidates_3fold_archive/baseline_cnn/{fold_0,fold_1,fold_2}.pt` + `model_card.json`
- `results/imaging/archive/oof_predictions_3fold.csv` (110,112 rows — both candidates × 3 folds)
- `imaging/train_segmentation_detection_3fold_archive.ipynb` (full executed notebook, outputs preserved as-is from this run)

Live paths (`checkpoints/imaging/candidates/`, `results/imaging/oof_predictions.csv`, `imaging/train_segmentation_detection.ipynb`) were superseded by the 5-fold rerun that followed this archive step — see the 5-fold results doc for current numbers.

## Checkpoint format note

`resnet50_unet` checkpoints are saved from the *underlying* (uncompiled) module even though training used `torch.compile` — `torch.compile()`'s `OptimizedModule` wrapper prefixes `state_dict()` keys with `_orig_mod.`, which a plain `ResNet50UNet()` instantiated later (e.g. in `imaging_evaluation.ipynb`) cannot load directly without stripping that prefix. Verified empirically on torch 2.8 before the run and confirmed clean on the actual saved checkpoints (no `_orig_mod.` keys present).
