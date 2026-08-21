"""Shared metric helpers for the PDAC FYP's new notebooks.

The tabular-side functions here are real, working code, ported from the
already-verified `clinical.ipynb` (see `docs/Tabular_Preprocessing_documentation.md`
Section 8) -- `clinical_model_comparison.ipynb` depends on `early_stage_recall`
producing the same numbers as before, so this is not a stub.

The imaging-side functions (dice_score, iou_score, detection_metrics) are real,
working code too, implemented alongside `train_segmentation_detection.ipynb`
which depends on them for its per-epoch and final-fold metrics.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Resectability-based cut, not a canonical clinical threshold -- judgment call,
# flagged for confirmation in docs/Tabular_Preprocessing_documentation.md Section 8.
EARLY_STAGES = {"I", "IA", "IB", "II", "IIA", "IIB"}


def pr_auc(y_true: pd.Series, y_proba: np.ndarray) -> float:
    """Precision-Recall AUC. NaN (not an error) if y_true is single-class,
    same convention as roc_auc_score's handling elsewhere in this project."""
    if y_true.nunique() <= 1:
        return np.nan
    return average_precision_score(y_true, y_proba)


def early_stage_recall(
    y_true: pd.Series, y_pred: np.ndarray, stage: pd.Series, early_stages: set = EARLY_STAGES
) -> float:
    """Recall restricted to patients whose `stage` is early (see EARLY_STAGES).
    `y_true`, `stage` must share the same index; `y_pred` is positional, same
    order as `y_true`. NaN if no early-stage-positive rows exist in this slice.
    """
    y_pred_series = pd.Series(np.asarray(y_pred), index=y_true.index)
    is_early = stage.isin(early_stages)
    early_idx = y_true.index[is_early.values]
    if len(early_idx) == 0:
        return np.nan
    return recall_score(y_true.loc[early_idx], y_pred_series.loc[early_idx])


def dice_score(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int = 3, foreground_only: bool = True) -> float:
    """Mean multiclass Dice coefficient. pred_mask/true_mask are integer label
    maps (values in [0, num_classes), NOT one-hot) -- background=0, pancreas=1,
    tumour=2 for this project's masks. Arrays may be a single (H, W) slice or
    a stacked (N, H, W) batch; in the batch case this POOLS pixels across the
    whole batch before dividing (a micro-average over pixels), not a per-slice
    mean -- call once over the full validation set for one aggregate number,
    not per-slice-then-averaged.

    foreground_only=True (default) skips background and averages Dice over
    the foreground classes only (pancreas, tumour) -- background is trivial
    to get right (it's most of the image) and would inflate the score.

    A class absent from BOTH pred and true for the given input is undefined
    and skipped (NaN if every class is skipped), not silently scored as 0 --
    a class present in true_mask but missed entirely by pred_mask still
    scores a real 0.0, since that IS a real miss, not a missing-data case.
    """
    classes = range(1, num_classes) if foreground_only else range(num_classes)
    scores = []
    for c in classes:
        pred_c = pred_mask == c
        true_c = true_mask == c
        denom = int(pred_c.sum()) + int(true_c.sum())
        if denom == 0:
            continue
        intersection = int((pred_c & true_c).sum())
        scores.append(2 * intersection / denom)
    return float(np.mean(scores)) if scores else np.nan


def iou_score(pred_mask: np.ndarray, true_mask: np.ndarray, num_classes: int = 3, foreground_only: bool = True) -> float:
    """Mean multiclass Intersection-over-Union. Same conventions as dice_score:
    integer label maps, pooled over a batch if given one, foreground-only by
    default, NaN only when a class has zero pixels in both pred and true."""
    classes = range(1, num_classes) if foreground_only else range(num_classes)
    scores = []
    for c in classes:
        pred_c = pred_mask == c
        true_c = true_mask == c
        union = int((pred_c | true_c).sum())
        if union == 0:
            continue
        intersection = int((pred_c & true_c).sum())
        scores.append(intersection / union)
    return float(np.mean(scores)) if scores else np.nan


def detection_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    """Precision, Recall/Sensitivity, Specificity, F1, ROC-AUC, Confusion Matrix
    for binary detection (0=healthy, 1=cancer). y_pred is the thresholded
    (0/1) prediction; y_proba is the raw probability/logit-sigmoid used for
    ROC-AUC only.

    Every ratio metric is NaN, not 0.0, when its own denominator is zero --
    verified against sklearn's actual per-metric zero_division=np.nan
    behavior (precision, recall, and F1 each have a different zero-division
    condition; np.nan is honored independently for each, not derived from
    the others). This distinction previously caused a real bug in the
    tabular branch where zero_division=0 silently reported 0.0 and changed
    the results before it was caught -- do not swap this back to 0.

    Specificity has no sklearn builtin -- computed by hand from the
    confusion matrix (TN / (TN + FP)), NaN if that denominator is 0 (no
    actual negatives in y_true).

    ROC-AUC is NaN if y_true is single-class (same convention as pr_auc).
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_proba = np.asarray(y_proba)

    precision = precision_score(y_true, y_pred, zero_division=np.nan)
    recall = recall_score(y_true, y_pred, zero_division=np.nan)
    f1 = f1_score(y_true, y_pred, zero_division=np.nan)
    roc_auc = np.nan if len(np.unique(y_true)) <= 1 else roc_auc_score(y_true, y_proba)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = np.nan if (tn + fp) == 0 else tn / (tn + fp)

    return {
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }
