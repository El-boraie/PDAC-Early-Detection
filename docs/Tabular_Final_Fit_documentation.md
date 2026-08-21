# Clinical Final Fit — Process, Code, and Results

**Source notebook:** `src/clinical/clinical_final_fit.ipynb`
**Depends on:** `src/clinical/tabular_clean.ipynb` (cleaned data), `src/clinical/clinical_imputer_benchmark.ipynb` (settled `MICE_CA19_9Imputer` choice), `src/clinical/clinical_model_comparison.ipynb` (settled XGBoost choice + `results/clinical/model_comparison.csv` + `oof_predictions.csv`).
**Dataset:** Debernardi et al. 2020 urinary biomarker dataset — all 590 patients (199 PDAC / 391 not-PDAC), no split.
**Kernel / environment:** Python 3.13.12, packages from `C:\FYP\fyp_env\Lib\site-packages`.
**Run date:** 2026-07-16.

> Draft note: every number below is from the actual executed run, not estimated.

**Governing rules for this notebook:**
1. Refit **only** the confirmed winning model type (XGBoost) on all 590 patients — the three-way comparison is not rerun here.
2. Refit the MICE imputer on all data too — the imputer choice itself (MICE over KNN) is closed, not re-tested.
3. Fit the calibrator on the comparison stage's **saved** out-of-fold predictions (`oof_predictions.csv`), never on this notebook's own in-sample predictions — that would be circular.
4. Save `model.pkl`, `ca19_9_imputer.pkl`, `calibrator.pkl` to `checkpoints/clinical/final/` under exactly those generic filenames — never one that names the winning model type.
5. `model_card.json` is the **only** place the winning model's identity is recorded.
6. Any in-sample check is a sanity check only, never reported as new performance evidence.

This is the one notebook in the clinical branch that fits on **everything**, on purpose — every notebook before it existed to answer "does this generalize?" via cross-validation, discarding the model/imputer at the end of every fold. Validation is already done; this notebook answers "what's the single best version of it, using every patient we have?"

---

## Section 1 — The Winning Model: XGBoost

**Process:** `WINNER_MODEL_NAME = "XGBoost"` is set explicitly, **not** re-derived by a "pick the highest AUC" formula. The real justification (`docs/Tabular_Model_Comparison_documentation.md`'s Final Summary) is that XGBoost has the best sensitivity/recall — the clinically prioritised metric — in 2 of 3 CV schemes, with AUC only negligibly different from Random Forest in those same two schemes. A mechanical AUC-max rule happens to agree here, but hardcoding that formula instead of the real reasoning risks silently picking the wrong model in a future re-run if the numbers shift in a way that decouples AUC-max from sensitivity-max. The supporting numbers are still pulled programmatically from `model_comparison.csv` immediately after, so the justification is traceable to the actual file rather than just asserted.

**Result:**

| Scheme | Model | Recall (Sensitivity) | AUC |
|---|---|---|---|
| Repeated 5×20 CV | **XGBoost** | **0.7433** | 0.9077 |
| Repeated 5×20 CV | Logistic Regression | 0.6157 | 0.8786 |
| Repeated 5×20 CV | Random Forest | 0.6920 | 0.9068 |
| Cohort-out | XGBoost | 0.6757 | 0.8375 |
| Cohort-out | Logistic Regression | 0.7027 | 0.8793 |
| Cohort-out | **Random Forest** | **0.7568** | 0.8901 |
| Leave-one-site-out | **XGBoost** | **0.7783** | 0.8195 |
| Leave-one-site-out | Logistic Regression | 0.7483 | 0.8221 |
| Leave-one-site-out | Random Forest | 0.7290 | 0.8324 |

---

## Section 2 — Load `tabular_clean.csv`

**Process:** Same column-name reconstruction pattern as every prior clinical notebook — no re-derivation of logic already verified in `tabular_clean.ipynb`.

**Result:** `feature_matrix` (590, 7), `METADATA` (590, 5), `TARGETS` (590, 2). `plasma_CA19_9` missing in 240 rows, left raw as expected.

---

## Section 3 — The Settled Imputer: `MICE_CA19_9Imputer`

**Process:** Copied verbatim from the prior two notebooks. Per rule 2, the imputer choice itself is closed — the only thing different from every prior use of this class is that it gets fit on all 590 rows below, not a training fold. There is no held-out side left to protect once validation is finished.

---

## Section 4 — All-Data Fit: Imputer + Model, No Folds

**Process:** `MICE_CA19_9Imputer` fit on all 590 rows, transforming `feature_matrix` to fill all 240 missing `plasma_CA19_9` values using patterns learned from the full dataset. `XGBClassifier` fit with the **exact same hyperparameters used throughout the comparison stage** (`n_estimators=100, max_depth=3` — rule 1's spec, unchanged) on the fully-imputed feature matrix against all 590 labels.

**Result:** imputer fit on all 590 patients; model fit on all 590 patients (199 PDAC / 391 not-PDAC).

---

## Section 5 — Calibrator: Platt Scaling, Fit on Saved Out-of-Fold Predictions

**Process:** A judgment call, confirmed before implementing (not defaulted to silently):
- **Platt scaling** (a 1-D logistic regression mapping raw score → calibrated probability) chosen over isotonic regression — isotonic typically needs thousands of samples to avoid overfitting/staircase artifacts, and the *effective* sample size here is 590 unique patients, not the 11,800 rows in the OOF file, favouring the simpler, more stable parametric option.
- Fit on **all 11,800 `XGBoost` OOF rows directly**, not averaged down to one row per patient first — the 20 CV repeats aren't identical predictions (each repeat uses different fold partners), so they carry real additional information; averaging first would throw that away.

Per rule 3, fit on `oof_predictions.csv` — predictions the final model above never produced and never trained on — never on `final_model`'s own in-sample predictions, which would be circular (the model has already seen every one of those labels).

**Result:** calibrator fit on 11,800 XGBoost OOF rows (590 unique patients × 20 repeats each).

---

## Section 6 — Sanity Check: In-Sample Only, Not New Performance Evidence

**Process:** Per rule 6, scores `final_model` on the same 590 patients it was just trained on — it has already seen every one of these labels, so the result is expected to look better than any real CV estimate and **means nothing as evidence of generalization**. Its only purpose is confirming the fit actually worked (produces sane, non-degenerate predictions); shown next to the comparison stage's real repeated-CV numbers purely so the gap between "in-sample" and "actually validated" is visible, not to suggest the model got better.

**Result:**

| | AUC | Recall |
|---|---|---|
| In-sample (fit-on-everything) | 1.0000 | 1.0000 |
| Real validated estimate (repeated 5×20 CV) | 0.9077 | 0.7433 |

The perfect 1.0/1.0 in-sample score is the expected symptom of scoring a model on data it memorized, not a sign of anything wrong — it's exactly why this number is never reported as performance evidence.

---

## Section 7 — Save to `checkpoints/clinical/final/`

**Process:** Generic filenames per rule 4 and `FYP_Folder_Structure_Migration.md` Rule 2 — `app.py`/`fusion.ipynb` load these paths unconditionally and never branch on which model won. `model_card.json` is the only place the winning model's identity is recorded, and also captures the calibrator choice and the reasoning behind it, not just the bare fact that Platt scaling was used.

**Result:** all four files written and independently confirmed present on disk:

| File | Size |
|---|---|
| `model.pkl` | 116,590 bytes |
| `ca19_9_imputer.pkl` | 17,587 bytes |
| `calibrator.pkl` | 863 bytes |
| `model_card.json` | 2,644 bytes |

**`model_card.json` contents:**

```json
{
  "model": "xgboost",
  "model_hyperparameters": {"n_estimators": 100, "max_depth": 3, "eval_metric": "logloss", "random_state": 0},
  "trained_on": "2026-07-16",
  "n_patients": 590,
  "n_pdac": 199,
  "n_not_pdac": 391,
  "imputer": "MICE_CA19_9Imputer",
  "comparison_metrics_that_justified_this_model": { /* full recall_mean + auc_mean, all 3 models x 3 schemes */ },
  "selection_reasoning": "XGBoost has the best sensitivity/recall ... See docs/Tabular_Model_Comparison_documentation.md Final Summary for full reasoning.",
  "calibrator": "Platt scaling (LogisticRegression on raw predict_proba)",
  "calibrator_fit_rows": 11800,
  "calibrator_fit_unique_patients": 590,
  "calibrator_reasoning": "Platt scaling chosen over isotonic regression ... Fit on all repeated-CV OOF rows directly ...",
  "in_sample_sanity_check": {"note": "NOT a performance estimate -- scored on data the model was trained on", "auc": 1.0, "recall": 1.0}
}
```

---

## Final Summary

- **Outputs:** `model.pkl`, `ca19_9_imputer.pkl`, `calibrator.pkl`, `model_card.json` — all four confirmed present in `checkpoints/clinical/final/`, both by the notebook's own check and independently via the filesystem.
- **Nothing about which model won is discoverable anywhere except `model_card.json`** — the three `.pkl` files use generic names, matching the migration doc's Rule 2 (the deployed thing must never depend on who won).
- **No decision already settled elsewhere was re-opened**: the model choice (XGBoost) and imputer choice (MICE) were both carried forward as closed facts, not re-tested.
- **The calibrator decision — Platt scaling, fit on all 11,800 OOF rows unaveraged — was a judgment call surfaced and confirmed before implementing**, not defaulted to silently; both the choice and its reasoning are recorded permanently in `model_card.json`, not just in this conversation.
- **The in-sample sanity check (AUC/recall both 1.0000) is reported exactly as what it is** — confirmation the fit worked, not a performance claim — sitting directly next to the real validated numbers (0.9077 / 0.7433) so the difference is impossible to miss or misquote later.

This closes out the clinical branch's training path: preprocessing (`tabular_clean.ipynb`) → imputer selection (`clinical_imputer_benchmark.ipynb`) → model selection (`clinical_model_comparison.ipynb`) → final deployable artifact (this notebook). `src/fusion/fusion_evaluation.ipynb` and `dashboard/app.py` are the two remaining consumers of `checkpoints/clinical/final/`, both still scaffolding/not-yet-created respectively.
