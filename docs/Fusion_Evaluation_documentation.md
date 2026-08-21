# Fusion Evaluation — Pair Comparison, Calibration, Worked Examples

**Source:** `src/fusion/fusion_evaluation.ipynb`.
**Purpose:** the one real comparison this project's data supports between the imaging and
clinical branches — each branch's own already-computed metrics side by side, each branch's
own calibration quality, and a small illustrative demo of what combining two calibrated
scores would look like. **Not** the inference-time combiner (`fusion.ipynb`, documented
separately in `docs/Fusion_documentation.md`).
**Run date:** 2026-07-19.

---

## The Hard Constraint This Whole Notebook Is Built Around

No patient in this project has both a CT scan and a urine sample. MSD, NIH, and the
Debernardi et al. urine cohort are three separate, unpaired sources — imaging's 361 patients
and clinical's 590 patients share no common individual. There is therefore **no ground
truth against which a fused prediction could be scored**, and this notebook never computes
or reports a joint precision/recall/F1/ROC-AUC/confusion matrix across the two branches.
This is a hard architectural limitation of the data, not a shortcut to fix, and not
something worked around with a synthetic paired test set.

Every number in this notebook is either (a) one branch's own metric, computed entirely
within that branch's own patients, or (b) explicitly labeled illustrative and disclaimed as
not a benchmark.

---

## Pair Comparison Table

`resnet50_unet` (imaging's winner) vs. `XGBoost` (clinical's winner) — one row, each
branch's own metrics as separate `imaging_*`/`tabular_*` columns, no joint column. Pulled
directly from each branch's own `results/{imaging,clinical}/model_comparison.csv`, **never
recomputed from raw predictions**, to avoid introducing a subtly different number for no
reason. Reference scheme: imaging's only scheme (`5-Fold CV`); clinical's `Repeated 5x20 CV`
(100 folds, the most statistically robust of its three schemes, and the same one
`clinical_final_fit.ipynb` reports its own in-sample check against).

| | imaging: `resnet50_unet` | tabular: `XGBoost` |
|---|---|---|
| Precision | 0.9920 | 0.7760 |
| Recall | 0.9628 | 0.7433 |
| Specificity | 0.9696 | 0.8884 |
| F1 | 0.9768 | 0.7573 |
| ROC-AUC | 0.9934 | 0.9077 |
| Confusion matrix (TN/FP/FN/TP) | 18,031 / 585 / 2,679 / 69,398 | 6,947 / 873 / 1,022 / 2,958 |

`resnet50_unet` looks stronger on every metric — but these come from two entirely different
evaluations: imaging is **slice-level** detection on CT scans (MSD/NIH), tabular is
**patient-level** on the Debernardi et al. urine cohort. They are not comparable as a
"better model wins" statement, only as two independent facts.

Written to `results/fusion/pair_comparison.csv` — includes the confusion matrices and a
`note` column restating the no-joint-metric limitation directly in the file, plus an
`imaging_known_limitations_caveat` column (see below) so the caveat travels with the CSV
even if opened outside this notebook. Figure: `outputs/eval/fusion/pair_comparison.png` — a
side-by-side bar chart with the full caveat text printed underneath.

---

## The Imaging Confound-Check Caveat

Per instruction, `resnet50_unet`'s strong headline numbers are never presented anywhere in
this notebook's output without the following caveat attached, quoted verbatim from
`checkpoints/imaging/final/model_card.json`'s `known_limitations.confound_check_summary`:

> A rigorous, 6-round confound-check investigation (Grad-CAM + 5 other attribution methods,
> then causal occlusion testing) found this architecture's detection head originally relied
> MORE on the BOX=320 packing's synthetic padding than on the real tumour region (Round 4:
> 2.03x a random-patch control, p=0.044). A random-resized-crop augmentation (augment=True,
> used in this final fit) verified-fixed the padding shortcut (Round 5: 1.08x control,
> p=0.917 — statistically indistinguishable from the control). A second remediation attempt
> (Round 6: mask-preserving random erase) targeting the model's remaining UNDER-reliance on
> the tumour region itself did NOT help (0.58x control, p=0.0148, statistically unchanged
> from Round 5's 0.62x) and was reverted — this promoted model does not include that change.
> TUMOUR UNDER-RELIANCE REMAINS OPEN AND UNRESOLVED in this promoted model: the real tumour
> region moves this model's prediction significantly LESS than a same-sized random patch of
> ordinary tissue. Detection metrics in this model card are strong and reproducible, but are
> NOT yet confirmed evidence of tumour-specific pixel-level reasoning.

**Where it appears:** the pair table's own CSV column, the `pair_comparison.png` figure
annotation, the Brier-score print-out, the `calibration_curves.png` figure annotation, every
worked-example row's imaging score, and the notebook's final summary cell — every single
place imaging's metrics surface in this notebook's output.

---

## Calibration Quality

Brier score (mean squared error between predicted probability and the true outcome; 0 =
perfect) and reliability diagrams, computed from each branch's own
`results/{imaging,clinical}/oof_predictions.csv` — never recomputed from raw inference.
Both a **raw** version (the OOF column as-is, what the Platt calibrator was fit *on*) and a
**calibrated** version (that same column run through the branch's own saved
`checkpoints/*/final/calibrator.pkl`) are shown, to confirm the calibrator is doing its job.

| | Tabular: `XGBoost` | Imaging: `resnet50_unet` |
|---|---|---|
| Brier (raw) | 0.1213 | 0.0303 |
| Brier (calibrated) | 0.1189 | 0.0282 |

**Caveat on the calibrated numbers:** mildly optimistic, since each calibrator was fit on
this exact OOF set rather than a further held-out split — the same accepted limitation each
branch's own final-fit notebook already carries (`FYP_Folder_Structure_Migration.md` Section
6). Shown to confirm the calibrator improves calibration, not as an independent estimate.

Reliability diagrams (10-bin, raw vs. calibrated, side by side per branch) saved to
`outputs/eval/fusion/calibration_curves.png`, with the imaging caveat printed beneath.

---

## Worked Fusion Examples — Illustrative Only, Not a Benchmark

Per rule 1, no real fused prediction is possible (no patient has both modalities). What's
shown instead: 5 "hypothetical patient" rows, built by taking a spread of **real** raw OOF
probabilities from each branch (low/medium/high, from that branch's real
`oof_predictions.csv`), running each through that branch's own real saved `calibrator.pkl`,
then **artificially pairing** one imaging score with one tabular score (shuffled, no real
correspondence) and combining them with a simple mean.

| patient | imaging_calibrated_proba | tabular_calibrated_proba | fused (simple mean) | band |
|---|---|---|---|---|
| H1 | 0.9947 | 0.0822 | 0.5384 | medium |
| H2 | 0.9947 | 0.0835 | 0.5391 | medium |
| H3 | 0.9947 | 0.1089 | 0.5518 | medium |
| H4 | 0.0909 | 0.7128 | 0.4018 | medium |
| H5 | 0.9884 | 0.8840 | 0.9362 | high |

**This is a mechanism demo, not a benchmark:** the pairings are fabricated (shuffled, not
matched to any real correspondence), the "simple mean" rule here is illustrative only
(the actual inference-time combiner in `fusion.ipynb` was later given a different,
deliberately-weighted rule — see `docs/Fusion_documentation.md`), and no accuracy number is
computed or implied, since no ground truth exists for any of these rows. Saved to
`results/fusion/worked_fusion_examples.csv`.

---

## Why There's No Single Joint Accuracy Number

Restated explicitly, because it's the central constraint of this whole notebook: the
numbers above are two **separate** evaluations, on two **separate, unpaired** cohorts. No
patient has both, so there is no way to score a combined prediction. This notebook does not
compute or report a joint precision/recall/F1/ROC-AUC/confusion matrix anywhere, and the
worked-example fusion rows are illustrative only.

---

## Artifacts

- `results/fusion/pair_comparison.csv`
- `results/fusion/worked_fusion_examples.csv`
- `outputs/eval/fusion/pair_comparison.png`
- `outputs/eval/fusion/calibration_curves.png`
