# Clinical SHAP Explainability

Companion documentation for `src/clinical/clinical_shap.ipynb`. This notebook adds
explainability to the **already-final** clinical XGBoost model
(`checkpoints/clinical/final/model.pkl`, `binary:logistic`, `n_estimators=100`,
`max_depth=3`, trained 2026-07-16 on all 590 patients). It does not retrain, refit, or
tune anything, and does not touch the imaging or fusion branches.

## Design decisions

**1. `shap.TreeExplainer`, not `KernelExplainer`.** `TreeExplainer` computes exact
Shapley values for tree ensembles by walking the actual XGBoost trees
(`feature_perturbation="tree_path_dependent"`) — it has a closed-form solution for
this model class. `KernelExplainer` is a model-agnostic Monte Carlo approximation
meant for models with no closed-form SHAP algorithm; using it here would be strictly
worse (slower, approximate) for a model where the exact method is available and free.

**2. Explain the imputed matrix, not the raw one.** The deployed pipeline never scores
a row with a missing `plasma_CA19_9` — it always runs the fitted `MICE_CA19_9Imputer`
first. SHAP is therefore run on `X_imputed` (the same matrix `model.predict` actually
sees at inference), not on the raw CSV with NaNs. Explaining the raw matrix would mean
explaining a hypothetical input the model never receives. The imputer is only ever
`.transform()`-ed here, never `.fit()`-ed — it is loaded exactly as `fusion.ipynb`
loads it, including copying the `MICE_CA19_9Imputer` class definition verbatim into
this notebook's kernel to satisfy `joblib.load`'s unpickling requirement.

**3. Raw log-odds (margin) space, not calibrated probability.** `TreeExplainer`'s
`model_output` is left at its default (raw margin). SHAP's additivity guarantee —
`expected_value + sum(shap_values) == model output` — only holds in the space the
trees themselves produce. `calibrator.pkl` (Platt scaling) sits *after* the raw model
as a separate, monotonic squashing function; explaining pre-calibration keeps every
attribution additive and traceable to a specific tree split, at the cost of the
numbers not being directly readable as "percentage points of risk."

**4. All 590 patients, in-sample, is the correct set to explain.** No train/test split
is used here. SHAP in this notebook answers "what did this model learn to do?", not
"how well does it generalize?" — the latter question was already answered by the
repeated-CV numbers in `model_card.json` (`recall_mean=0.7433`, `auc_mean=0.9077`
across repeated 5x20 CV). Explaining the model against every patient it has ever seen
is the right question for a decision-function explanation; a held-out subset would
just be a smaller, noisier sample of the same fixed function.

## Additivity check

For every row, `expected_value + shap_values.sum(axis=1)` was checked against
`model.predict(X_imputed, output_margin=True)`.

**Result: PASS. Max absolute error = 4.77e-06** (threshold was `1e-4`).

This confirms the SHAP values are not an approximation drifting from the model's
actual output — they reconstruct it to within floating-point noise. All plots below
were generated only after this check passed.

## Global feature ranking

Mean |SHAP value| across all 590 patients, in the raw log-odds space:

| Rank | Feature | mean(\|SHAP\|) | Biological / clinical read |
|---|---|---|---|
| 1 | LYVE1 | 1.669 | Urinary lymphatic marker from the Debernardi panel — the single strongest driver of this model's output. |
| 2 | plasma_CA19_9 | 1.293 | The established serum biomarker for PDAC in clinical practice — second here, and notably compressed by the fact that 240/590 values are imputed rather than measured (see caveat below). |
| 3 | creatinine | 0.832 | General renal-function marker included as a covariate in the Debernardi panel. |
| 4 | age | 0.810 | Demographic risk factor — PDAC incidence rises with age. |
| 5 | TFF1 | 0.599 | Urinary marker from the Debernardi panel. |
| 6 | REG1B | 0.382 | Urinary marker from the Debernardi panel. |
| 7 | sex | 0.073 | Weakest driver — the model relies on it very little. |

The Debernardi urinary panel (LYVE1, REG1B, TFF1) and CA19-9 together dominate the
model's decisions, which is consistent with why this feature set was chosen for the
clinical branch in the first place. LYVE1 outranking CA19-9 here is an empirical
finding of *this specific fitted model*, not an assumption — it is worth flagging in
any presentation of results, since CA19-9 is the biomarker most familiar to a clinical
audience.

## Limitation: the ranking is on the log-odds scale, not the probability scale

Every ranking above measures each feature's effect on the model's raw log-odds output,
not on the final predicted probability a clinician would read. Converting log-odds to
a probability is a nonlinear (sigmoid) squashing: the same size push on the raw score
translates into a large probability swing for a patient near the 50/50 decision
boundary, but almost no probability swing for a patient the model is already very
confident about (near 0% or 100%). Since this model is close to saturated in-sample
(`auc=1.0`, `recall=1.0` per `model_card.json`'s own sanity check), a meaningful number
of patients likely sit in those flattened regions of the curve, so the log-odds ranking
is not guaranteed to be the ranking a clinician reading percentages would actually see.

This was checked directly: a second `TreeExplainer` was run with
`model_output="probability"` (necessarily using the approximate `interventional`
algorithm with a background dataset, rather than the exact `tree_path_dependent`
algorithm used everywhere else in this notebook, and — a shap-library compatibility
detail, not a modeling choice — via an in-memory copy of the model with its
leftover `enable_categorical=True` flag turned off, verified to produce byte-identical
predictions first). The result (`results/clinical/shap_ranking_comparison.csv`):
**Spearman rank correlation of 0.9643 between the two rankings — only LYVE1 and
plasma_CA19_9 swap places** (CA19-9 actually edges ahead of LYVE1 once measured in
probability terms), every other feature keeps the same rank in both spaces. So the
ranking is largely robust to this choice, with one genuine exception: **whether LYVE1
or CA19-9 is "the top feature" depends on which scale you ask the question in**, and
that ambiguity should be carried into any claim about which single feature matters
most. This check is itself approximate (feature-independence assumption, and shap's
default cap subsampled the background from 590 to 100 patients) — it is a secondary
robustness check, not a replacement for the primary log-odds-based analysis above.

## Imputation caveat — do not omit when reading the plots

**240 of 590 patients (41%) never had `plasma_CA19_9` measured.** For those rows, the
value SHAP is attributing an effect to is the `MICE_CA19_9Imputer`'s *prediction* of
what CA19-9 would have been, based on creatinine, LYVE1, REG1B, TFF1, and age — not a
real lab measurement. Concretely:

- In `shap_dependence_CA19_9.png`, the imputed rows (orange ×) cluster in the low-value
  range and largely track the same trend as measured rows (blue ○), which is expected
  since the imputer was fit to reproduce that relationship — but it means the apparent
  smoothness of the curve in that region is partly circular: low imputed CA19-9 values
  are themselves derived from the other four correlated features, which the model *also*
  sees directly.
- `shap_waterfall_example_2.png` deliberately shows a patient (row 576) whose
  `plasma_CA19_9` was imputed (imputed value 1919.95) and whose CA19-9 SHAP
  contribution (+2.75, the largest single contribution in that row) is the *biggest*
  driver of that patient being classified as PDAC. Reading that waterfall as "this
  patient's measured CA19-9 pushed the model toward PDAC" would be wrong — it was a
  model-estimated value doing the pushing.
- Any global importance attributed to `plasma_CA19_9` (rank 2 above) is therefore a mix
  of genuine biomarker signal (from the 350 measured rows) and imputer-model behavior
  (from the 240 imputed rows). The ranking should not be read as "CA19-9 is this
  informative as an actual lab test" without that caveat.

## Outputs

- `results/clinical/shap_values.csv` — 590 rows x 7 feature columns (SHAP values,
  `feature_order` = `creatinine, LYVE1, REG1B, TFF1, plasma_CA19_9, age, sex`).
- `results/clinical/shap_values_meta.json` — `expected_value` (base value, log-odds),
  `output_space`, `feature_order`, `additivity_max_abs_error`.
- `results/clinical/shap_ranking_comparison.csv` — log-odds vs. probability-space
  mean |SHAP| and rank per feature, for the robustness check above.
- `outputs/eval/clinical/shap_global_importance.png` — mean |SHAP| bar chart.
- `outputs/eval/clinical/shap_beeswarm.png` — SHAP summary beeswarm.
- `outputs/eval/clinical/shap_dependence_CA19_9.png` — dependence plot, imputed rows
  marked distinctly (orange ×) from measured rows (blue ○).
- `outputs/eval/clinical/shap_dependence_LYVE1.png` — dependence plot for the
  top-ranked feature.
- `outputs/eval/clinical/shap_waterfall_example_0.png` — most-confident PDAC
  prediction (row 502, raw margin 8.01).
- `outputs/eval/clinical/shap_waterfall_example_1.png` — most-confident control
  prediction (row 381, raw margin -9.28).
- `outputs/eval/clinical/shap_waterfall_example_2.png` — a CA19-9-imputed patient
  (row 576), chosen as the imputed row where that feature's SHAP contribution is
  largest, so the caveat above is visible rather than incidental.

## `explain_tabular()` helper

The notebook also exposes `explain_tabular(row_df) -> (shap_per_feature: dict,
base_value: float)`, which imputes and explains one submitted row using the same
loaded model/imputer/explainer, for a dashboard to call on new patients. It was smoke
tested against the notebook's own imputed-example row and reconstructs the model's raw
margin prediction exactly (within `1e-4`).
