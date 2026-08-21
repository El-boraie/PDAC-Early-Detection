# Imaging Branch Confound Investigation — Full Technical Dossier

**Audience note:** this document is written for an AI assistant (Claude) to later parse and
translate into plain English / presentation language on request. It is deliberately dense,
exhaustive, and technically precise rather than narrative — every method, formula, sample
size, statistical test, and result is stated exactly as executed, with no simplification.
Do not treat this as the version to read aloud to a non-technical audience; see
`docs/Imaging_GradCAM_Confound_Supervisor_Briefing.md` for that.

**Source of truth:** `docs/Imaging_Confound_Check_documentation.md` (primary),
`docs/Imaging_Session_Summary_2026-07-18.md`, `docs/Imaging_Evaluation_documentation.md`,
`docs/Imaging_5Fold_Training_Results_documentation.md`, `src/imaging/models.py`,
`imaging/imaging_confound_check.ipynb`, `imaging/slice_cache_dataset.py`.

---

## 1. Problem Statement — Exact Formulation

**Model under test:** `resnet50_unet`, the winning candidate from the imaging branch's
5-fold cross-validation comparison (see §9 for architecture spec). This is the model that
was slated for promotion to `checkpoints/imaging/final/`.

**Dataset confound, stated precisely:** the imaging training set is the union of two source
collections:
- **MSD** (Medical Segmentation Decathlon, Task07 Pancreas): 281 patients, all `class=1`
  (cancer), all with tumour+pancreas segmentation masks.
- **NIH** (Pancreas-CT / TCIA CT-82): 80 patients, all `class=0` (healthy), no segmentation
  mask (image-only cohort, added specifically to supply true negatives).

`dataset ∈ {MSD, NIH}` and `class ∈ {cancer, healthy}` are **deterministically identical
partitions of the patient population** — confirmed exactly in the manifest: 281/281 MSD
patients are class=1, 80/80 NIH patients are class=0, zero exceptions. This is a
100%-collinear nuisance variable with the label. Any model feature correlated with
scanner/institution/acquisition protocol (not disease) is therefore statistically
indistinguishable, at the level of aggregate accuracy metrics, from a feature correlated
with real pathology. ROC-AUC, recall, precision, F1 — none of these metrics can
discriminate "the model learned pathology" from "the model learned acquisition source."

**Formal question being tested:** does `resnet50_unet`'s detection head (binary logit
output, cancer-slice-presence) base its decision on spatial evidence that is anatomically
localized to the tumour/pancreas region, or on some other spatially-distributed or
non-anatomical cue correlated with dataset origin?

**Reported baseline performance at the time this investigation began** (5-fold CV, pre-any-
remediation, `no_augmentation` archive): ROC-AUC 0.9938, Recall 0.9833, Precision 0.9644,
Specificity 0.8614, F1 0.9736 (mean across 5 folds). These numbers are consistent whether
the "real pathology" or "scanner shortcut" hypothesis is true — this is the entire
motivation for the investigation.

---

## 2. Scope Constraint — Why the Notebook Was Built Standalone

`src/imaging/imaging_evaluation.ipynb` had unimplemented `raise NotImplementedError()` stub
cells for candidate comparison / promotion logic at the time this investigation began,
which would halt top-to-bottom execution before reaching any confound-check section. The
investigation notebook (`imaging/imaging_confound_check.ipynb`) was built as a fully
independent artifact, depending only on already-finalized upstream artifacts:
`checkpoints/imaging/candidates/resnet50_unet/{fold_0..fold_4}.pt`,
`results/imaging/oof_predictions.csv`, and the packed slice cache at `data/processed/cache/`.
It does not implement, modify, or depend on the comparison/promotion logic.

---

## 3. Sampling Protocol (applies to Rounds 1–4 and the Round 5/6 re-verifications)

- **Population sampled:** correctly-classified MSD (cancer=1) true positives only — i.e.,
  slices where ground truth = cancer AND model prediction = cancer, under the fold-
  appropriate held-out checkpoint (never a model scored on data it trained on).
- **Sample size per fold:** 40 slices per fold × 5 folds = **200 total sampled slices**.
- **Mask availability constraint:** the `class` label is patient-level, not slice-level —
  many individual axial slices of a cancer patient's volume (chest, hip, other
  non-abdominal anatomy) do not intersect the pancreas at all and have an empty
  tumour/pancreas mask. Of the 200 sampled slices, only **59 had a non-empty mask**. All
  overlap-based metrics (`overlap_precision`, `overlap_recall`, `enrichment`) are computed
  and averaged only over this **n=59** subset — this is a real property of the sampling
  procedure, disclosed and accounted for, not a bug.
- **`border_fraction`** does not require a mask and is computed over the full sample: 200
  MSD (all sampled, masked or not) + 100 NIH (sampled separately as a healthy-cohort
  comparison set for border concentration only, since NIH has no mask to check overlap
  against).
- **Target layer for layer-based methods:** `layer4[-1]` of the ResNet-50 encoder — the
  final convolutional block feeding the detection head's global-average-pool, exposed via a
  `DetectionOnlyWrapper` that returns only `det_logit` from `ResNet50UNet.forward()`
  (discarding `seg_logits`), so every attribution method targets the detection decision
  specifically, not the segmentation head. Integrated Gradients is the sole exception,
  attributing directly to input pixels rather than through this intermediate layer.

---

## 4. Metric Definitions (exact)

For a given slice, define the **attended region** as the top 20% of the attribution
heatmap's values (by magnitude), and the **real anatomy region** as the ground-truth
tumour+pancreas mask (label ∈ {1,2}).

- **`overlap_precision`** = |attended ∩ real anatomy| / |attended region| — fraction of the
  model's attended pixels that fall on real anatomy.
- **`overlap_recall`** = |attended ∩ real anatomy| / |real anatomy| — fraction of real
  anatomy the attended region covers.
- **`enrichment`** = `overlap_precision` ÷ (mask area fraction of the slice) — i.e.,
  precision normalized against the chance rate a spatially-random 20%-area region would
  achieve given how much of the slice the mask actually occupies. `enrichment = 1.0` means
  exactly chance-level overlap; `> 1.0` means the model's attention is enriched for real
  anatomy beyond what randomness would produce; `< 1.0` means anti-enriched (worse than
  random).
- **`border_fraction`** = fraction of the attended region's pixel mass falling within the
  outer 10%-width image margin on all four sides. Computed on all 200 MSD + 100 NIH sampled
  slices (mask-independent).

---

## 5. Attribution Methods Used — Full Technical Roster

| Method | Mechanism | Gradient required? | Target |
|---|---|---|---|
| **Grad-CAM** | Weights `layer4[-1]` activation maps by the global-average-pooled gradient of the detection logit w.r.t. those activations; ReLU'd weighted sum | Yes | `layer4[-1]` |
| **GradCAM++** | Generalization of Grad-CAM with pixel-wise weighting derived from higher-order gradients, better multi-instance/fine localization in theory | Yes | `layer4[-1]` |
| **HiResCAM** | Element-wise (not spatially-pooled) gradient × activation product, theoretically more faithful to the exact linear contribution than Grad-CAM's global pooling | Yes | `layer4[-1]` |
| **XGradCAM** | Axiom-based re-weighting of Grad-CAM's channel weights to better satisfy sensitivity/conservation axioms | Yes | `layer4[-1]` |
| **EigenCAM** | Principal component (via SVD) of the raw `layer4[-1]` activation tensor — no class/logit signal, no backward pass at all | **No** (gradient-free) | `layer4[-1]` |
| **Integrated Gradients** | Path integral of gradients from a baseline (zero) input to the actual input, attributing directly to input pixels | Yes | Input pixels (not `layer4`) |
| ~~ScoreCAM~~ / ~~AblationCAM~~ | Perturbation-based: forward-pass each channel independently to measure its causal contribution | N/A | **Excluded — see §5.1** |

### 5.1 Why ScoreCAM / AblationCAM Were Excluded (explicit scoping decision)

Both are perturbation-based attribution methods that require one full forward pass per
channel of the target layer to estimate each channel's causal contribution to the output.
`layer4` in this architecture has **2,048 channels**. Running either method across the full
5-fold × 40-slices-per-fold sweep would require on the order of `2048 × 200` = ~409,600
additional forward passes just for the attribution stage — judged prohibitively expensive
given the project's compute budget and timeline. This is a disclosed scoping decision, not
a hidden gap. **Note the terminological distinction:** "AblationCAM" (excluded, above) is a
specific internal-channel-perturbation attribution technique, distinct from the
**input-region occlusion / ablation study** actually performed and described in §7 — the
latter perturbs the raw input image (not internal channels) and is computationally cheap
(3 conditions × 59 slices = 177 forward passes total), which is why it was feasible where
ScoreCAM/AblationCAM were not.

---

## 6. Round-by-Round Results (exact numbers)

### Round 1 — Grad-CAM Alone

| Metric | Value |
|---|---|
| Mean enrichment (n=59 masked) | 0.003× |
| Mean overlap recall | 0.0005 |
| Median overlap precision (n=59) | 0.0 exactly |
| Mean border_fraction, MSD (n=200) | 0.346 |
| Mean border_fraction, NIH (n=100) | 0.311 |

Consistent across all 5 independently-trained fold checkpoints — not a single-fold anomaly.
Visual panels: `outputs/eval/imaging/gradcam_confound_check_fold{0..4}.png` — hottest region
visibly sits at image top corners in multiple examples, not over the pancreas.

### Round 2 — Integrated Gradients Cross-Check

| Metric | Value |
|---|---|
| Enrichment | 1.081× |
| border_fraction | 0.116 |

Directly contradicts Round 1's near-zero enrichment and border concentration on the same 59
slices, same checkpoints. Judged inconclusive alone — motivated further investigation rather
than a conclusion at this stage.

### Round 2b — Padding Investigation (data-level, not model-level)

**Detection method:** synthetic padding produces a *contiguous margin of exactly 0.0
intensity* — a signature real anatomy never produces (even air/background voxels carry
noise/texture in the original data). Checked directly against the cached (post-crop/pad)
images since raw pre-crop data was not retained on the compute environment used.

**Sample:** n=300 per dataset (MSD, NIH) — a corrected, adequately-powered redraw; an
earlier n=150 draw had overstated the effect and was superseded.

| Padding dimension | Finding |
|---|---|
| Bottom padding | Substantial, nearly identical between MSD/NIH (median ~34–35px) — a shared artifact, not a distinguishing cue on its own |
| Top padding | **Differs with statistical significance**, Mann-Whitney U test, p=0.011; NIH averages more |
| Left/right padding | No significant difference (p=0.256, p=0.835) |
| Tail behavior | NIH has a heavier tail of near-fully-padded, low-content slices (95th percentile hits the max padding cap on every side for NIH; MSD's is much lower) |

**Conclusion at this stage:** padding is real, and differs between MSD/NIH with some
statistical support, but not as a dramatic uniform difference — a plausible partial
contributor, not by itself sufficient evidence, motivating the full 6-method sweep.

### Round 3 — Six-Method Attribution Sweep

Same 59 masked slices, same fold-appropriate checkpoints.

| Method | Enrichment | Overlap recall | Border fraction |
|---|---|---|---|
| Integrated Gradients (input-space) | 1.081 | 0.216 | 0.116 |
| EigenCAM (gradient-free, layer4) | 0.640 | 0.209 | 0.329 |
| GradCAM++ (gradient, layer4) | 0.006 | 0.001 | 0.323 |
| GradCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |
| HiResCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |
| XGradCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |

**Pattern structure, two independent axes:**
1. All four `layer4`-gradient methods converge to near-identical near-zero enrichment
   (0.003–0.006) despite genuinely different mathematical weighting schemes — this
   cross-method agreement is itself evidence the near-zero result isn't an idiosyncrasy of
   vanilla Grad-CAM specifically.
2. Removing the gradient step alone (EigenCAM, same layer): enrichment → 0.640 (>200×
   improvement over the gradient methods, still not confidently above 1.0× chance).
3. Removing `layer4` and the backward-pass-through-a-late-layer mechanism entirely
   (Integrated Gradients, input-space): enrichment → 1.081 (at chance), border_fraction
   drops to roughly a third of every layer4-based method's value (0.116 vs. ~0.32–0.33).

Visual: `outputs/eval/imaging/confound_check_all_methods_comparison.png` (3 examples) — all
four gradient-based methods fixate on the same top-corner region regardless of true mask
location in every example; EigenCAM diffuse/non-specific in one example, corner-biased in
others; Integrated Gradients' hot pixels scatter through actual tissue content in all three.

**Interim interpretive read at this checkpoint (superseded by Round 4, stated for
completeness):** "substantially a gradient/layer4-attribution-method artifact, not strong
independent evidence of a confound" — reasoned from the convergence pattern above. Flagged
explicitly in the source documentation as provisional, since every method up to this point,
gradient-free included, still only *infers* importance from gradients or activations; none
*intervenes* on the input.

### Round 4 — Occlusion-Based Sensitivity Analysis (Causal) — see §7 for full method detail

**Headline result table:**

| Condition | Mean \|Δ logit\| | Ratio vs. control | Wilcoxon signed-rank p (paired, vs. control) |
|---|---|---|---|
| Tumor occlusion | 0.110 | 0.47× | 0.0075 |
| Padding occlusion | 0.475 | 2.03× | 0.0442 |
| Random control | 0.234 | 1.00× (reference) | — |

Both differences statistically significant at α=0.05, paired on the identical 59 slices.
**This directly reverses the Round 3 provisional interpretation** — causal evidence from
physical input manipulation is treated as outranking correlational evidence from
gradient/activation-based inference. Visual: `outputs/eval/imaging/confound_check_occlusion_sensitivity.png`.

**Final verdict as of Round 4 (pre-remediation):** `resnet50_unet`'s near-perfect detection
ROC-AUC (0.994 at the time) reflects, with direct causal support, a real non-anatomical
shortcut (padding-reliance) alongside genuine tumour-under-reliance — not confirmed
pixel-level pathological reasoning. Does not prove the *specific* mechanism is
"scanner/institution identity encoded via padding" (would need controlled metadata
manipulation this dataset's structure doesn't support) and does not establish the model is
useless — the accuracy is real and reproducible, just not evidence of the reasoning it might
appear to reflect.

---

## 7. Occlusion / Input-Ablation Study — Full Methodological Detail (this is "the ablation
study")

### 7.1 Design

Three occlusion conditions per slice, applied to the identical 59 correctly-classified
masked MSD slices used in Rounds 1–3, scored with the fold-appropriate held-out checkpoint
for each slice (never a checkpoint scored on its own training fold):

1. **Tumor occlusion:** the real, ground-truth pancreas+tumour mask region (label ∈ {1,2})
   is replaced with a neutral fill value equal to that image's own median intensity.
2. **Padding occlusion:** the image-specific detected synthetic-padding region (using the
   same exact-zero-margin detector developed in the §6 Round 2b padding investigation) is
   replaced with the same neutral-fill convention. **Only run where ≥100px of real padding
   was detected** — 52 of the 59 slices qualified (the padding-occlusion condition's n=52,
   distinct from the tumour/control conditions' n=59).
3. **Random control:** a same-area random patch of ordinary tissue is selected, constrained
   to avoid both the tumour region and any detected padding (placement-retry logic, up to
   30 attempts, matching the avoidance logic later reused by the Round 6
   `_apply_mask_preserving_erase` augmentation), then filled with the same neutral-fill
   convention. This is the essential baseline against which the other two conditions are
   judged — without it, a raw tumour/padding delta has no reference point for "how much
   does occluding *anything* this size move the prediction."

### 7.2 Output Space — the Mid-Analysis Methodological Correction

**First attempt (superseded):** measured Δ in **predicted probability** space
(`sigmoid(det_logit)`). Found no statistically significant effect in any of the three
conditions.

**Diagnosis:** baseline P(cancer) on these correctly-classified slices averages **0.9991**
— i.e., the sigmoid nonlinearity is already saturated at this operating point. Because
`d(sigmoid)/dx → 0` as `sigmoid(x) → 1`, no local input perturbation can meaningfully move a
probability that's already pinned near 1.0, *regardless of whether the perturbed region is
actually decision-relevant*. This is a measurement-space artifact, not evidence that
occlusion has no effect.

**Fix:** re-measured Δ in **logit space** (the model's raw, pre-sigmoid score,
`det_logit`), which is unbounded and does not saturate. This produced the informative
Round 4 result reported in §6.

### 7.3 Statistical Test

**Wilcoxon signed-rank test** (paired, non-parametric — appropriate since the same 59 (or
52, for padding) slices contribute one paired difference each between a treatment condition
and the control condition; no assumption of normality required). Two comparisons:
tumor-Δ-logit vs. control-Δ-logit (n=59 pairs), padding-Δ-logit vs. control-Δ-logit (n=52
pairs, restricted to the padding-detected subset).

### 7.4 Result Interpretation, Precisely Stated

- **Tumor occlusion, 0.47× control, p=0.0075:** the real tumour region, when blanked, moves
  the detection logit by *significantly less* than an arbitrarily-placed same-area random
  patch of ordinary tissue does. This is not "no different from control" — it is
  significantly *less* important than control, i.e. the model's decision depends *less* on
  the tumour region than on a randomly chosen region of comparable size.
- **Padding occlusion, 2.03× control, p=0.044:** the detected synthetic padding, when
  blanked, moves the detection logit by *significantly more* than the random-patch control
  — direct causal evidence the model's decision depends on the padding pattern.

---

## 8. Remediation Round 5 — Random-Resized-Crop Augmentation

### 8.1 Implementation

`imaging/slice_cache_dataset.py`, `SliceCacheDataset(..., augment=True)`. Random-resized-crop
applied **at train time only** (never at validation/test/inference time): each training
sample is cropped to a randomly-chosen **85–100% scale** of the original 320×320 frame, then
resized back to 320×320. This varies the absolute position and *proportion* of padding
present in any given training instance across epochs, removing padding-fraction-and-position
as a stable, memorizable per-slice fingerprint. `resnet50_unet` only —
`CANDIDATE_SPECS["resnet50_unet"]["augment"] = True` in
`imaging/train_segmentation_detection.ipynb`; `baseline_cnn` untouched (`augment=False`,
not the winning/promoted candidate).

### 8.2 Retraining

All 5 folds retrained on the RunPod RTX 6000 Ada pod (48GB VRAM). Pre-fix state archived at
`checkpoints/imaging/candidates_5fold_no_augmentation_archive/`.

### 8.3 Detection/Segmentation Metrics, Before vs. After (mean across 5 folds)

| Metric | Before (no augmentation) | After (augment=True) | Δ |
|---|---|---|---|
| ROC-AUC | 0.9938 | 0.9934 | −0.0004 (negligible) |
| Recall | 0.9833 | 0.9628 | −0.0205 |
| Precision | 0.9644 | 0.9920 | **+0.0276** |
| Specificity | 0.8614 | 0.9696 | **+0.1082** |
| F1 | 0.9736 | 0.9768 | +0.0032 |
| Mean stopped epoch | 5.4 | 8.0 | +2.6 (later plateau — consistent with genuine regularization) |
| Dice (segmentation) | 0.395 | 0.396 | ~unchanged |
| IoU (segmentation) | 0.282 | 0.283 | ~unchanged |

### 8.4 Occlusion Re-Verification, Before vs. After

| Condition | Before: ratio (p) | After: ratio (p) | Interpretation |
|---|---|---|---|
| Tumor occlusion | 0.47× (p=0.0075) | 0.62× (p=0.0065) | Numerically improved, **still statistically significant below 1×** — not fixed |
| Padding occlusion | **2.03× (p=0.0442)** | **1.08× (p=0.9168)** | **Statistically indistinguishable from control — fixed** |

### 8.5 Attribution Sweep Re-Run (secondary, correlational, for completeness only)

`results/imaging/confound_check_all_methods_metrics.csv`: Integrated Gradients enrichment
1.08× → 1.55×; EigenCAM 0.64× → 0.09×; all four gradient-based CAM methods remain ~0.00×.
Mixed-direction movement, explicitly noted as non-authoritative relative to the causal
occlusion result (Round 3's methods were already shown to be architecture/layer-biased).

### 8.6 Verdict

Padding shortcut: **resolved, causally re-verified with the identical test that originally
detected it.** Tumor under-reliance: **not resolved** — improved numerically (0.47×→0.62×)
but remains statistically significant in the same direction (below-control, i.e.
under-reliance). This is the version subsequently promoted to
`checkpoints/imaging/final/model.pt`.

---

## 9. Model Architecture Reference (for precise technical description)

`src/imaging/models.py::ResNet50UNet` — 71,876,484 total parameters.

- **Encoder:** `torchvision.models.resnet50`, `ResNet50_Weights.IMAGENET1K_V2` pretrained
  weights (when `pretrained=True`, the default; `pretrained=False` available to skip
  redundant weight loading when about to immediately overwrite via
  `load_state_dict()` from a trained checkpoint, used in the confound-check notebook).
  Stages: `stem` (conv1+bn1+relu, /2, 64ch) → `maxpool` (/4) → `layer1` (/4, 256ch) →
  `layer2` (/8, 512ch) → `layer3` (/16, 1024ch) → `layer4` (/32, 2048ch).
- **Decoder (U-Net):** 4× `UpBlock` (`up4`: 2048→1024 matching `layer3`; `up3`: 1024→512
  matching `layer2`; `up2`: 512→256 matching `layer1`; `up1`: 256→64 matching stem output),
  each performing `ConvTranspose2d(stride=2)` upsample → channel-concat with the matching
  encoder skip connection → two `Conv2d(k=3,p=1)+BatchNorm2d+ReLU` layers. Final
  `ConvTranspose2d(64→32)` restores full input resolution, followed by `seg_head =
  Conv2d(32, num_seg_classes=3, k=1)` producing per-pixel logits over
  {background, pancreas, tumour}.
- **Detection head:** `det_pool = AdaptiveAvgPool2d(1)` applied to `layer4`'s 2048-channel
  bottleneck feature map → `det_head = Linear(2048, 1)` → single scalar `det_logit`.
  Segmentation and detection are trained **jointly** off the same shared encoder; the
  detection head reads only the deepest encoder bottleneck (`layer4`), never anything from
  the decoder.
- **`DetectionOnlyWrapper`:** thin wrapper exposing only `det_logit` from
  `ResNet50UNet.forward()` (which natively returns `(seg_logits, det_logit)`), required
  because `pytorch-grad-cam`-style tooling expects a model whose forward pass returns
  exactly the scalar(s) to explain.
- **Input:** 3-channel, 320×320 (channel-replicated from the single-channel HU-windowed
  slice by the `Dataset`, not stored 3-channel on disk).
- **`baseline_cnn`** (control candidate): 97,761 params, from-scratch (no pretraining),
  detection head only, no segmentation head — `segmentation_metrics: null` by architectural
  design, not missing data.

---

## 10. Remediation Round 6 — Mask-Preserving Random Erase

### 10.1 Rationale

Round 5 fixed padding-reliance but left tumour under-reliance open. Hypothesis: if every
non-tumour region is made an *unreliable* training cue (randomly vanishing), the tumour
region — always present, never erased — should become the one signal the model can
consistently depend on, potentially shifting learned reliance toward it.

### 10.2 Implementation

`imaging/slice_cache_dataset.py`, new method `_apply_mask_preserving_erase`. Applies **only
to MSD rows** (`has_mask=True` — undefined operation on NIH, which has no ground-truth
tumour mask to avoid). With probability `MASK_ERASE_PROB=0.4` per sample, selects a random
square patch covering **5–15% of image area**, constrained via up-to-30 placement attempts
to never overlap the real tumour mask (reusing the same avoidance logic as
`random_control_region` in the occlusion test, §7.1 condition 3), filled with that image's
own median intensity (same neutral-fill convention as the occlusion test). Applied **in
addition to** the Round 5 random-resized-crop augmentation (both active simultaneously),
not as a replacement for it. `resnet50_unet` retrained, all 5 folds; `baseline_cnn`
untouched. Pre-Round-6 state archived at
`checkpoints/imaging/candidates_5fold_padding_fix_archive/` (etc., full path list in §12).

### 10.3 Detection/Segmentation Metrics, Round 5 vs. Round 6 (mean across 5 folds)

| Metric | Round 5 | Round 6 | Δ |
|---|---|---|---|
| ROC-AUC | 0.9934 | 0.9928 | −0.0006 |
| Recall | 0.9628 | 0.9790 | +0.0162 |
| Precision | 0.9920 | 0.9832 | −0.0088 |
| Specificity | 0.9696 | 0.9359 | −0.0337 |
| F1 | 0.9768 | 0.9811 | +0.0043 |
| Dice / IoU | 0.396 / 0.283 | 0.369 / 0.262 | slightly down |
| Mean stopped epoch | 8.0 | 5.4 | earlier plateau |

Characterized as "roughly a wash on headline numbers, no meaningful change" in the source
documentation — none of these deltas are being treated as evidence for or against the
remediation's success; the occlusion re-test (§10.4) is the deciding evidence.

### 10.4 Occlusion Re-Verification, Round 5 vs. Round 6

| Condition | Round 5 | Round 6 | Δ interpretation |
|---|---|---|---|
| Tumor occlusion vs. control | 0.62×, p=0.0065 (significant) | 0.58×, p=0.0148 (still significant) | **Statistically indistinguishable from Round 5 — remediation did not move this number** |
| Padding occlusion vs. control | 1.08×, p=0.9168 (fixed) | 2.86×, p=0.1353 (n=52, not significant) | **Point estimate elevated above even the original pre-fix Round 4 value (2.03×), but underpowered to confirm regression at this n** |

### 10.5 Attribution Sweep Re-Run (secondary)

Integrated Gradients enrichment 1.55×→1.76×; EigenCAM 0.09×→0.17×; all four gradient-based
CAM methods remain ~0.00–0.008×. Same non-authoritative, mixed-direction pattern as Round 5;
explicitly noted as uninformative for this specific question.

### 10.6 Decision

**Reverted.** Rationale, stated precisely: (a) no statistically demonstrated improvement on
the targeted metric (tumor-occlusion ratio), and (b) a real, not-statistically-confirmable-
but-not-excludable risk that the previously-resolved padding metric's point estimate had
moved substantially toward (and numerically past) the original problem's magnitude.
Expected-value reasoning: zero demonstrated upside + non-zero plausible downside on an
already-fixed axis → revert. `checkpoints/imaging/final/`,
`checkpoints/imaging/candidates/`, and the corresponding `results/imaging/*.csv`,
`outputs/eval/imaging/*.png`, and `imaging/imaging_confound_check.ipynb` executed state were
restored **byte-for-byte** to the verified Round 5 state. Round 6 artifacts fully preserved
in a parallel archive path (§12), not deleted — `_apply_mask_preserving_erase` remains in
the codebase, unused by the live model, with its docstring updated to record this outcome
for any future retry with different hyperparameters (e.g. higher `MASK_ERASE_PROB`, larger
erase-area range) or combination with a different remediation approach.

---

## 11. Current State (as of the last executed run) — Precise Status

| Axis | State | Evidence basis |
|---|---|---|
| Padding shortcut | **Resolved.** 1.08× control, p=0.9168 (statistically indistinguishable from no-effect baseline) | Round 5 occlusion re-test, causal, same methodology as the original Round 4 detection |
| Tumor under-reliance | **Open, unresolved.** Best achieved: 0.62× control, p=0.0065 (Round 5) — still significantly below 1× after 2 independent remediation attempts | Rounds 4/5/6 occlusion tests, all causal, all statistically significant in the under-reliance direction |
| Promoted model | `checkpoints/imaging/final/model.pt` = the Round 5 (padding-fix-only) state | Confirmed via byte-for-byte restoration after Round 6 revert |
| Disclosure mechanism | `checkpoints/imaging/final/model_card.json`'s `known_limitations.confound_check_summary` field, quoted verbatim into `checkpoints/fusion/final/model_card.json` and every place `resnet50_unet`'s metrics surface in `fusion_evaluation.ipynb` output | Verified present by direct read-back from disk after writing, per `docs/Fusion_documentation.md` |

**Two live, currently-indistinguishable hypotheses for the residual tumor-under-reliance
gap** (stated as open, not resolved, in the source documentation):
1. A subtler, not-yet-identified non-anatomical shortcut remains, distinct from the
   already-fixed padding mechanism.
2. Raw tumour-region pixels in isolation constitute a genuinely weaker standalone predictive
   signal than the full surrounding anatomical context (organ boundary shape, ductal/vessel
   involvement, adjacent tissue texture) that a real diagnostic read would also use jointly
   — i.e., the model may be using *legitimate* but *spatially diffuse* anatomical context
   rather than a shortcut, and the occlusion test as designed cannot distinguish "diffuse
   legitimate reasoning" from "diffuse illegitimate shortcut" with the evidence gathered.

No further test was run to adjudicate between these two hypotheses before the investigation
was paused (explicit decision, not abandonment — see §12 "if resuming" priority list).

---

## 12. Full Artifact Provenance (for verification / audit)

**Live (Round 5 state, currently promoted):**
- `checkpoints/imaging/candidates/resnet50_unet/{fold_0..fold_4}.pt` + `model_card.json`
- `checkpoints/imaging/final/model.pt` + `pod_training_run.json`
- `results/imaging/oof_predictions.csv`
- `results/imaging/confound_check_overlap_metrics.csv` (300 rows, Grad-CAM, all sampled MSD+NIH slices)
- `results/imaging/confound_check_integrated_gradients_metrics.csv` (65 rows, IG, masked MSD subset)
- `results/imaging/confound_check_all_methods_metrics.csv` (390 rows, all 6 methods, masked MSD subset)
- `results/imaging/confound_check_occlusion_sensitivity.csv` (65 rows, logit+probability deltas, all 3 conditions)
- `outputs/eval/imaging/gradcam_confound_check_fold{0,1,2,3,4}.png`,
  `confound_check_all_methods_comparison.png`, `confound_check_occlusion_sensitivity.png`
- `imaging/models.py`, `imaging/slice_cache_dataset.py`
  (`augment=True` random-resized-crop live; `_apply_mask_preserving_erase` present, unused)
- `imaging/imaging_confound_check.ipynb`, `imaging/train_segmentation_detection.ipynb`

**Archived, chronological:**
- `checkpoints/imaging/candidates_3fold_archive/` — pre-5-fold
- `checkpoints/imaging/candidates_5fold_no_augmentation_archive/{resnet50_unet,baseline_cnn}/`,
  `results/imaging/archive/oof_predictions_5fold_no_augmentation.csv`,
  `results/imaging/archive/confound_check_no_augmentation/confound_check_*.csv`,
  `outputs/eval/imaging/archive_no_augmentation/*.png`,
  `imaging/imaging_confound_check_no_augmentation_archive.ipynb` — Rounds 1–4, pre-padding-fix
- `checkpoints/imaging/candidates_5fold_padding_fix_archive/`,
  `checkpoints/imaging/final_padding_fix_archive/`,
  `results/imaging/archive/oof_predictions_padding_fix_archive.csv`,
  `results/imaging/archive/confound_check_padding_fix_archive/confound_check_*.csv`,
  `outputs/eval/imaging/archive_padding_fix/*.png`,
  `imaging/imaging_confound_check_padding_fix_archive.ipynb` — Round 5 snapshot (identical
  content to current Live, kept as explicit pre-Round-6 checkpoint)
- `checkpoints/imaging/candidates_5fold_mask_erase_archive/{resnet50_unet,baseline_cnn}/`,
  `results/imaging/archive/oof_predictions_mask_erase_archive.csv`,
  `results/imaging/archive/confound_check_mask_erase_archive/confound_check_*.csv`,
  `outputs/eval/imaging/archive_mask_erase/*.png`,
  `imaging/imaging_confound_check_mask_erase_archive.ipynb` — Round 6, reverted, unpromoted

---

## 13. If Resuming — Ranked Priority List (verbatim reasoning preserved)

1. ~~Mask-preserving random erase~~ — tried (Round 6), did not move the targeted metric.
   Not ruled out for retry with different hyperparameters (higher `MASK_ERASE_PROB`, larger
   erase-area range), but not the next default choice given a null result once already.
2. **Attention-supervision auxiliary loss** (currently top-ranked). Add a training loss term
   rewarding stronger detection-head-relevant activation inside the ground-truth mask than
   outside it, MSD rows only. Structural gap: the segmentation head is already supervised
   against the mask; the detection head currently has no gradient path linking it to mask
   geometry at all. Needs a differentiable attention-map proxy (e.g. Grad-CAM-style map
   computed in-graph, or a dedicated auxiliary pooling branch) and loss-weight tuning.
   Ranked above a third augmentation-only attempt because two independent augmentation-only
   interventions (random-resized-crop targeting padding, mask-preserving-erase targeting
   tumor-reliance) have both left this specific number unmoved or unresolved — suggests the
   fix needs to act more directly on the detection head's learned feature geometry rather
   than indirectly via input-distribution perturbation.
3. **Segmentation-gated detection** (architectural). Multiply encoder features by the
   predicted segmentation mask before the detection head's pooling step, structurally
   preventing the detection head from attending outside the plausible anatomical region.
   Strongest guarantee (rules out non-anatomical shortcuts by construction, not just by
   discouragement), highest risk/cost — new architecture to validate, changes what the
   "winning candidate" architecturally is, would require re-running the full 5-fold
   comparison and confound-check pipeline from scratch against the new architecture.
4. **Fix the packing pipeline at the source.** Crop to the real body/anatomy bounding box
   and resize (interpolate) instead of zero-padding to a fixed 320×320, removing the
   large-constant-value regions Round 4 demonstrated the model causally depends on. **Not
   currently actionable** — the raw pre-crop per-slice source imagery is not retained on the
   compute environment used for training; only the already-packed cache is available. A
   cheaper partial variant: exclude near-fully-padded, low-content slices (the tail
   identified in the §6 Round 2b padding investigation, concentrated in NIH) from training
   entirely.
5. **Acknowledged structural ceiling.** Since MSD is 100% cancer and NIH is 100% healthy by
   the dataset's own construction, no purely algorithmic intervention (augmentation,
   architecture, loss term) can *fully* eliminate the possibility of a residual
   dataset-level shortcut — only training data where disease status and
   scanner/institution/acquisition protocol are not perfectly collinear can close this gap
   completely. Stated as the honest gold-standard ceiling, acknowledged as outside this
   project's data-access scope.

---

## 14. Precise Numeric Index (for quick lookup / cross-referencing)

- Round 1 Grad-CAM enrichment: **0.003×**
- Round 2 Integrated Gradients enrichment: **1.081×**
- Round 2b padding top-margin Mann-Whitney p: **0.011**
- Round 3 gradient-CAM-family enrichment band: **0.003–0.006×**
- Round 3 EigenCAM enrichment: **0.640×**
- Round 4 tumor-occlusion ratio / p: **0.47× / p=0.0075**
- Round 4 padding-occlusion ratio / p: **2.03× / p=0.0442**
- Round 4 baseline saturated probability (motivating the logit-space switch): **0.9991**
- Round 5 (post-fix) padding-occlusion ratio / p: **1.08× / p=0.9168**
- Round 5 (post-fix) tumor-occlusion ratio / p: **0.62× / p=0.0065**
- Round 5 detection metrics (mean, 5-fold): **ROC-AUC 0.9934, Recall 0.9628, Precision
  0.9920, Specificity 0.9696, F1 0.9768**
- Round 6 tumor-occlusion ratio / p: **0.58× / p=0.0148**
- Round 6 padding-occlusion ratio / p: **2.86× / p=0.1353 (n=52, not significant)**
- Sample sizes: 200 sampled slices → 59 masked (overlap metrics), 52 padding-detected
  (padding-occlusion condition only), 100 NIH (border_fraction comparison only)
- Model size: **71,876,484 parameters** (`resnet50_unet`); **97,761** (`baseline_cnn`)

---

*Compiled 2026-08-12. This document is the maximal-detail companion to
`docs/Imaging_GradCAM_Confound_Supervisor_Briefing.md` (plain-language/narrative version) and
`docs/Presentation_Full_Documentation_and_QA.md` (general project reference). All numbers
here are traceable to `docs/Imaging_Confound_Check_documentation.md` and
`docs/Imaging_Session_Summary_2026-07-18.md`, which remain the canonical technical record if
any discrepancy is ever found.*
