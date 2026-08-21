# Imaging Confound Check — Grad-CAM / Segmentation-Overlap

**Run date:** 2026-07-17 (Rounds 1–5), continued 2026-07-18 (Round 6)
**Notebook:** `imaging/imaging_confound_check.ipynb` (standalone — see "Why standalone" below)
**Candidate checked:** `resnet50_unet` only (the winner from the 5-fold comparison — the model that would actually be promoted)
**See also:** `Imaging_Session_Summary_2026-07-18.md` for a shorter, narrative walkthrough of the 2026-07-18 session (Round 6 attempt + revert) with the same numbers presented alongside the rest of that session's work.

## The concern, precisely stated

In this dataset, `dataset` (MSD vs. NIH) and `class` (cancer vs. healthy) are perfectly correlated — every MSD patient is class=1, every NIH patient is class=0 (confirmed in the manifest: 281 MSD patients all class=1, 80 NIH patients all class=0). That means the detection head could reach very high accuracy by learning to recognize *which scanner/institution produced this image* rather than by recognizing actual pancreatic pathology. Every training-loss number and every metric already reported (ROC-AUC 0.994, recall 0.983) looks identical whether the model learned real pathology or this shortcut — this check is the only way in this project to tell the two apart.

## Why standalone, not filled into `imaging_evaluation.ipynb`

That notebook's earlier scaffold cells ("candidate comparison", "promotion to `checkpoints/imaging/final/`") are still unimplemented `raise NotImplementedError()` stubs. Executing that notebook top-to-bottom would halt before ever reaching the confound-check section. This notebook was built standalone, depending only on already-finished artifacts (`checkpoints/imaging/candidates/resnet50_unet/`, `results/imaging/oof_predictions.csv`, the packed cache) — it doesn't implement or touch the comparison/promotion logic, which is out of scope for this task.

## Method: quantitative anatomy-overlap on MSD, six independent attribution methods

For each of the 5 folds, the checkpoint **held out on that fold** loads a fresh, uncompiled `ResNet50UNet` (factored into a new `imaging/models.py`). Every method targets `layer4[-1]` (the last conv block feeding the detection head's global-average-pool) via a `DetectionOnlyWrapper` exposing just the detection logit — the head under suspicion, not the segmentation head — except Integrated Gradients, which attributes directly to input pixels.

40 correctly-classified MSD (cancer) true positives were sampled per fold (200 total); only **59 of 200 had a non-empty tumour/pancreas mask** — the `class` label is patient-level, and many individual axial slices of a cancer patient's scan (chest, hip, etc.) don't show the pancreas at all. This is a real finding from sampling, not a bug: the overlap metrics below are computed and averaged only over those 59 slices.

For each slice, "attended region" = top 20% of the heatmap by value. Three numbers per method:
- **`overlap_precision`**: what fraction of attended pixels are real anatomy
- **`overlap_recall`**: what fraction of real anatomy the attended region covers
- **`enrichment`**: `overlap_precision` ÷ chance precision (mask area fraction) — `enrichment > 1` means attention beats spatially-random chance
- **`border_fraction`**: what fraction of the attended region falls in the outer 10% image margin — computed on all 200 MSD + 100 NIH sampled slices (doesn't need a mask)

## Round 1: Grad-CAM alone — an alarming signal

| Metric (MSD, n=59 masked slices) | Value |
|---|---|
| Mean enrichment | **0.003x** |
| Mean overlap recall | 0.0005 |
| Mean border_fraction (MSD, n=200) | 0.346 |
| Mean border_fraction (NIH, n=100) | 0.311 |

Not noise — median overlap precision across all 59 masked slices was exactly 0.0, consistent across all 5 independently-trained folds. Visual panels (`outputs/eval/imaging/gradcam_confound_check_fold{0..4}.png`) showed several examples with Grad-CAM's hottest region sitting at the image's top corners, not the pancreas.

## Round 2: cross-check with Integrated Gradients — the methods disagreed

Same 59 slices, same checkpoints. Integrated Gradients (input-space, not layer4) showed **enrichment 1.081x** (roughly chance, not near-zero) and **border_fraction 0.116** (below the ~19% base rate a random 10%-margin border region would get, i.e. *not* border-concentrated). A genuine disagreement between two methods with different, unrelated failure modes — inconclusive on its own, so the check was extended rather than settled at this point.

## Investigation: is this caused by the BOX=320 crop/pad? (partial yes, more nuanced than first thought)

Checked directly against the cached images (raw pre-crop data isn't retained on this pod, but padding leaves a detectable signature: a contiguous margin of *exactly* 0.0, which real anatomy — even dark/air regions — doesn't produce, since real tissue has noise/texture).

**Confirmed: substantial synthetic padding exists**, added by the packing pipeline to reach exactly 320×320. With a properly-sized, statistically tested sample (n=300 per dataset — an earlier n=150 draw overstated the effect and is corrected here):
- **Bottom padding is substantial and nearly identical in both datasets** (median ~34–35px) — a shared packing artifact, not something that distinguishes MSD from NIH on its own.
- **Top padding differs with actual statistical significance** (Mann-Whitney p=0.011) — modest, NIH averages more.
- **NIH has a heavier tail of near-fully-padded, low-content slices** than MSD (95th percentile hits the maximum padding cap on every side for NIH; MSD's is much lower).
- Left/right padding differences did **not** reach significance (p=0.256, p=0.835).

**Conclusion on this question**: padding is real and does differ somewhat between MSD and NIH (giving the model *a* non-anatomical, dataset-correlated cue available to exploit), but it is not the dramatic, uniform difference a first quick look suggested. It's a plausible partial contributor, not a complete explanation by itself — which is why the check continued to a full 6-method sweep rather than stopping here.

## Round 3: six attribution methods, same 59 slices, same checkpoints — the deciding evidence

Added GradCAM++, HiResCAM, XGradCAM (three different gradient-weighting schemes through the same `layer4[-1]`) and EigenCAM (**completely gradient-free** — PCA on `layer4[-1]`'s activations, no backward pass at all). ScoreCAM/AblationCAM were excluded — perturbation-based methods needing one forward pass per channel (`layer4` has 2048), too expensive for a full fold sweep; a scoping decision, not a hidden gap.

| Method | Enrichment | Overlap recall | Border fraction |
|---|---|---|---|
| **Integrated Gradients** (input-space) | **1.081** | 0.216 | **0.116** |
| EigenCAM (gradient-free, layer4) | 0.640 | 0.209 | 0.329 |
| GradCAM++ (gradient, layer4) | 0.006 | 0.001 | 0.323 |
| GradCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |
| HiResCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |
| XGradCAM (gradient, layer4) | 0.003 | 0.001 | 0.324 |

**The pattern is clean and splits along two, not one, dimensions:**
1. **All four gradient-based methods through `layer4` are near-identical** (enrichment 0.003–0.006, essentially the same number despite genuinely different weighting math) — this consistency across four different gradient-weighting schemes is itself evidence the pattern isn't an idiosyncrasy of vanilla Grad-CAM specifically; it's shared by anything that backpropagates a gradient through this particular layer on these particular (heavily-padded) images.
2. **Removing just the gradient step** (EigenCAM, same layer): enrichment jumps to 0.640 — a >200x improvement, though still short of confidently-above-chance.
3. **Removing `layer4` entirely** (Integrated Gradients, input-space): enrichment reaches 1.081 (right at chance) with border_fraction dropping to a third of every layer4-based method's.

Visual confirmation (`outputs/eval/imaging/confound_check_all_methods_comparison.png`, 3 examples): **all four gradient-based methods fixate on the same top-corner region in every example, regardless of where the real tumour mask actually is.** EigenCAM is diffuse/non-specific in one example, corner-biased in the others. Integrated Gradients' hot pixels scatter through actual tissue content in all three examples — visibly different behavior, not just a different summary number.

## Round 3 interim read (superseded by Round 4 below)

At this point the tentative read was: "substantially a gradient/layer4-attribution-method artifact, not strong evidence of a confound" — four gradient-weighting schemes converging on the same near-zero number, and removing gradients (EigenCAM) then removing layer4 entirely (Integrated Gradients) each independently improving the picture. That reasoning was sound *given only attribution-method evidence*. But every method up to this point, including the "gradient-free" ones, still only **infers** importance from gradients or activations — none of them actually intervene on the input and watch what happens. Round 4 does that, and changes the conclusion.

## Round 4: Occlusion-Based Sensitivity Analysis — direct causal evidence, and it reverses the Round 3 read

**Method:** three occlusions per slice, same 59 MSD slices, same fold-appropriate held-out checkpoints as every method above — but this time nothing is inferred from gradients or activations. Regions are physically replaced with a neutral fill (the image's own median intensity) and the model is simply re-run:
1. **Tumor occlusion**: blank the real pancreas+tumour mask region.
2. **Padding occlusion**: blank *this specific image's* detected synthetic padding (via the same exact-zero-margin detector from the packing investigation; only run where ≥100px of real padding exists — 52 of 59 slices qualified).
3. **Random control**: blank a same-area random patch of ordinary tissue, away from both the tumour and any padding — the baseline "how much does occluding *anything* this size move the prediction" comparison, without which a raw tumor/padding delta has nothing to be judged against.

**A methodological correction made mid-analysis, worth stating plainly:** the first run measured deltas in *probability* space and found nothing significant anywhere. The reason wasn't that occlusion doesn't matter — baseline P(cancer) on these correctly-classified slices averages **0.9991**, i.e. the sigmoid is already saturated. At that operating point, no local occlusion of any kind can move the *probability* much, regardless of whether the occluded region is actually important. Switching to **logit space** (unbounded, doesn't saturate) fixed this and produced a genuinely informative result.

| Condition | Mean \|Δ logit\| | vs. control | Wilcoxon p (vs. control) |
|---|---|---|---|
| Tumor occlusion | 0.110 | **0.47x** | **p = 0.0075** |
| Padding occlusion | 0.475 | **2.03x** | **p = 0.0442** |
| Random control | 0.234 | — (baseline) | — |

Both differences are statistically significant, in a paired test, on the same 59 slices:

- **Occluding the real tumour region moves the prediction significantly *less* than occluding an arbitrary same-sized patch of tissue** (p=0.0075). Not "no different" — significantly *less* important than a random patch. This is a genuinely concerning, direct causal result: it is not merely that the model doesn't specially privilege the tumour region, but that comparably-sized patches of ordinary tissue carry more decision-relevant signal to this model than the tumour itself does.
- **Disrupting the detected padding pattern moves the prediction significantly *more* than occluding a random patch of tissue** (p=0.0442, 2.03x). This is direct, method-independent causal evidence that the synthetic padding added by the BOX=320 packing step is something the model's decision actually depends on.

Visual: `outputs/eval/imaging/confound_check_occlusion_sensitivity.png`.

## Final Verdict (supersedes the Round 3 interim read)

**Round 3's attribution-method evidence pointed toward "probably a measurement artifact." Round 4's direct causal evidence points the opposite way, and causal evidence from physically manipulating the input outranks inference from gradients/activations.** Two independent, statistically significant results, both on the same 59 held-out slices with the same checkpoints: the tumour region matters *less* than a random patch of tissue to this model's decision, and the packing-induced padding pattern matters *more* than a random patch. Together these are the strongest evidence in this whole investigation that **`resnet50_unet`'s near-perfect detection ROC-AUC (0.994) reflects a real, causally-demonstrated non-anatomical shortcut, not pixel-level pathological reasoning about the tumour** — not merely an inconclusive attribution-method read.

This does not prove the *specific* mechanism is "scanner/institution identity" (that would need controlled metadata manipulation this dataset's structure doesn't support), and it does not mean the model is *useless* — ROC-AUC 0.994 is a real, reproducible number, just not evidence of the kind of reasoning it might look like it reflects. But it does mean: **do not promote `resnet50_unet` to `checkpoints/imaging/final/` on the strength of its detection metrics without disclosing this finding**, and the remediation options below are no longer "nice to have if there's time" — the padding-disruption result specifically motivates trying to remove that shortcut before trusting this model further.

## How to solve this

Ranked roughly by cost/effort — not mutually exclusive. Priority changed given the Round 4 result: (1) is done (this is what produced the finding above); (2) is now the most directly motivated next step, not merely a good-practice suggestion.

1. ~~Occlusion-based sensitivity analysis~~ — **done above.** Result: significant evidence the model under-uses the tumour region and over-uses the padding pattern, relative to a random-patch control.

2. **Random crop/translate augmentation (cheap, requires retraining) — now the most directly motivated fix.** Add random translation to `SliceCacheDataset.__getitem__` (or the training loop) so the *absolute position and amount* of padding varies per epoch rather than being a stable per-slice fingerprint the model can key off. This targets exactly the mechanism Round 4 demonstrated matters causally. Standard practice, low implementation cost, one retraining run (~$1–2 at this pod's measured rate) — and re-running this same occlusion test afterward would directly confirm whether it worked (padding sensitivity should drop toward/below the control level).

3. **Fix the packing pipeline itself (moderate cost — the original raw per-slice source images aren't on this pod, only the already-packed cache, so this would need to happen wherever the raw data lives).** Instead of zero-padding to reach exactly 320×320, crop to the real body/anatomy bounding box and resize (interpolate) instead — removes the large constant-value padding regions Round 4 showed the model actually relies on. A cheaper partial fix: exclude near-fully-padded, low-content slices (the ones driving NIH's heavier tail found in the earlier investigation) from training.

4. **Attention-supervision / guided-attention auxiliary loss (most involved).** Add a training-time loss term that explicitly encourages the detection head's relevant features to align with the ground-truth mask on MSD slices. Directly steers the model toward the tumour region specifically (not just away from padding), at the cost of added training complexity and a full retrain.

5. **The structural fix this dataset can't support.** Since MSD is 100% cancer and NIH is 100% healthy, no amount of augmentation or architecture change can *fully* eliminate the possibility of a dataset-level shortcut — the only complete fix is data where class and dataset/scanner aren't perfectly correlated. Worth stating as the ideal-but-likely-infeasible gold standard for the write-up.

**Recommendation given the deadline and the Round 4 result**: (2) is now worth doing if there's any time left before the deadline — it's a cheap retrain that directly targets the demonstrated shortcut, and re-running this notebook's occlusion section afterward gives a clean before/after comparison. If there's no time to retrain, the finding itself must still be disclosed in the write-up — presenting `resnet50_unet`'s detection metrics without this caveat would misrepresent what the model has been shown to actually do.

## Round 5: Post-Augmentation Retrain — Verification

**Fix applied**: `resnet50_unet` retrained (all 5 folds, `checkpoints/imaging/candidates_5fold_no_augmentation_archive/` holds the pre-fix version) with `SliceCacheDataset(..., augment=True)` — a random-resized-crop (85–100% scale, resized back to 320×320) applied at train time only, varying how much of each image is padding per sample per epoch. `baseline_cnn` untouched. Implementation in `imaging/slice_cache_dataset.py`; training-loop wiring in `train_segmentation_detection.ipynb` (`CANDIDATE_SPECS["resnet50_unet"]["augment"] = True`).

**Detection metrics, before vs. after** (mean across 5 folds):

| Metric | Before | After |
|---|---|---|
| ROC-AUC | 0.9938 | 0.9934 |
| Recall | 0.9833 | 0.9628 |
| Precision | 0.9644 | **0.9920** |
| Specificity | 0.8614 | **0.9696** |
| F1 | 0.9736 | 0.9768 |
| Mean stopped epoch | 5.4 | 8.0 |
| Dice / IoU (segmentation) | 0.395 / 0.282 | 0.396 / 0.283 |

ROC-AUC essentially unchanged; precision and specificity jumped substantially (fewer false positives), recall dropped modestly, F1 net positive, segmentation unaffected. Training took longer to plateau (5.4 → 8.0 epochs) — consistent with genuine regularization, not a fluke.

**Occlusion sensitivity, before vs. after** (same test, same slices, new checkpoints):

| Condition | Before: vs. control | Before: p | After: vs. control | After: p |
|---|---|---|---|---|
| Tumor occlusion | 0.47x (significantly less) | 0.0075 | 0.62x (still significantly less) | 0.0065 |
| Padding occlusion | **2.03x (significantly more)** | **0.0442** | **1.08x (no longer distinguishable from control)** | **0.9168** |

**The padding shortcut is fixed.** Padding-occlusion sensitivity dropped from significantly *above* the random-patch control (2.03x, p=0.044) to statistically indistinguishable from it (1.08x, p=0.917) — the exact, specific mechanism the occlusion test demonstrated the model was relying on is gone. This is a clean causal result, not an attribution-method artifact — same test that found the problem, applied identically to the fix.

**Tumor under-reliance is not fixed.** The tumor region still moves the prediction significantly *less* than an arbitrary same-sized patch of tissue (0.62x, p=0.0065) — improved numerically from 0.47x, but still a real, significant effect in the wrong direction. Removing the padding shortcut did not make the model specifically prioritize the tumor pixels; it appears to now rely on some more diffuse combination of surrounding tissue instead of either the padding or the tumor specifically. That could reflect a subtler remaining shortcut, or it could partly reflect that raw tumor pixels alone (without surrounding anatomical context — organ shape, ductal/vessel involvement) are a genuinely weaker standalone signal than a holistic view of the region, similar to how radiologists don't read tumors in isolation either. This check cannot distinguish those two explanations with the evidence gathered.

Six-method attribution sweep also re-run for completeness (`results/imaging/confound_check_all_methods_metrics.csv`): Integrated Gradients' enrichment rose to 1.55x (from 1.08x), EigenCAM dropped to 0.09x (from 0.64x), all four gradient-based CAM methods sit at 0.00x. These attribution-method numbers moved in mixed directions post-fix and are noted for completeness, but the occlusion test above remains the authoritative evidence — it's causal, the attribution methods are correlational and were already shown (Round 3) to be architecture-biased on this model.

**Bottom line**: the specific, demonstrated confound (padding-as-shortcut) is resolved and verified by direct re-test. The model is not yet demonstrated to specifically rely on the tumor region more than generic tissue — an open question, not a new red flag, and one this project's timeline may not have room to chase further. `resnet50_unet`'s detection metrics are now on firmer ground than before this retrain, but should still be presented with the caveat that pixel-level tumor-specific reasoning remains unconfirmed.

## Round 6: Mask-Preserving Random Erase — attempted, did not help, reverted

**Decision to resume:** the user explicitly asked to resume remediation after reviewing the Round 5 result, with the instruction to keep everything properly archived/documented at each step (this is that continuation).

**Fix applied:** `SliceCacheDataset` (`imaging/slice_cache_dataset.py`) gained a second, independent training-time augmentation on top of the existing random-resized-crop: `_apply_mask_preserving_erase`. For MSD rows only (`has_mask=True` — NIH has no ground-truth mask, so "outside the tumour" isn't defined), with probability `MASK_ERASE_PROB=0.4`, a random square patch covering 5–15% of the image area is picked so that it never overlaps the real tumour mask (up to 30 placement attempts, matching the avoidance logic already used by `random_control_region` in the occlusion test), then filled with the image's own median intensity — the same neutral-fill convention validated by the occlusion test itself. Rationale: make every region *except* the tumour an unreliable, randomly-vanishing cue during training, so the tumour becomes the one signal that's always present. `resnet50_unet` retrained, all 5 folds, `baseline_cnn` untouched (not the winning candidate, `augment=False`). Archived before this run: `checkpoints/imaging/candidates_5fold_padding_fix_archive/`, `checkpoints/imaging/final_padding_fix_archive/`, `results/imaging/archive/confound_check_padding_fix_archive/`, `results/imaging/archive/oof_predictions_padding_fix_archive.csv`, `outputs/eval/imaging/archive_padding_fix/`, `imaging/imaging_confound_check_padding_fix_archive.ipynb`.

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

Roughly a wash on the headline numbers — no meaningful change, modestly earlier early-stopping.

**Occlusion sensitivity, Round 5 vs. Round 6 (same 59 MSD slices, same methodology, fold-matched held-out checkpoints):**

| Condition | Round 5 (after padding fix) | Round 6 (after mask-erase) |
|---|---|---|
| Tumor occlusion | 0.62x control, p=0.0065 (significant) | **0.58x control, p=0.0148 (still significant)** |
| Padding occlusion | 1.08x control, p=0.9168 (fixed) | **2.86x control, p=0.1353 (not significant, n=52)** |

**Verdict: this fix did not work, and its effect on padding is genuinely ambiguous rather than clearly safe.**

1. **Tumor under-reliance is unresolved, not improved.** 0.58x vs. control is statistically indistinguishable from Round 5's 0.62x (both significantly below 1x) — the mask-erase augmentation did not make the model rely more on the tumour region, which was its entire purpose.
2. **Padding sensitivity's point estimate got worse, but not to a confirmed degree.** The ratio jumped from 1.08x (essentially exactly at control, p=0.917) to 2.86x — numerically *larger* than the original pre-fix Round 4 value of 2.03x that was itself judged a real effect (p=0.044 at the time, larger n). Here p=0.135 with n=52 paired differences does not cross the 0.05 threshold, so this cannot be reported as "the padding shortcut came back" — but it equally cannot be reported as "still fixed" the way a purely threshold-based read would suggest. Most likely explanation: the added erase augmentation increases training noise/regularization pressure in a way that made padding-reliance variance-y again, without the sample size here to resolve whether it's a real partial regression or noise.
3. Six-method attribution sweep, re-run for completeness (`results/imaging/archive/confound_check_mask_erase_archive/confound_check_all_methods_metrics.csv`): Integrated Gradients enrichment 1.76x (from 1.55x), EigenCAM 0.17x (from 0.09x), all four gradient-based CAM methods still ~0.00x. Same mixed-bag pattern as Round 5 — not informative for this specific question, noted for completeness only.

**Decision: reverted.** Given (a) no demonstrated improvement on the targeted problem and (b) a not-fully-ruled-out risk of nudging the already-fixed padding shortcut back toward significance, `checkpoints/imaging/final/` and `checkpoints/imaging/candidates/` (plus the corresponding `results/imaging/*.csv`, `outputs/eval/imaging/*.png`, and `imaging/imaging_confound_check.ipynb`'s executed state) were restored byte-for-byte to the Round 5 (padding-fix-only, verified) versions. Round 6's checkpoints and results are preserved, not discarded — archived at `checkpoints/imaging/candidates_5fold_mask_erase_archive/`, `results/imaging/archive/confound_check_mask_erase_archive/`, `results/imaging/archive/oof_predictions_mask_erase_archive.csv`, `outputs/eval/imaging/archive_mask_erase/`, and `imaging/imaging_confound_check_mask_erase_archive.ipynb`. `imaging/slice_cache_dataset.py`'s `_apply_mask_preserving_erase` method is left in the codebase (it's not incorrect, just unproven) with its module docstring updated to record this outcome, in case a future attempt wants to retry with different hyperparameters (higher `MASK_ERASE_PROB`, larger erase area) or combine it with option 2 below rather than relying on it alone.

## Status: stopped here by decision, 2026-07-18 — not abandoned, a deliberate checkpoint

Investigation paused after Round 6's negative result, by explicit decision, with the option to resume later if time permits before the deadline.

**If resuming, in order of effort vs. how directly each targets the remaining problem** (the tumour region still moving the prediction *less* than a random patch of tissue, 0.58–0.62x control across two independent fixes now):

1. ~~Mask-preserving random erasing~~ — **tried in Round 6, did not help.** Left in the codebase for a possible retry with different hyperparameters, but not as the next thing to reach for unmodified.

2. **Attention-supervision auxiliary loss (moderate effort, most principled direct fix, now the top candidate).** Add a training loss term that explicitly rewards stronger detection-head activation inside the real tumour/pancreas mask than outside it (MSD rows only). The segmentation head is already supervised toward the mask; the detection head currently isn't linked to it at all. Needs a differentiable attention-map proxy and some loss-weight tuning; same retrain/verify cost pattern as the prior two rounds. Worth prioritizing over another try at random erasing, since two rounds of augmentation-only fixes (crop, then erase) have now failed to move this specific number.

3. **Segmentation-gated detection (biggest change, strongest guarantee, most risk).** Architectural change: multiply encoder features by the predicted segmentation mask before the detection head's pooling step, so it structurally cannot see outside the plausible anatomical region. Rules out non-anatomical shortcuts by construction rather than discouraging them, but is the most invasive option — new architecture to debug, and changes what the "winning candidate" architecturally is.

4. **Fix the packing pipeline at the source (not currently actionable on this pod).** Crop to the real body bounding box instead of zero-padding, before BOX=320 packing. Needs the raw pre-crop per-slice data, which isn't on this pod — only the already-packed cache is. Only actionable if the raw data becomes available.

**If not resuming**: the finding stands as documented above and should be disclosed as a limitation of the imaging branch in the write-up — specifically, that `resnet50_unet`'s detection metrics are not confirmed evidence of tumour-specific pixel-level reasoning, notwithstanding the padding-shortcut fix, and that one further attempt (mask-preserving erase) did not improve this and was reverted. This is a legitimate, defensible, fully-verified result either way.

## Artifacts

**Live (Round 5, padding-fix-only — current and promoted; Round 6 was reverted, see above):**
- `checkpoints/imaging/candidates/resnet50_unet/{fold_0..fold_4}.pt` + `model_card.json` (padding-fix augmented retrain)
- `checkpoints/imaging/final/model.pt` + `pod_training_run.json` (all-data fit derived from the above)
- `results/imaging/oof_predictions.csv` (both candidates, 5 folds, all 361 patients, augmented resnet50_unet)
- `results/imaging/confound_check_overlap_metrics.csv` (300 rows — Grad-CAM, all MSD + NIH sampled slices)
- `results/imaging/confound_check_integrated_gradients_metrics.csv` (65 rows — IG, masked MSD subset)
- `results/imaging/confound_check_all_methods_metrics.csv` (390 rows — all 6 attribution methods, masked MSD subset)
- `results/imaging/confound_check_occlusion_sensitivity.csv` (65 rows — logit and probability deltas, all three occlusion conditions)
- `outputs/eval/imaging/gradcam_confound_check_fold{0,1,2,3,4}.png`, `confound_check_all_methods_comparison.png`, `confound_check_occlusion_sensitivity.png`
- `imaging/models.py` (`ResNet50UNet`/`UpBlock`/`DetectionOnlyWrapper`, for reloading checkpoints outside the training notebook)
- `imaging/slice_cache_dataset.py` (`augment=True` random-resized-crop; also now has `_apply_mask_preserving_erase`, present but not currently used by the live/promoted checkpoints — see Round 6)
- `imaging/imaging_confound_check.ipynb`, `imaging/train_segmentation_detection.ipynb` (both reflect the Round 5 verified state; the training notebook's Round 6 cells document the attempt and revert inline)

**Archived (chronological, for the before/after comparisons above):**
- `checkpoints/imaging/candidates_3fold_archive/`, `results/imaging/archive/oof_predictions_3fold.csv` (pre-5-fold)
- `checkpoints/imaging/candidates_5fold_no_augmentation_archive/{resnet50_unet,baseline_cnn}/`, `results/imaging/archive/oof_predictions_5fold_no_augmentation.csv`, `results/imaging/archive/confound_check_no_augmentation/confound_check_*.csv`, `outputs/eval/imaging/archive_no_augmentation/*.png`, `imaging/imaging_confound_check_no_augmentation_archive.ipynb` (Rounds 1–4, pre-padding-fix)
- `checkpoints/imaging/candidates_5fold_padding_fix_archive/`, `checkpoints/imaging/final_padding_fix_archive/`, `results/imaging/archive/oof_predictions_padding_fix_archive.csv`, `results/imaging/archive/confound_check_padding_fix_archive/confound_check_*.csv`, `outputs/eval/imaging/archive_padding_fix/*.png`, `imaging/imaging_confound_check_padding_fix_archive.ipynb` (Round 5 — this is the same content as the current "Live" set above, kept as an explicit pre-Round-6 snapshot)
- `checkpoints/imaging/candidates_5fold_mask_erase_archive/{resnet50_unet,baseline_cnn}/`, `results/imaging/archive/oof_predictions_mask_erase_archive.csv`, `results/imaging/archive/confound_check_mask_erase_archive/confound_check_*.csv`, `outputs/eval/imaging/archive_mask_erase/*.png`, `imaging/imaging_confound_check_mask_erase_archive.ipynb` (Round 6 — the reverted, unpromoted attempt)
