# Imaging Branch: Explaining the Numbers, the Grad-CAM Finding, and Our Remediation Efforts

**Purpose of this document:** a supervisor-facing briefing you can present from directly —
plain-English explanations, the full chronological story of how the problem was found, every
method we tried to fix it, and a script-style Q&A for the meeting itself. This is the one
part of the project most likely to draw hard questions, and it's also the part that shows
the most rigorous science in the whole FYP — the goal here is to help you present it as a
strength ("we tested our own model and caught something"), not as a weakness to apologize
for.

**Source material:** `docs/Imaging_Confound_Check_documentation.md` (full technical record,
Rounds 1–6), `docs/Imaging_Session_Summary_2026-07-18.md`, `docs/Imaging_Evaluation_documentation.md`,
`docs/Imaging_5Fold_Training_Results_documentation.md`.

---

## 1. The 30-second version (say this first, then let them ask for detail)

> "Our imaging model gets 99.3% ROC-AUC and 96.3% recall detecting pancreatic cancer on CT
> slices. Before trusting that number, we ran an explainability check to see *where* the
> model was looking — and found it wasn't looking at the tumour, it was partly looking at a
> processing artifact. We traced the artifact to a specific step in our preprocessing
> pipeline, fixed it, and verified the fix worked with a direct causal test. One remaining
> issue — the model still doesn't specifically prioritize the tumour region over
> surrounding tissue — we attempted to fix twice, it didn't work, and we're disclosing that
> honestly rather than hiding it. The detection numbers are real and reproducible; what
> they represent is now a documented, evidence-based question, not an assumption."

That's the whole story in one breath. Everything below is the evidence behind each sentence.

---

## 2. What the numbers actually are, and what they legitimately show

| Metric | Value | What it means |
|---|---|---|
| ROC-AUC | 0.9934 | Across all threshold choices, the model separates cancer slices from healthy slices almost perfectly |
| Recall (sensitivity) | 0.9628 | It correctly flags ~96% of actual cancer slices |
| Precision | 0.9920 | Of the slices it flags as cancer, ~99% really are |
| Specificity | 0.9696 | Of healthy slices, ~97% are correctly left unflagged |

**These numbers are real and reproducible** — measured over full 5-fold cross-validation
(every one of 361 patients held out exactly once), not a lucky single split. They are not
in dispute. **What is in question is *why* the model achieves them** — that's a completely
separate question from *whether* it achieves them, and it's important to keep those two
things distinct when you present this. A model can be measurably accurate and still be
right for a reason you haven't confirmed yet.

**Why that distinction matters here specifically:** in this dataset, the cancer patients
(MSD source) and the healthy patients (NIH source) come from two different collections
scanned on different equipment at different institutions. Every cancer patient is from MSD;
every healthy patient is from NIH. That means "cancer vs. healthy" and "which dataset this
image came from" are the same split in our data — so a model could theoretically hit 99%
AUC by learning to recognize the scanner/institution's imaging signature instead of the
disease. Both explanations would produce an identical AUC number. This is the exact reason
we didn't stop at reporting the metric.

---

## 3. How we found the problem — the Grad-CAM story, in order

### Step 1 — We asked "where is the model actually looking?"

**Grad-CAM** (Gradient-weighted Class Activation Mapping) is a standard explainability
technique for CNNs: it produces a heatmap over the input image showing which pixels most
influenced the model's prediction, computed from the gradient of the output flowing back
into the last convolutional layer. If the model is reasoning about the tumour, the hottest
region of that heatmap should sit over the pancreas/tumour.

**What we found:** it didn't. Across correctly-classified cancer slices, the Grad-CAM
heatmap's hottest region concentrated in the image corners, not on the pancreas. Quantified
two ways:
- **Enrichment score** (how much more the model's attention overlaps real tumour tissue
  than random chance would predict): **0.003×** — essentially zero. A value of 1.0× would
  mean "no better than chance"; anything meaningfully above 1.0× is what you'd want to see.
- This pattern was **consistent across all 5 independently-trained cross-validation folds**
  — not a fluke of one run.

### Step 2 — We didn't trust one method's word for it

A single explainability method disagreeing with intuition isn't automatically damning —
attribution methods have known failure modes and artifacts of their own. So before
concluding anything, we cross-checked with a second, differently-designed method:
**Integrated Gradients**, which attributes importance directly to input pixels rather than
through a late convolutional layer. It told a *different* story: enrichment around 1.08×
(roughly chance — not the near-zero signal Grad-CAM showed) and much less concentration in
the image border. **The two methods disagreed with each other** — which meant the question
wasn't settled, it needed to be investigated further, not just reported as a red flag.

### Step 3 — We checked the preprocessing pipeline directly for an explanation

Before assuming this was purely a model behaviour, we checked whether it could be a data
artifact. Our preprocessing pipeline pads every CT slice to a fixed 320×320 size, and
padding leaves a detectable signature (a contiguous region of exactly zero intensity, which
real tissue — even dark, air-filled regions — never produces because real tissue has noise
and texture). We measured this directly against the cached images: **real, measurable
synthetic padding exists**, and its *amount* differs somewhat, with statistical
significance, between the cancer-source and healthy-source datasets (Mann-Whitney
p=0.011 on top-padding). This gave the model a non-anatomical, dataset-correlated cue it
could exploit — plausible, but on its own not proof the model was actually using it.

### Step 4 — We ran a full six-method attribution sweep to see if this was one method's
quirk or a real pattern

We added four more attribution methods on top of the original two — GradCAM++, HiResCAM,
XGradCAM (three different mathematical variants of the same gradient-through-a-late-layer
idea), and EigenCAM (a completely gradient-free method, based on PCA of the layer's
activations, with no backward pass at all).

**Result — a clean, two-part pattern:**
1. **All four gradient-based methods through the same layer agreed with each other almost
   exactly** (enrichment 0.003–0.006×) — four independently-designed weighting schemes
   converging on the same near-zero number is itself informative: it's not one method's
   idiosyncrasy.
2. **Removing the gradient step (EigenCAM) jumped enrichment to 0.640×** — over 200× better,
   though still not confidently above chance.
3. **Removing the late convolutional layer entirely (Integrated Gradients, input-space)
   reached ~1.08×** — right around chance, no border concentration.

**Our interim read at this point** (later overturned, and we say so honestly): this looked
like it might be substantially an artifact of *how gradient-based attribution behaves at
that specific layer on these specific heavily-padded images*, rather than strong evidence
of a real confound. That was a reasonable read given the evidence available — but it turned
out to be incomplete, which is exactly why we didn't stop there.

---

## 4. The critical insight: every method so far only *infers*, none of them *test*

This is the single most important methodological point to make to your supervisors, because
it's what elevates this from "we made a nice chart" to "we ran a real experiment":

> Every attribution method above — including the gradient-free one — only *infers*
> importance from gradients or internal activations. None of them actually change the input
> and observe what happens. That's an important limitation: gradient-based explanations are
> known in the literature to have their own failure modes (saturation, layer sensitivity,
> noise), so agreement or disagreement between them is suggestive, not conclusive.

We closed that gap with a **direct causal experiment**.

---

## 5. The occlusion-based ablation study — our causal test

**What "ablation" means here:** in the general machine-learning sense, an ablation study
means systematically *removing or disabling* part of a system's input (or architecture) and
measuring the effect on its output, to establish whether that part is actually responsible
for the behaviour you observe — as opposed to merely *correlated with* it. That is precisely
what we did to the model's **input**: this is an **input-ablation / occlusion sensitivity
study**, run as a direct causal test of what the detection head actually depends on.

*(A note on terminology, for transparency: there is also a specific attribution technique
in the literature called "AblationCAM," which perturbs internal feature-map channels one at
a time. We deliberately scoped that particular method out — it needs one forward pass per
channel, and the last convolutional layer here has 2,048 channels, making it prohibitively
expensive to run across a full 5-fold sweep. Our ablation study is a different, and for our
purposes more directly useful, design: we ablate regions of the actual input image, not
internal channels, and measure the real change in the model's output — described below.)*

### Method

For each of 59 correctly-classified cancer slices (sampled across all 5 folds, using each
fold's own held-out model — never testing a model on data it was trained on), we ran three
conditions, each by physically replacing (blanking) a region with a neutral fill value (the
image's own median intensity) and simply re-running the model:

1. **Tumour occlusion** — blank the real, ground-truth pancreas+tumour region.
2. **Padding occlusion** — blank the detected synthetic padding specific to that image.
3. **Random control** — blank a same-sized random patch of ordinary tissue, away from both
   the tumour and any padding. This is the essential baseline: it answers "how much does
   blanking *anything* this size move the prediction," so the tumour/padding results have
   something meaningful to be compared against, rather than being read in isolation.

**A real methodological correction we had to make mid-analysis, worth mentioning to show
rigor rather than hiding it:** our first attempt measured the change in *predicted
probability* and found nothing significant anywhere. The reason wasn't that occlusion
doesn't matter — it's that the model's probability output on these confidently-correct
slices is already saturated near 1.0 (average 0.9991), so the sigmoid curve is flat there:
no local occlusion can move a probability much regardless of whether the occluded region is
actually important. We switched to measuring the change in **logit space** (the model's raw,
unsquashed score, which doesn't saturate) — and that's what produced the real, informative
result below.

### Result — direct causal evidence

| Condition | Mean absolute change in model output | vs. random-patch control | Statistical significance |
|---|---|---|---|
| Tumour occlusion | 0.110 | **0.47× the control's effect** | p = 0.0075 |
| Padding occlusion | 0.475 | **2.03× the control's effect** | p = 0.0442 |
| Random control | 0.234 | (baseline) | — |

Both results are statistically significant, from a paired test on the same 59 slices:

- **Blanking the real tumour moves the model's prediction significantly *less* than
  blanking an arbitrary same-sized patch of ordinary tissue.** Not "no different" — the
  tumour region is measurably *less* important to this model than a random patch of tissue
  is. This is the direct, causal confirmation that the model is not doing tumour-specific
  reasoning the way its accuracy numbers might suggest.
- **Disrupting the padding pattern moves the prediction significantly *more* than blanking a
  random patch.** This is direct, method-independent, causal evidence that the synthetic
  padding introduced by our own preprocessing pipeline is something the model's decision
  actually depends on.

**Why this result overturned our Step-4 interim read:** causal evidence from physically
manipulating the input outranks inference from gradients or activations, because it isn't
subject to the same known failure modes. The earlier "maybe it's just a gradient-attribution
artifact" reading was reasonable given the evidence available at the time — but a real,
statistically significant, directly causal result takes priority once we had one.

---

## 6. Remediation effort #1 — fixing the padding shortcut (successful)

**Diagnosis → fix, directly targeted at the demonstrated mechanism:** since the occlusion
test showed the model depends on padding *position and amount as a stable per-slice
fingerprint*, we added **random-resized-crop augmentation** at training time — each training
image is randomly cropped to 85–100% of its area and resized back, so the exact amount and
position of padding varies from epoch to epoch and can no longer serve as a reliable cue.
Retrained all 5 folds.

**Verification — the same occlusion test, re-run on the new checkpoints:**

| Condition | Before (Round 4) | After fix (Round 5) |
|---|---|---|
| Padding occlusion vs. control | **2.03×, p=0.044 (significant)** | **1.08×, p=0.917 (statistically indistinguishable from control)** |
| Tumour occlusion vs. control | 0.47×, p=0.0075 | 0.62×, p=0.0065 (still significant) |

**The padding shortcut is fixed, and this is a clean result, not a partial one:** the exact
metric that flagged the problem, applied identically after the fix, shows the effect is
gone. This is the strongest kind of evidence available — same test, same slices, same
methodology, before and after — not a different, more favourable measurement chosen after
the fact.

**Detection performance, before vs. after the fix (mean across 5 folds) — the fix did not
come at the cost of accuracy:**

| Metric | Before augmentation | After augmentation |
|---|---|---|
| ROC-AUC | 0.9938 | 0.9934 (essentially unchanged) |
| Recall | 0.9833 | 0.9628 (modest drop) |
| Precision | 0.9644 | **0.9920** (substantial improvement) |
| Specificity | 0.8614 | **0.9696** (substantial improvement) |
| F1 | 0.9736 | 0.9768 (net improvement) |

Precision and specificity improved substantially (far fewer false positives), recall
dropped modestly, and the model took longer to plateau during training (5.4→8.0 epochs on
average) — consistent with the augmentation acting as genuine regularization, not a fluke.
**This is the version currently promoted and deployed** (`checkpoints/imaging/final/`).

### 6a. Implementation, in enough detail to explain "how", not just "that"

**Why crop, not the simpler idea of just shifting/translating the image?** This is worth
leading with if asked "how" — it shows the fix was targeted at the actual mechanism, not a
generic guess. The detection head reads a single pooled feature vector from the last
convolutional layer via **global average pooling (GAP)**, and GAP is translation-invariant —
shifting the image around leaves the *fraction* of the frame that's padding completely
unchanged, only its position moves. Since the occlusion test showed the model depends on
padding as a stable per-slice cue, and GAP would still let that cue survive a plain shift
untouched, translation alone would not have fixed anything. What needed to vary was the
padding *fraction itself* — which is exactly what a **random-resized-crop** does.

**Exact mechanism (`imaging/slice_cache_dataset.py`):** each training image is cropped to a
randomly chosen **85–100% scale window** of the 320×320 canvas, then resized back up to
320×320. The segmentation mask receives the identical crop box (so image and mask stay
pixel-aligned) but a different interpolation mode: **bilinear** for the image, **nearest-
neighbour** for the mask — bilinear on integer class labels (background/pancreas/tumour)
would invent invalid fractional label values at patch boundaries, which nearest-neighbour
avoids.

**Where it's applied:** a boolean flag (`augment=True`) on the dataset class, active **train-
time only** — every validation/test/inference pass still sees the real, unmodified image, so
what the model is being *evaluated* against never changed, only what it's *trained* against.
Enabled for `resnet50_unet` only (the candidate being promoted); `baseline_cnn` was left
untouched.

**Verification protocol:** full retrain, all 5 cross-validation folds, same splits as every
other run in this project. Then the *identical* occlusion test (same 59 slices, same
methodology) was re-run against the new checkpoints — not a different or more favourable
measurement chosen after the fact.

---

## 7. Remediation effort #2 — trying to fix tumour under-reliance (attempted, did not work,
honestly reverted)

The padding fix didn't address the *other* half of the problem: the tumour region still
mattered significantly *less* than a random patch of tissue (0.62× control). We made one
further, deliberate attempt at fixing this too, rather than stopping at the first success.

**Fix attempted:** **mask-preserving random erase** — during training, for cancer-source
images only, we randomly blanked a 5–15%-area patch of tissue chosen so it never overlaps
the real tumour region, with 40% probability per image. The idea: make every region *except*
the tumour an unreliable, randomly-vanishing training signal, so the tumour becomes the one
part of the image the model can always rely on. Retrained all 5 folds.

**Result:**

| Condition | Round 5 (padding-fix only) | Round 6 (+ mask-erase attempt) |
|---|---|---|
| Tumour occlusion vs. control | 0.62×, p=0.0065 (significant) | 0.58×, p=0.0148 (still significant — essentially unchanged) |
| Padding occlusion vs. control | 1.08×, p=0.917 (fixed) | 2.86×, p=0.135 (not significant, but numerically the highest value seen in the whole investigation, even above the original problem) |

**Verdict — this fix did not work, and we say so plainly rather than quietly dropping it
from the story:**
1. Tumour under-reliance was statistically unchanged — the augmentation did not make the
   model rely on the tumour more, which was its entire purpose.
2. The already-fixed padding metric's *point estimate* rose substantially, though not to
   statistical significance at this sample size — not proof the shortcut came back, but not
   proof it stayed fixed either. A genuinely ambiguous result on a previously-solved
   problem.

**Decision: reverted.** With no demonstrated benefit and a real, not-fully-ruled-out risk of
nudging a fixed problem back toward significance, we restored the model to the verified
Round 5 state. **Nothing from this attempt was discarded** — the code, checkpoints, and
results are fully archived, specifically so a future attempt (with different hyperparameters,
or a different technique entirely) doesn't have to start from zero.

### 7a. Implementation, in enough detail to explain "how", not just "that"

**The reasoning behind the specific technique chosen:** fix #1 solved the padding problem
but left the tumour-under-reliance finding completely untouched — it targeted a different
mechanism. The hypothesis behind fix #2: if the tumour region is under-relied-on, maybe it's
because everything *else* in the image currently functions as a more stable, dependable cue
than the tumour is. So the fix tries to remove that stability from everywhere *except* the
tumour, forcing the tumour to become the one region the model can always count on.

**Exact mechanism (`_apply_mask_preserving_erase` in `imaging/slice_cache_dataset.py`):**
applied only to cancer-source (MSD) training images — never NIH, which has no tumour mask,
so "a region outside the tumour" isn't even a defined concept there. With **40% probability**
per image, a random square patch covering **5–15% of the image area** is chosen and blanked,
filled with **that image's own median pixel intensity** — deliberately the same neutral-fill
convention the occlusion test itself uses, so the augmentation doesn't introduce some new
artificial black/white region the model could start keying off instead (which would just be
trading one shortcut for another). Patch placement is retried up to **30 times** to find a
square that doesn't overlap the real tumour mask; if none of the 30 attempts succeeds, the
last attempted placement is used anyway rather than looping indefinitely. This augmentation
stacks **on top of** fix #1's crop augmentation, not in place of it — both are active
simultaneously during this retrain.

**Verification protocol — identical to fix #1:** full 5-fold retrain, then the same
occlusion test re-run on the new checkpoints, directly comparable to the Round 5 numbers
because nothing about the test itself changed, only the model being tested.

**On hyperparameter choices, if asked directly:** 85% (crop lower bound), 40% (erase
probability), 5–15% (erase area), and 30 (placement retries) were reasonable first-pass
engineering choices, not the product of a hyperparameter sweep — worth stating plainly if
asked, rather than implying a search happened that didn't. The Round 6 write-up explicitly
leaves the door open to retrying with different values (higher probability, larger area) as
a next step, precisely because these specific values were never tuned.

---

## 8. Honest current status of the promoted model

| Issue | Status |
|---|---|
| **Padding shortcut** (model relying on a preprocessing artifact instead of anatomy) | **Fixed and verified**, via a direct causal re-test after augmentation |
| **Tumour under-reliance** (model relying on the real tumour region *less* than ordinary tissue) | **Still open.** Two remediation attempts made (augmentation-based); neither resolved it. Not a new finding — the same open issue, disclosed everywhere the model's numbers are reported downstream (its own model card, the fusion model card, the fusion evaluation notebook). |

**The practical, presentable conclusion:** the detection metrics (ROC-AUC 0.993, recall
0.963) are strong and reproducible, but **are not yet confirmed evidence of tumour-specific,
pixel-level pathological reasoning**. That sentence, almost verbatim, is written directly
into the model's own `model_card.json` under `known_limitations` — this isn't something we
are choosing to admit only in this meeting; it's a permanent, disclosed part of the shipped
artifact.

**Two honest, currently-indistinguishable explanations for the remaining gap** (worth
stating if a supervisor pushes on "so what's actually happening"):
1. There may be a subtler, still-unidentified non-anatomical shortcut we haven't found yet.
2. Raw tumour pixels in isolation may genuinely be a weaker standalone signal than the
   broader surrounding anatomical context (organ shape, duct/vessel involvement) — not
   unlike how a radiologist doesn't read a tumour in complete isolation from the organ
   around it either. Our occlusion test can't distinguish these two explanations with the
   evidence gathered so far.

---

## 9. What we didn't get to (be upfront if asked "what's next")

Ranked by how directly each targets the *remaining* problem, in order of what we'd try if
given more time — this is a real, considered plan, not a vague "future work" placeholder:

1. **Attention-supervision auxiliary loss** (top-ranked next step) — add a training loss
   term that explicitly rewards the detection head's activations being stronger inside the
   real tumour mask than outside it, on cancer-source images. The segmentation head is
   already supervised against the mask; the detection head currently has no such link.
   Two rounds of augmentation-only fixes (crop, then erase) have both failed to move this
   number, which is why this more targeted, structural approach is now ranked above trying
   a third augmentation variant.
2. **Segmentation-gated detection** — a more invasive architectural change: multiply the
   encoder features by the predicted segmentation mask before the detection head pools
   them, so the detection head structurally *cannot* see outside the plausible anatomical
   region. Strongest guarantee, but the most invasive change and the most new debugging
   surface.
3. **Fix the packing pipeline at the source** — crop to the real body bounding box instead
   of zero-padding, before the fixed-size packing step, which would remove the padding
   artifact at its origin rather than training around it. Not currently actionable: the raw
   pre-crop imaging data isn't available on the compute environment we used — only the
   already-packed cache is.
4. **The structural fix this dataset can't support** — worth naming as the honest gold
   standard: since the cancer cohort and healthy cohort come from entirely different source
   collections, no amount of augmentation or architecture change can *fully* rule out a
   dataset-level shortcut. The only complete fix is data where disease status and
   scanner/institution aren't perfectly correlated — outside this project's data access.

---

## 10. Anticipated supervisor questions — script-style answers

**Q: "If the tumour matters less than random tissue, is the model just wrong?"**
A: No — "matters less to the decision" isn't the same as "not detecting cancer." The model's
detection accuracy is real and well-validated. What we've shown is that we can't yet confirm
it's using tumour-specific pixel evidence to get there — it may be using a broader,
still-legitimate signal (e.g., whole-organ or whole-slice context) that we haven't fully
characterized, alongside the resolved padding artifact. We've been precise about what we can
and can't claim, rather than either overclaiming interpretability or dismissing the accuracy.

**Q: "Why should we trust the occlusion test more than Grad-CAM, when Grad-CAM is the more
standard/well-known method?"**
A: Grad-CAM and its variants are correlational — they infer importance from how gradients
flow, which has documented failure modes (they can be layer-dependent, saturate, or be
biased by architecture). Occlusion is a direct intervention: we physically change the input
and measure the actual output change. That's a causal experiment, not an inference, which is
why it's the deciding evidence here, and why it's standard practice to prefer it when the two
disagree.

**Q: "Couldn't the occlusion result itself be an artifact — e.g., of the neutral-fill
value you chose?"**
A: It's a fair challenge, and we controlled for it directly: all three conditions (tumour,
padding, random control) use the *same* neutral-fill convention (the image's own median
intensity), so any systematic effect of the fill choice applies equally to all three and
cancels out in the comparison. What's being compared is region *identity* (tumour vs.
padding vs. random tissue), not fill method.

**Q: "Why not just exclude NIH (the healthy cohort) and only use MSD, to remove the
confound entirely?"**
A: Because MSD alone is 100% cancer patients — without a healthy comparison group, the
detection head has no true negative examples to learn a presence/absence distinction from
at all, and we couldn't measure recall/specificity meaningfully in the first place. NIH was
added specifically to give the model that comparison; it's what makes the confound possible
to have, but also what makes cancer-vs-healthy detection possible to train and measure at
all. Removing NIH would remove the confound risk and the ability to do the task.

**Q: "Is this a failure of the project?"**
A: The opposite — it's evidence the evaluation was rigorous rather than superficial. A
weaker project would have reported 0.993 AUC and stopped there. We instead tested our own
model's reasoning, found a real problem, fixed the part that was fixable with direct causal
verification, made a genuine additional attempt at the harder remaining problem, and are
disclosing the outcome honestly rather than only the numbers that look best. That process —
not just the accuracy number — is the actual deliverable of this branch of the project.

**Q: "How confident are you the padding fix is real and not also a coincidence?"**
A: As confident as this kind of test can make you: it's the same measurement, same 59
slices, same methodology, applied to before-and-after checkpoints, and it moved from clearly
significant (p=0.044, 2.03× control) to statistically indistinguishable from the control
(p=0.917, 1.08×) — not just "smaller," but landing almost exactly on the no-effect baseline.
That specific pattern is what a genuine fix looks like, as opposed to a fix that merely
shrinks an effect without eliminating it.

**Q: "Why does none of this get resolved in the raw accuracy numbers — wouldn't a
confounded model just have lower accuracy?"**
A: No, and that's the core danger this whole investigation exists to address: if
"scanner/institution identity" is a *reliable* cue in a training set where it happens to
perfectly track the true label, a model exploiting it will still score extremely well on
data drawn from the same two sources — the shortcut and the real signal are indistinguishable
by accuracy alone. That's precisely why standard metrics can't catch this and a dedicated
explainability + causal-testing pass was necessary.

---

## 11. One-page summary table (for a slide)

| Round | What we did | Method type | Key result |
|---|---|---|---|
| 1 | Grad-CAM on correctly-classified cancer slices | Gradient-based attribution (correlational) | Enrichment 0.003× — attention in image corners, not on tumour |
| 2 | Cross-check with Integrated Gradients | Gradient-based attribution, input-space | Disagreed with Grad-CAM (~1.08×) — inconclusive alone |
| — | Investigated preprocessing pipeline directly | Direct measurement of cached data | Confirmed real, dataset-correlated synthetic padding exists |
| 3 | 6-method attribution sweep (4 gradient CAM variants + gradient-free EigenCAM + Integrated Gradients) | Correlational, multiple designs | Gradient-through-late-layer methods agree (~0.003–0.006×); removing gradients/layer improves the picture |
| **4** | **Occlusion-based input-ablation study** (tumour / padding / random-control blanking, logit-space) | **Causal intervention** | **Tumour matters 0.47× a random patch (p=0.0075); padding matters 2.03× a random patch (p=0.044) — reverses the Round 3 read** |
| 5 | Fix: random-resized-crop augmentation, retrain, re-run the identical occlusion test | Remediation + causal re-verification | **Padding shortcut fixed** (2.03×→1.08×, p=0.917). Tumour issue improved numerically but still significant |
| 6 | Fix attempt: mask-preserving random erase, retrain, re-run the identical occlusion test | Remediation + causal re-verification | **Did not fix tumour issue** (0.58× vs. 0.62×, unchanged); padding metric ambiguous (not significant but elevated). **Reverted.** |

**Bottom line for the slide:** one real problem found and fixed with causal verification;
one real problem found, honestly attempted twice, not yet solved, and disclosed — not hidden
— in every artifact downstream of this model.

---

*Compiled 2026-08-12, companion to `docs/Presentation_Full_Documentation_and_QA.md`. Full
technical detail and every raw number behind this document lives in
`docs/Imaging_Confound_Check_documentation.md` and `docs/Imaging_Session_Summary_2026-07-18.md`.*
