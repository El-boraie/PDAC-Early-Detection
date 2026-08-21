# CT EDA — Process, Code, and Results

**Source notebook:** `notebooks/CT EDA.ipynb`
**Dataset:** Medical Segmentation Decathlon Task07 — Pancreas (281 training CT volumes + matched segmentation masks, 139 unlabelled test volumes not used here) **+** NIH Pancreas-CT (TCIA CT-82, 80 healthy subjects) as the healthy-negative addendum
**Labels:** 0 = background, 1 = pancreas, 2 = cancer (PDAC tumour) — MSD only; NIH has no tumour label (healthy cohort)
**Kernel / environment:** Python 3.13.12, venv at `C:\FYP\fyp_env`

> Draft note: numbers below are pulled directly from the executed notebook's printed output — nothing here is estimated or assumed. Headings below match the notebook's own Section A–L (H2) structure so the copy-paste mapping stays obvious.

---

## Section A — Setup & Overview

Imports, plot style, path constants, and the shared helper functions (`load_ct`, `load_mask`, `load_ct_mask_pair`, `patient_id_from_filename`, `mask_region_values`, `CTLoadError`) — see the closing "Helper functions" note at the end of this document for what each one does and why they exist.

---

## Section B — Dataset Structure

**Process:** Every `.nii.gz` file in `imagesTr/` and `labelsTr/` is listed, filtered to drop macOS resource-fork junk files (`._*`, left over from a zip created on a Mac), then intersected so only patients with *both* an image and a mask are kept.

**Code (essentials):**
```python
img_files  = sorted([f for f in os.listdir(IMG_DIR) if f.endswith('.nii.gz') and not f.startswith('._')])
mask_files = sorted([f for f in os.listdir(MASK_DIR) if f.endswith('.nii.gz') and not f.startswith('._')])
valid_files = sorted([f for f in img_files if f in set(mask_files)])
```

**Result:** 281 raw image files, 281 raw mask files, 281 valid matched pairs, 0 excluded (no junk files present at run time — they were already stripped during the earlier repo-scaffolding pass, this notebook's own filter is a safety net for whoever re-runs it against a fresh unzip).

---

## Section C — Data Integrity Checks

**Why added:** the original notebook trusted every file to load correctly and every mask to contain only the expected labels, without ever checking. One pass over all 281 patients now verifies three things at once (checking them together avoids loading each 3D volume three separate times):

1. **Load errors** — every `nib.load()` call is wrapped so a corrupted/missing file is recorded by patient ID instead of crashing the loop.
2. **Shape & orientation consistency** — `ct.shape == mask.shape` for every pair, and `nib.aff2axcodes()` compared across all 281 patients (flags anyone whose scan orientation disagrees with the dataset majority).
3. **Mask label validation** — `np.unique(mask)` confirmed to be a subset of `{0, 1, 2}` for every patient.

**Code (essentials):**
```python
for fname in img_files:
    try:
        ct, mask, ct_nib, mask_nib = load_ct_mask_pair(fname, IMG_DIR, MASK_DIR)
    except CTLoadError as e:
        failed_loads.append((pid, str(e))); continue
    if ct.shape != mask.shape:
        shape_mismatches.append((pid, ct.shape, mask.shape))
    axcodes_by_patient.append((pid, nib.aff2axcodes(ct_nib.affine)))
    unique_labels = set(np.unique(mask).astype(int).tolist())
    if not unique_labels.issubset({0, 1, 2}):
        label_violations.append((pid, sorted(unique_labels)))
```

**Result:** 0 failed loads, 0 shape mismatches, 0 orientation outliers (majority/consistent orientation across all 281 patients: `('R', 'A', 'S')`), 0 mask label violations. Every one of the 281 patients passed all four checks — no exclusions needed before preprocessing.

**C2. Cross-check against `dataset.json`:** the dataset ships its own metadata file declaring patient count and label meanings — worth confirming the notebook's independently-computed numbers actually agree with it, rather than assuming. **Result:** `dataset.json` declares `numTraining = 281`, notebook found 281 valid pairs — match confirmed. Declared labels `{0: background, 1: pancreas, 2: cancer}` vs labels actually observed in the masks `[0, 1, 2]` — match confirmed.

---

## Section D — Class Labels and Balance

**Process:** For every patient, checks whether the mask contains any tumour voxels (`==2`) and/or pancreas voxels (`==1`) — patient-level balance. Separately, every 2D slice across all volumes is classified as tumour / pancreas-only / background-only — slice-level balance (this is the class distribution the segmentation loss actually has to deal with).

**Result:**
- Patient-level: 281/281 patients (100%) have tumour present, 0 do not — every training case in this MSD task is a cancer case, there's no purely-benign class in the training split.
- Slice-level (26,719 total slices across all 281 volumes): 2,537 tumour slices (9.5%), 6,255 pancreas-only slices (23.4%), 17,927 background-only slices (67.1%).
- Imbalance ratio: roughly 1 tumour slice for every 10 total slices.

**Interpretation:** the imbalance ratio is why the preprocessing plan (Section K) calls for Dice loss and class weighting rather than plain cross-entropy. The 100%-tumour patient-level split also means patient-level "risk" classification isn't meaningful for MSD alone — the slice-level risk score (Section I) is the more actionable label, since tumour presence/extent varies slice-to-slice within every patient even though every patient has tumour somewhere. This is also the reason Section L (NIH) exists: MSD alone gives the model no true healthy-negative examples at all.

**D2. Lesion Size Distribution:** tumour *presence* was already tracked, but not tumour *size* — relevant because very small lesions can disappear under aggressive downsampling. Reuses the mask already loaded during the class-balance pass (no second read) — tumour voxel count × voxel volume gives physical lesion volume in mm³ per patient. **Result:** smallest lesion — `pancreas_347` at 412.9 mm³; largest lesion — `pancreas_415` at 732,388.1 mm³ (roughly 1,770× larger than the smallest). Histogram saved as `eda_lesion_size_distribution.png`. This ~1,770× spread is the concrete justification for keeping the 1mm isotropic resampling target rather than downsampling further in-plane.

---

## Section E — Image Properties

**Process:** For every patient, reads resolution (H×W×slices), voxel spacing (X/Y/Z in mm), and HU (Hounsfield Unit) range (min/max/mean/std) directly from the NIfTI header + array.

**Result:** all 281 volumes are 512×512 in-plane (no resizing needed there). HU range: -2048 to 4009 (global min/max across all patients), global mean -572.9 HU. Voxel spacing: X/Y 0.605–0.977mm (fairly consistent), Z (slice thickness) 0.700–7.500mm (highly variable — over 10× range between thinnest and thickest scans).

**Interpretation:** wide HU range confirms windowing is required before training; variable voxel spacing (especially Z / slice thickness) confirms resampling to isotropic spacing is required before 2D slice extraction.

---

## Section F — Distribution of Image Sizes

**Process:** Histograms of slices-per-volume, in-plane spacing (X), slice thickness (Z), height, width, and a spacing X-vs-Z scatter — all against a 1.0mm target line, visually showing how far the raw data is from the target resampling grid.

**Result:** saved as `eda_size_distributions.png` — key numeric takeaways already captured in Section E above.

---

## Section G — Pixel Intensity Histograms

**Process:** HU distributions computed from a 10-patient random sample (seed 42), split into global / pancreas-region / tumour-region, to (a) confirm the wide HU range needs windowing and (b) check whether tumour tissue is visibly hypodense (darker) relative to pancreas tissue — the basis for the [-150, +250] HU window choice.

**Result (10-patient sample):** global HU — min -1024.0, max 3071.0, mean -543.2, std 491.5, median -883.0 (the very negative median reflects how much of any abdominal CT slice is just air/background). Pancreas region — mean 89.0 HU, std 109.8. Tumour region — mean 65.9 HU, std 55.7. Difference (pancreas − tumour): 23.1 HU — tumour reads measurably darker (hypodense) than surrounding pancreas tissue, consistent with the clinical literature.

**G2. HU Windowing Validation (50-patient sample):** the window was chosen from only a 10-patient sample; re-checking against a larger, independently-sampled 50-patient set (different random seed) tests whether 10 patients were actually enough. **Result:** 0.5th percentile = -1024.0 HU, 99.5th percentile = 440.0 HU — this **flags** against the chosen [-150, +250] window (differs by well over the 50 HU tolerance on both ends). **Caveat, not glossed over:** this percentile is computed over *every* voxel including background air (HU ≈ -1024), which dominates the low end by sheer pixel count — a whole-image statistic, not a body/soft-tissue statistic. The window itself was chosen from clinical HU reference ranges (pancreas 25–55 HU, tumour 10–40 HU), not from a raw whole-image percentile, so this flag is real but doesn't necessarily mean the clinically-derived window is wrong.

---

## Section H — Mean and Standard Deviation Analysis

**Process:** Per-patient global mean/std HU, plus per-region (pancreas, tumour) mean/std, computed across all 281 patients (not just the 10-patient sample) — checks inter-patient variability to justify per-volume normalisation.

**Result (all 281 patients):** global volume mean HU -572.87 ± 67.90 (std across patients), global std HU averages 490.42. Pancreas region mean 80.63 ± 30.95 HU. Tumour region mean 76.36 ± 34.65 HU. The ±68 HU spread in *global mean* between patients is the concrete justification for per-volume normalisation rather than a single dataset-wide normalisation constant.

---

## Section I — Slice-Level Risk Score Label Definition

**Why added:** the multitask U-Net's second head (slice-level risk score) needs a numeric training target that the raw MSD dataset doesn't provide — this had to be defined, not just discovered.

**Decision:** **tumour area fraction per slice** (tumour pixel count ÷ total pixels in that slice), continuous in [0, 1], rather than a plain binary "tumour present in slice" label. A risk *score* implies a continuum, not a class — a slice that's 2% tumour and one that's 80% tumour shouldn't get an identical label. Area fraction still reduces to the binary label for free by thresholding at `>0`.

```python
def slice_risk_scores(mask: np.ndarray) -> np.ndarray:
    total_px = mask.shape[0] * mask.shape[1]
    return np.array([(mask[:, :, i] == 2).sum() / total_px for i in range(mask.shape[2])])
```

**Result:** demonstrated on 2 sample tumour-bearing patients (seed 1), plotted as a per-slice risk curve — saved as `eda_slice_risk_score_demo.png`. Full-dataset label generation is deferred to the preprocessing pipeline (and, per Section K1, actually uses a *different*, organ-relative formula — see the note at the end of this section).

> **Formula update, flagged not hidden:** the actual preprocessing pipeline (`docs/CT_Preprocessing_documentation.md`) uses `tumour_px / (pancreas_px + tumour_px)` instead of the `tumour_px / total_px` demonstrated here — the total-pixel version was later found "barely learnable" (background-dominated, near-zero on almost every slice) once the pipeline was actually built. This section's demo code is left as originally written for historical accuracy; the pipeline doc is the authoritative formula.

---

## Section J — Visual Inspection of Sample Images

**Process:** For 3 randomly selected patients (seed 42), the CT slice containing the middle tumour slice is shown four ways: raw HU, HU-windowed, mask only, and CT+mask overlay — a sanity check that the mask actually aligns with visible tumour tissue.

**Result:** saved as `eda_visual_inspection.png` — visual output only, no numeric summary.

---

## Section K — Preprocessing Plan (derived from EDA)

| Step | Tool |
|---|---|
| Load `.nii.gz` volume | nibabel |
| Resample to 1×1×1mm | SimpleITK |
| Clip HU to [-150, +250] | numpy |
| Normalise to [0, 1] | numpy |
| Extract 2D slices | numpy |
| Compute slice-level risk-score label (tumour area fraction) | numpy |
| Save slices as `.npy` arrays | numpy |
| Record patient-slice mapping | CSV |

**Full findings table** (4 new findings first, then the 5 original):

| # | Finding | EDA value | Solution |
|---|---|---|---|
| 1 | Data integrity confirmed | 0 load failures, 0 shape mismatches, 0 orientation outliers, 0 label violations across 281 patients | No cleaning required — confirmed clean |
| 2 | Lesion size varies widely | 413 – 732,388 mm³ across 281 tumour-bearing patients | Keep 1mm resampling target, avoid further in-plane downsizing |
| 3 | HU window validated on larger sample | 50-patient 0.5th/99.5th percentile: -1024 / 440 HU vs chosen window [-150, +250] | Flagged — see Section G2 caveat; window itself still clinically justified |
| 4 | Slice-level risk label defined | Tumour area fraction per slice (continuous, thresholdable to binary) | Compute for all 281 patients during slice extraction, store alongside `.npy` arrays |
| 5 | Variable slice count | Min=37, Max=751 | Extract 2D slices independently, treat each as a separate sample |
| 6 | Variable voxel spacing | Z: 0.70 – 7.50 mm | Resample all volumes to 1×1×1mm (SimpleITK) before slice extraction |
| 7 | Wide HU range | -2048 to 4009 HU | Apply HU windowing: clip to [-150, +250] |
| 8 | Values not normalised | Range after windowing: [-150, +250] | Normalise to [0, 1]: `pixel = (pixel - (-150)) / (250 - (-150))` |
| 9 | Heavy class imbalance | Tumour = 9.5% of slices | Use Dice Loss during training, apply class weights to tumour label |

---

## Section L — Dataset 2 (NIH) Addendum: Healthy Negative Cohort

**Why added:** MSD Task07 is 100% cancer patients (Section D) — the model has no true healthy-negative examples to learn a presence/absence distinction from. NIH Pancreas-CT (TCIA "Pancreas-CT", Roth et al. 2016; 80 healthy subjects) fills that gap.

**Data provenance — important, not glossed over:** this copy came via a Hugging Face mirror (`CADS-dataset/0024_pancreas_ct`) that repackages the original raw TCIA DICOM as pre-converted `.nii.gz`. Confirmed genuine (same DOI/citation as the real TCIA Pancreas-CT collection) but **not** raw DICOM as originally planned. Two concrete consequences:
1. Images are single-file `.nii.gz`, same format as MSD — the existing `load_ct`/`CTLoadError` helpers are reused unchanged, no DICOM series reader needed.
2. The publisher's own metadata shows all 80 patients at Z-spacing ≈ 1.0mm, not the 1.5–2.5mm slice thickness raw TCIA actually has — this copy was already resampled before it reached this project.

**Masks are not used in this addendum.** The segmentation files shipped with this mirror turned out to be multi-organ auto-segmentation output — each file contains ~10-25 numeric label IDs with no label-to-organ mapping available anywhere in the repo. Guessing which numeric ID means "pancreas" risked silently mislabeling a different organ across all 80 patients, so this pass is **image-only**; anything requiring a mask is explicitly deferred, not faked.

**L3. NIH Structure & Integrity (image-only).** Reuses the same integrity-gate pattern as Section C, image-only:

**Result:** 80/80 patients checked, 80/80 loaded successfully, 0 failures. Orientation consistent (`R, A, S` — same majority as MSD) across all 80, 0 outliers. Shape-vs-mask and label validation explicitly **deferred** (no usable pancreas mask this pass).

**L4. Merged Patient-Level Class Balance — the headline number:**

| Cohort | Patients | % |
|---|---|---|
| MSD (cancer) | 281 | 77.8% |
| NIH (healthy) | 80 | 22.2% |
| **Total** | **361** | |

**L5. Merged Slice-Level Balance:**

| | Total slices | Tumour | Pancreas-only | Background-only |
|---|---|---|---|---|
| MSD | 26,719 | 2,537 (9.5%) | 6,255 (23.4%) | 17,927 (67.1%) |
| NIH | 18,942 | 0 (0.0%, by definition) | undetermined | undetermined |
| **Merged** | **45,661** | **2,537 (5.6%)** | not reported | not reported |

NIH's tumour-slice count is confidently 0 — true by definition for a healthy cohort regardless of mask availability. The pancreas-only/background-only split for NIH's 18,942 slices genuinely can't be determined without a pancreas mask, so it's reported as undetermined rather than guessed into either bucket. Note the tumour-slice *percentage* dropped from MSD-alone's 9.5% to 5.6% once healthy volume is added — same absolute cancer content, diluted by additional non-cancer slices, exactly as expected.

**L6. NIH Image Properties & HU Window Check.**

**Result:** all 80 volumes 512×512 in-plane. Slices per volume: 181–466. Spacing X/Y: 0.664–0.977mm. Spacing Z: 0.500–1.000mm (pre-resampled by the publisher — see provenance note above; raw TCIA's actual slice thickness is 1.5–2.5mm).

HU distribution (20-patient sample, global — no pancreas mask to isolate organ-specific HU this pass): 0.5th percentile -2048.0 HU, 99.5th percentile 384.0 HU, vs the MSD-derived window [-150, +250]. **Flagged, not confirmed**: NIH's empirical percentiles differ from the window by more than the 50 HU tolerance — same whole-image-percentile caveat as Section G2 (background air dominates the low end), so this doesn't necessarily mean the window is wrong for NIH tissue, but the raw numbers don't confirm it either.

> **Update (preprocessing pipeline, 2026-07-14):** a tighter follow-up check was run in `docs/CT_Preprocessing_documentation.md` — same percentile check, but excluding voxels below -500 HU first (a crude body mask) to remove the background-air contamination this section flagged as a likely cause. Result: -457.0 / 612.0 HU — narrower, but **still not confirmed** against tolerance. The decision made there (and this section's final word on the question): proceed with the existing window anyway, since a body-minus-air mask still isn't organ-specific and properly resolving this needs an NIH pancreas mask that doesn't exist in this data copy. See the preprocessing doc's "NIH HU Window Validation" section for the full reasoning.

**L7. Lesion Size — Not Applicable.** NIH patients are healthy; there is no lesion to measure, independent of mask availability.

> **Update (preprocessing pipeline, 2026-07-14):** the speculative "plain patient-level random split" mentioned in an earlier version of this note is not what was built. NIH patients instead share one flat `"healthy"` stratification bucket alongside MSD's existing lesion-quartile buckets, so the merged split (`docs/CT_Preprocessing_documentation.md`, "Merged Patient Split") is still stratified — by class first, and by lesion quartile within the cancer class only — rather than falling back to an unstratified random split for NIH.

**L8. Risk-Score Validation — Deferred, Not Fabricated.** The preprocessing pipeline's risk-score formula needs a tumour mask and a pancreas mask per slice; neither exists for NIH in this data copy. This cannot be empirically validated against real mask arrays right now. **By construction**, the correct value for every NIH slice is 0 — there is no tumour label possible for a healthy patient — but this is a logical deferral, not a computed confirmation. Revisit once a genuine single-organ pancreas mask is available for NIH (e.g. from the original raw TCIA/NBIA download).

> **Update (preprocessing pipeline, 2026-07-14):** implemented exactly as deferred here — every NIH slice gets a fixed `risk_score = 0.0`, asserted (not just claimed) against all 18,616 NIH manifest rows. Still a logical deferral, not an empirical confirmation, for the same reason given above.

---

## Helper functions added during refactor (not analysis, just code structure)

Introduced once near the top of the notebook to remove duplicated `nib.load(...).get_fdata()` calls that were copy-pasted across 5 cells:

- `load_ct(fname, img_dir)` / `load_mask(fname, mask_dir)` / `load_ct_mask_pair(fname, img_dir, mask_dir)` — centralised NIfTI loading, each wrapped in try/except raising `CTLoadError(fname, original_exception)` so a bad file is reported by patient ID instead of crashing whichever loop hit it.
- `patient_id_from_filename(fname)` — strips `.nii.gz`.
- `mask_region_values(ct, mask, label)` — boolean-indexes CT voxels by mask label (0/1/2).
- `nih_patient_id_from_filename(fname)` *(Section L)* — extracts the short case number from NIH's UID-based filenames, e.g. `'00001_1.2.826...⁠_0000.nii.gz'` → `'00001'`.

This is why every subsequent section's code got shorter and more uniform without changing what any of them compute or print.
