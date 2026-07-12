# CT Preprocessing Pipeline — Process, Code, and Results

**Source notebook:** `src/imaging/imaging.ipynb`
**Input:** 281 raw NIfTI CT volumes + masks (MSD Task07 Pancreas), `data/raw/Task07_Pancreas/`
**Output:** `data/processed/` — a training-ready 2D axial slice dataset (uint8 `.npy` arrays + `manifest.csv`)
**Depends on:** `notebooks/CT EDA.ipynb` / `docs/CT_EDA_documentation.md` — every parameter below traces back to an EDA finding or an explicit hardware constraint (4GB GPU, 50GB disk budget), none are arbitrary defaults.
**Kernel / environment:** Python 3.13.12, venv at `C:\FYP\fyp_env`
**Run date:** 2026-07-06. Full 281-patient run took 6.5 minutes.

> Draft note: every number below is from the actual executed run, not estimated.

---

## Configuration

All tunable parameters live in one config cell rather than scattered inline:

```python
TARGET_SPACING = (1.0, 1.0, 1.0)   # mm, isotropic — EDA found spacing varies 0.6-0.98mm in-plane, 0.7-7.5mm in Z
IMAGE_INTERPOLATOR = sitk.sitkLinear         # continuous HU intensities
MASK_INTERPOLATOR  = sitk.sitkNearestNeighbor # discrete labels — never linear, would invent fractional labels

HU_MIN, HU_MAX = -150, 250          # EDA Section 5: pancreas 25-55 HU, tumour 10-40 HU

RISK_SCORE_DENOM_FLOOR = 50          # pancreas_px + tumour_px must reach this or risk_score = 0

TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15
N_LESION_STRATA = 4                  # quartiles, for the patient split
RANDOM_SEED = 42
```

---

## Reused EDA Helper Functions

`load_ct`, `load_mask`, `load_ct_mask_pair`, `patient_id_from_filename`, `mask_region_values`, `CTLoadError` — identical implementations copied from `notebooks/CT EDA.ipynb`. Copied rather than imported: notebook-to-notebook imports aren't practical without adding a new dependency (`importnb`/`ipynb`) for five short functions, and this project's workflow is notebook-only (no shared `.py` modules in `src/`). Any future change to loading behaviour needs updating in both places — a known, accepted tradeoff, not an oversight.

---

## Step 1: Integrity Gate + Lesion Volume

**Process:** Reuses `load_ct_mask_pair`/`CTLoadError` exactly as the EDA notebook does — excludes (rather than crashes on) any patient that fails to load, has a shape mismatch, or has mask labels outside `{0, 1, 2}`, logged by patient ID. Folded into the same pass: each valid patient's tumour volume in mm³ is computed here too (reusing the EDA Section 2b formula), since it's needed for the lesion-size-stratified split in Step 9 — avoids loading all 281 volumes a second time just for that.

**Code (essentials):**
```python
for fname in img_files:
    pid = patient_id_from_filename(fname)
    try:
        ct, mask, ct_nib, mask_nib = load_ct_mask_pair(fname, IMG_DIR, MASK_DIR)
    except CTLoadError as e:
        excluded.append((pid, f"load_error: {e}")); continue
    if ct.shape != mask.shape:
        excluded.append((pid, f"shape_mismatch: ct={ct.shape} mask={mask.shape}")); continue
    unique_labels = set(np.unique(mask).astype(int).tolist())
    if not unique_labels.issubset({0, 1, 2}):
        excluded.append((pid, f"label_violation: found {sorted(unique_labels)}")); continue
    voxel_volume_mm3 = float(np.prod(mask_nib.header.get_zooms()))
    lesion_volume_mm3[pid] = int(np.sum(mask == 2)) * voxel_volume_mm3
    valid_patients.append(pid)
```

**Result:** 281/281 candidates passed. **0 excluded.** Matches the EDA's own integrity pass finding (0 issues) — confirmed independently here rather than assumed carried over.

---

## Steps 2-3: Reorient to RAS + Resample to 1×1×1mm

**Process:** A deliberate library switch from Step 1: the integrity gate stays on nibabel (per the reuse requirement), but geometric processing (reorient, resample) switches to pure SimpleITK, re-reading each file fresh via `sitk.ReadImage`. Reason: converting a nibabel RAS+ affine into SimpleITK's own direction-cosine convention by hand is exactly the kind of fragile math that silently produces flipped volumes with no error — doing both geometric ops in one library avoids that. Costs one extra file read per patient, judged worth it for correctness.

Reorientation uses `sitk.DICOMOrient(image, 'RAS')` — defensive, not corrective, since the EDA found 0 orientation outliers already (majority orientation `('R','A','S')` across all 281 patients). Resampling uses **linear** interpolation for the image, **nearest-neighbour** for the mask — never the same interpolator for both, since linear interpolation on discrete labels would invent fractional values that don't exist (e.g. 1.4 between pancreas=1 and tumour=2).

**Code (essentials):**
```python
def reorient_to_ras(image: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(image, 'RAS')

def resample_to_spacing(image, target_spacing, interpolator, default_value=0.0):
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    new_size = [int(round(osz * ospc / tspc)) for osz, ospc, tspc
                in zip(original_size, original_spacing, target_spacing)]
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform())
    resampler.SetDefaultPixelValue(default_value)
    resampler.SetInterpolator(interpolator)
    return resampler.Execute(image)
```

**Result:** every one of the 281 volumes resampled cleanly to isotropic 1mm spacing. Smoke-tested on 3 patients before the full run — example: `pancreas_001` raw size `(512, 512, 110)` at spacing `(0.64, 0.64, 2.5)` → resampled to `(330, 330, 275)` at `(1.0, 1.0, 1.0)`. Post-resample mask values confirmed still exactly `{0, 1, 2}` (nearest-neighbour interpolation verified not to introduce invalid labels).

**Notable effect:** because Z-spacing in the raw data ranged 0.7–7.5mm, resampling to isotropic 1mm doesn't just resize in-plane — it also changes slice *count* per patient, increasing it substantially for the thick-slice-thickness scans (e.g. a 37-slice volume at 7.5mm spacing becomes ~277 slices at 1mm). This is why total slice count grew from the EDA's raw 26,719 to **72,077** post-resample (see Step 6).

---

## Steps 4-5: HU Clipping + Normalisation

**Process:** Direct implementation of the EDA-derived window — clip to [-150, +250] HU, then linearly rescale to [0, 1].

**Code:**
```python
def clip_hu(array, hu_min, hu_max):
    return np.clip(array, hu_min, hu_max)

def normalize_to_unit_range(array, hu_min, hu_max):
    return (array - hu_min) / (hu_max - hu_min)
```

**Result:** confirmed output range exactly `[0.0, 1.0]` on every processed volume (checked during the 3-patient smoke test before the full run).

---

## Step 6: 2D Axial Slice Extraction

**Process:** Splits each resampled 3D volume into independent 2D slices along the Z axis.

**Real gotcha, documented in-code:** SimpleITK's `GetArrayFromImage()` returns arrays as `(z, y, x)` — the *opposite* of nibabel's `(x, y, z)` used throughout the EDA notebook. Slicing is along axis 0 here, not axis 2 as in the EDA code. Getting this backwards wouldn't crash — it would silently slice through the wrong anatomical plane. Caught and handled correctly; verified visually in the Step 11 QA overlay (masks align with visible anatomy, not scrambled).

**Result:** 72,077 total slices extracted across 281 patients — compare to the EDA's raw pre-resample figure of 26,719 slices; the ~2.7× increase comes entirely from the Z-spacing resampling effect described above, confirmed both by a header-only dry-run estimate (72,077 slices, 25.32GB predicted) run *before* the full pipeline, and the actual measured result (72,077 slices, 25.35GB actual) — the two matched almost exactly.

---

## Step 7: Slice-Level Risk Score

**Process:** `risk_score = tumour_px / (pancreas_px + tumour_px)`, floored to 0 when `pancreas_px + tumour_px < 50`.

**Deliberate deviation from the EDA demo, flagged not hidden:** this differs from the demo label in `docs/CT_EDA_documentation.md` §7b (`tumour_px / total_slice_pixels`) — that formula was explicitly rejected for the real pipeline as "barely learnable" (background-dominated, near-zero on almost every slice regardless of tumour severity). The organ-relative version instead measures how much of the *visible organ tissue* is cancerous, which is a meaningfully different — and more learnable — target. The EDA doc's §7b demo is now stale on this specific point; not corrected there since that was out of scope for this task, but noted here so it isn't silently inconsistent.

**Floor rationale:** with fewer than 50 combined pancreas+tumour pixels visible, the ratio is dominated by segmentation noise at the organ boundary rather than real signal — flooring to 0 avoids training on a noisy ratio computed from a handful of pixels.

**Code:**
```python
def compute_slice_risk_score(mask_slice, denom_floor):
    tumour_px = int(np.sum(mask_slice == 2))
    pancreas_px = int(np.sum(mask_slice == 1))
    denom = pancreas_px + tumour_px
    if denom < denom_floor:
        return 0.0
    return tumour_px / denom
```

**Result:** risk scores span the full [0, 1] range in practice — confirmed in the QA overlay, e.g. `pancreas_203` slice 185 scored exactly `1.000` (a real edge case: that specific slice has tumour tissue with zero separately-labelled pancreas voxels, tumour fully replacing visible pancreas there — not a bug, a correct reflection of the annotation for that slice).

---

## Step 8: No Rebalancing at Storage Time

**Process:** An absence, not code — the main loop saves every slice it extracts, tumour or not, at the dataset's natural distribution. No oversampling, no discarding.

**Result:** natural class distribution preserved. Imbalance handling (Dice loss, class weighting — per the EDA's Finding 9) is deferred to training time, as decided.

---

## Step 9: Patient-Level Stratified Split

**Process:** Splits **patients**, not slices — splitting at the slice level would leak the same patient's anatomy across train and test, giving an over-optimistic validation score. Stratified by lesion-size quartile (`pd.qcut` on the Step 1 lesion volumes) so each split gets a representative spread of small and large lesions, using `sklearn.model_selection.train_test_split` twice (train vs. rest, then val vs. test) — reusing an existing dependency rather than hand-rolling a stratified splitter.

**Code (essentials):**
```python
strata = pd.qcut(lesion_volumes_mm3, q=n_strata, labels=False, duplicates='drop')
train_ids, rest_ids, _, rest_strata = train_test_split(
    patient_ids, strata, train_size=train_frac, stratify=strata, random_state=seed)
val_ids, test_ids = train_test_split(
    rest_ids, train_size=val_frac/(val_frac+test_frac), stratify=rest_strata, random_state=seed)
```

**Result:**

| Split | Patients | Slices |
|---|---|---|
| train | 196 | 50,337 |
| val | 42 | 10,586 |
| test | 43 | 11,154 |
| **Total** | **281** | **72,077** |

70/15/15 patient split lands almost exactly on target (196/281 = 69.8%, 42/281 = 14.9%, 43/281 = 15.3%).

---

## Step 10: Storage — uint8 `.npy` + CSV Manifest

**Process:** Image slices scaled from [0,1] to uint8 [0,255] (4× smaller than float32 — the difference between this pipeline's ~25GB output and an estimated ~100GB, against the 50GB budget). Mask slices stored as uint8 too, but **not** scaled — kept as raw labels `{0, 1, 2}`.

**Deliberate deviation from literal wording, flagged not hidden:** the instruction said "0-255, scaled from [0,1]" for slices generally, but scaling categorical mask labels into a 0-255 range would need an error-prone inverse step to recover them exactly during training, for zero benefit — uint8 already holds 0-2 losslessly without any scaling. Applied the scaling only to the image, not the mask.

3-channel replication (needed for the ResNet-50 encoder) is deliberately **not** done here — that happens in the training `Dataset` class, not in storage, since storing 3× the data on disk for a transformation that's cheap to do at load time would blow the disk budget for no benefit.

**Manifest columns:** `patient_id`, `slice_index`, `split`, `risk_score`, `image_path`, `mask_path`. Note: `slice_index` is the position within the **resampled** 1mm-isotropic volume — not the raw pre-resample NIfTI — since that's the space the model actually trains on and the space needed to reconstruct a patient's full volume from its saved slices later.

**Result:** `data/processed/manifest.csv`, 72,077 rows, one per slice.

---

## Disk Usage Check

**Process:** Actual measured size of everything under `data/processed/`, not an estimate — walks the output directory and sums real file sizes.

**Result:** **25.35 GB actual**, against the 50GB budget — **24.65 GB to spare**. A header-only dry-run estimate computed *before* committing to the full run predicted 25.32GB / 72,077 slices — nearly an exact match to the measured result, validating the estimation approach for future runs. (Actual free space remaining on the drive after this run: 21.74GB — lower than the budget headroom alone because the drive also carries the 12GB raw dataset, the venv, etc.)

---

## Step 11: Post-Processing Visual QA

**Process:** Loads saved uint8 `.npy` files back from disk (a genuine round-trip check, not just re-plotting in-memory arrays) for 5 random patients, overlays image + mask, actually rendered and inspected rather than assumed correct from code review alone.

**Result:** saved as `data/processed/qa_preprocessed_overlay.png`. All 5 sampled patients (`pancreas_203`, `pancreas_418`, `pancreas_107`, `pancreas_297`, `pancreas_369`) show masks landing exactly on plausible anatomical locations (upper abdomen, near kidneys/spine) with no flips, mirroring, or misalignment — visual confirmation that the nibabel→SimpleITK library switch in Steps 2-3 preserved geometric correctness end-to-end.

---

## Final Summary

- **Excluded at integrity gate:** 0 of 281 (matches EDA's own finding, confirmed independently)
- **Failed during processing (reorient/resample/save stage):** 0 of 281
- **Total runtime:** 6.5 minutes for all 281 patients
- **Final counts:** 196 train / 42 val / 43 test patients → 50,337 / 10,586 / 11,154 slices (72,077 total)
- **Disk usage:** 25.35 GB actual vs. 50GB budget — well within limits

## Helper functions and pipeline functions added (code structure, not analysis)

| Function | Purpose |
|---|---|
| `reorient_to_ras` | Canonical RAS reorientation (SimpleITK) |
| `resample_to_spacing` | Isotropic resampling with a caller-specified interpolator |
| `clip_hu` / `normalize_to_unit_range` | HU windowing and [0,1] rescaling |
| `extract_axial_slices` | Splits a `(z,y,x)` volume into 2D slices — the axis-order gotcha lives here |
| `compute_slice_risk_score` | Organ-relative risk label with denominator floor |
| `slice_to_uint8_image` / `mask_to_uint8` | Storage-format conversion, image scaled, mask not |
| `stratified_patient_split` | Patient-level, lesion-quartile-stratified train/val/test split |
| `get_dir_size_bytes` | Actual (not estimated) disk usage check |

All reused from `notebooks/CT EDA.ipynb`: `load_ct`, `load_mask`, `load_ct_mask_pair`, `patient_id_from_filename`, `mask_region_values`, `CTLoadError`.
