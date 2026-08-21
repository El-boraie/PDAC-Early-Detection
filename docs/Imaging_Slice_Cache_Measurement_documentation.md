# Imaging Branch — Slice Cache Packing, Task 1 (Measurement)

**Source:** `src/imaging/pack_slice_cache.py` — one new script, part of the official pipeline. Reads only `data/processed/manifest.csv`, `data/processed/images/`, and `data/processed/masks/`; writes nothing into `data/processed/` itself.
**Purpose:** decide the fixed crop-box size a slice cache needs *from measured data*, not a guess, before building it.
**Scope boundary:** `src/imaging/train_segmentation_detection.ipynb` was not touched and remains an untouched stub, gated behind its own separate approval. No model was trained, fine-tuned, or evaluated.
**Hardware:** RTX 3050 4GB VRAM, 16GB RAM, ~21GB free disk at the time of the run.
**Run date:** 2026-07-16.

---

## The Question This Task Answers

The dataset is 361 patients / 90,693 processed 2D axial slices (MSD Task07: 281 cancer patients, 72,077 slices, with pancreas/tumour masks; NIH Pancreas-CT: 80 healthy patients, 18,616 slices, image-only, no masks at all). Every volume was resampled to 1×1×1mm isotropic spacing before slicing, so 1 pixel = 1mm for every patient — but that also means slice dimensions vary per patient (e.g. one MSD slice is 330×330, one NIH slice is 460×460).

Batching requires a uniform tensor shape. The decision made up front: **center-crop (and pad only when a slice is smaller than the box), never resize** — resizing a 330×330 and a 460×460 slice both to some fixed size would rescale their pixel spacing differently (e.g. 1.29mm vs. 1.80mm per pixel for a 256 target), destroying the isotropic spacing the preprocessing pipeline specifically paid for. Cropping preserves 1mm/pixel by construction; the only open question was **what box size**.

---

## What Was Built

`pack_slice_cache.py`, one small docstringed function per step:

- `load_manifest()` — loads `manifest.csv`, sanity-checks row counts (72,077 MSD / 18,616 NIH).
- `measure_dimensions(df)` — gets every slice's (height, width).
- `report_dimension_distribution(dims_df)` — min/median/max/value_counts, MSD vs NIH separately.
- `compute_crop_retention(dims_df, box_sizes)` — for every MSD slice with a mask, what fraction of nonzero mask pixels (pancreas=1, tumour=2) a center crop-or-pad to each candidate box size would lose.
- `report_crop_retention(retention_df, box_size)` — % of slices with zero loss, and the loss distribution for the rest.
- `compute_disk_requirements(box_size, n_images, n_masks)` / `report_disk_requirements(req)` — exact byte math against the ~21GB budget.

All file I/O is wrapped in try/except with patient-ID-level error logging (log and continue, not crash), and every scan uses a `tqdm` progress bar. Candidate box sizes (`[256, 320]`), paths, and the disk budget are configured as constants at the top of the file, not hardcoded inline.

---

## Two Runs: a Performance Detour

**First run: ~80 minutes, then diagnosed.** The initial implementation read every one of the 90,693 image files (via `mmap_mode='r'`, header-only) plus loaded all 72,077 MSD masks once per candidate box size (144,154 full reads) — roughly 235,000 individual file opens in total. Mid-run, elapsed wall-clock time (~79 min) was wildly out of proportion to actual CPU time consumed (~5 min) — a ~94% idle/wait ratio, meaning the process was blocked, not computing.

Diagnosis, in order:
1. **Ruled out OneDrive cloud placeholders** — checked file attributes (`attrib`) and reparse-point status on the actual data files; `C:\FYP` is a plain local folder, not a synced cloud placeholder.
2. **Identified Windows Defender real-time protection** (enabled, no exclusions configured) as the likely cause — each of ~235,000 individual small-file opens gets intercepted by the AV filter driver for scanning, and the fixed per-file latency this adds is consistent with the observed wall-clock/CPU ratio and rough per-file overhead math.

The first run was allowed to finish naturally (it completed with exit code 0 and valid results while the fix was being discussed) — the numbers below are confirmed identical between both runs, which cross-validates the refactor introduced no bugs.

**Refactor, then re-run: ~39 seconds**, after the user added a Windows Defender exclusion for `C:\FYP\`. Three changes, all still producing byte-identical results:
1. **Per-patient dimension reads, not per-slice.** Verified first (sampling 5 patients, multiple slices each) that every slice belonging to one patient shares one shape — expected, since they all come from the same resampled volume at the same spacing. This cut the dimension scan from 90,693 file opens to 361 (one per patient).
2. **Single-pass mask retention.** Each of the 72,077 MSD masks is now loaded once and checked against *all* candidate box sizes in that one pass, instead of being reloaded once per box size (144,154 reads → 72,077).
3. **Threaded I/O** (`ThreadPoolExecutor`, 16 workers) for both the dimension scan and the mask-retention pass — file I/O releases the GIL, so concurrent reads overlap the AV-scan/disk-wait latency instead of serializing it.

---

## Results

### Slice dimension distribution (slices are square — height == width in every row)

| Dataset | n | min | median | max |
|---|---|---|---|---|
| MSD | 72,077 | 310 | 412 | 500 |
| NIH | 18,616 | 340 | 440 | 500 |

The smallest slice anywhere in the dataset is 310×310 (MSD). No slice, in either dataset, is smaller than 310 in either dimension.

### Crop retention (MSD slices with a nonzero mask, n = 23,417)

| Box | Zero mask-pixel loss | Some loss | Loss distribution among the lossy slices |
|---|---|---|---|
| 256×256 | 23,216 (99.142%) | 201 (0.858%) | min 0.03%, median 9.35%, p90 13.85%, p99 14.55%, max 14.79% |
| 320×320 | 23,417 (**100.000%**) | 0 | — |

### Disk requirement (images: 90,693 × box², masks: 72,077 × box², MSD-only)

| Box | images.npy | masks.npy | Total | vs. ~21GB budget |
|---|---|---|---|---|
| 256×256 | 5.535 GB | 4.399 GB | 9.935 GB | Fits, ~11 GB headroom |
| 320×320 | 8.649 GB | 6.874 GB | 15.523 GB | Fits, ~5.5 GB headroom |

---

## Recommendation Given vs. Decision Made

**Recommendation was 320×320** — the only option with zero mask-pixel loss, at an incremental cost of ~5.6GB that still fit comfortably inside the stated budget. The reasoning: losing up to ~14.8% of a slice's mask pixels on 201 MSD slices (0.86%) is a silent, uneven corruption of ground truth for a segmentation task, and it isn't evenly distributed — it concentrates on the slices where the pancreas/tumour footprint is largest, which are often the more informative training examples, not noise to discard.

**Decision made: 256×256.** The user chose 256 over the recommendation. This is a deliberate, informed trade-off, not an oversight — the exact cost is on record above (0.858% of MSD mask-bearing slices lose some mask signal, median 9.35%/max 14.79% of pixels on the ones affected), against ~5.6GB of saved disk space (9.935GB vs. 15.523GB) and a slightly smaller per-slice tensor for training.

Task 2 (packing to `data/processed/cache/` at BOX=256) proceeded from this decision — see `cache_meta.json` in that directory for the exact provenance record (source manifest checksum, pad values, row counts) of the resulting cache.

---

## Task 2: Packing, Verification, and the Later Repack to BOX=320

### Initial pack, at the decision above (BOX=256)

`run_task2_pack()` in the same script: center-crop/pad every slice (pure index slicing —
no interpolation, so mask class values are never blended into invalid fractional labels),
write `images.npy` (all 90,693 rows) and `masks.npy` (72,077 MSD rows only, no placeholder
rows for NIH), `manifest_cache.csv` (original manifest + `img_row`/`mask_row`), and
`cache_meta.json` (box size, dtype, row counts, pad values, source-manifest checksum/mtime).

**Disk usage:** `images.npy` 5.535 GB + `masks.npy` 4.399 GB + `manifest_cache.csv` 10.4 MB
+ `cache_meta.json` = **9.944 GB total**, fit the ~21 GB budget with ~11 GB headroom.

**Verification (6 sampled rows, 3 MSD + 3 NIH):** every packed row checked for **exact
array equality** against a fresh crop of its original source file — not just a shape check,
which would miss an off-by-one between the image and mask arrays. All 6 matched exactly.
Row-index integrity confirmed separately: `img_row` unique and contiguous 0–90692;
`mask_row` unique and contiguous 0–72076 for MSD rows, exactly -1 for all NIH rows. Visual
plot (`outputs/qa/ct/pack_verification_box256.png`) confirmed crops centered, MSD mask
overlays aligned, no obvious clipping of anatomy.

### The repack to BOX=320 (2026-07-17)

Training moved to a rented RunPod GPU (see
`docs/Imaging_Local_Preflight_and_Migration_documentation.md` for why) — this removed the
**retrain-risk constraint** that made BOX=256 the reluctant local choice in the first place
(a slower/more expensive local GPU meant repacking and retraining at a larger box size was
costly to redo if wrong; a rented GPU made that no longer true). With that constraint gone,
the recommendation from Task 1 — BOX=320, the only option with **zero** measured
mask-pixel loss — was adopted.

`pack_slice_cache.py`'s `SELECTED_BOX_SIZE` was changed to 320, and repacked
**non-destructively** to a new directory (`data/processed/cache_box320/`), leaving the
working BOX=256 cache untouched at `data/processed/cache/` in case of a failed repack or a
need to roll back.

**Disk usage:** `images.npy` 8.649 GB + `masks.npy` 6.874 GB + `manifest_cache.csv` 10.4 MB
+ `cache_meta.json` = **15.533 GB total**, still comfortably inside budget.

**Verification:** identical rigor and sample size as the BOX=256 pack — all 6 sampled rows
matched their source crop exactly, row-index integrity held, visual plot saved
(`outputs/qa/ct/pack_verification_box320.png`).

### Packaging and promotion

The BOX=320 cache was zipped for transport to the rented pod — `zipfile.ZIP_DEFLATED`,
internal folder deliberately renamed to `cache/` (not `cache_box320/`) inside the archive,
so extracting it on the pod at `data/processed/cache/` would match what `config.py` already
expects there, with no path reconfiguration needed remotely. Real measured result:
**15.533 GB → 4.803 GB, 69.1% smaller** (masks compressed extremely well, being mostly
background zeros — 6.874GB → ~10MB; images compressed far less, since real CT intensity
data isn't very compressible — 8.649GB → ~4.79GB). Verified with `zipfile.testzip()` (no
corruption) before upload.

After the pod confirmed training against this cache, the local directories were promoted to
match what's now canonical: `data/processed/cache/` (the old BOX=256 pack) was renamed to
`data/processed/cache_box256_archive/`, and `data/processed/cache_box320/` was renamed to
`data/processed/cache/` — so `data/processed/cache/` now **is** the BOX=320 pack, matching
every checkpoint and result that depends on it. `utils/config.py` and `pack_slice_cache.py`
were both updated to reflect this (comments and `CACHE_DIR`, respectively) — `ROOT` stays
`C:\FYP` in both; only the pod's own copy of `config.py` points `ROOT` at `/workspace`.
