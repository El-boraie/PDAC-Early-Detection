# Imaging Branch — Local Training Pre-Flight & Migration to a Rented GPU

**Source:** `src/imaging/train_segmentation_detection.ipynb`, built against the packed
BOX=256 slice cache on the local RTX 3050 laptop.
**Purpose:** measure one real training epoch before committing to a multi-fold, multi-epoch
run, per the notebook's own explicit rule — if the measured cost exceeds a stated trigger,
stop and move to a rented GPU rather than absorb the overrun locally.
**Run date:** 2026-07-16/17.

**Not to be confused with** `docs/Imaging_Timing_Diagnostic_documentation.md` — that was an
earlier, separate, throwaway diagnostic using dummy targets and `weights=None`, built purely
to get a rough compute/VRAM floor before the real training notebook existed. This document
covers the **real** pre-flight, run inside the actual `train_segmentation_detection.ipynb`
training loop, using the real dataset, real losses, real pretrained weights.

---

## Pre-Flight Result: 1 Real Epoch, Fold 0, `resnet50_unet`

| Metric | Value |
|---|---|
| Rows trained (strided, offset 0 of 3) | 24,140 (1,508 batches) |
| Wall-clock | **2268.7s = 37.81 min** |
| Peak VRAM | 2.511 GB (expected ~2.30GB — within tolerance) |
| Realised class balance | {0: 12,054, 1: 12,074} — sampler held after striding |

**37.81 min/epoch exceeded the notebook's stated ~15 min/epoch stop trigger** (expected
~9–11 min/epoch, based on the earlier throwaway timing diagnostic). Per the notebook's own
rule, this printed an explicit `*** STOP ***` and did **not** proceed to a full run.
Extrapolated to 3 folds: ~17–28 hours depending on assumed epoch count — clearly
incompatible with the project deadline on this hardware.

---

## Diagnosis: Is This a Fixable Local Problem, or a Real Hardware Ceiling?

Rather than accept the number and immediately rent a GPU, four specific hypotheses were
tested directly before deciding — each is a real, falsifiable check, not just a guess.

### 1. Is it a one-time cold-start cost (cudnn autotuning, first-batch overhead)?

Ran a second, fully instrumented epoch on the same fold, same model/optimizer (warm state):

| | Epoch 0 (cold) | Epoch 1 (warm) |
|---|---|---|
| Total | 38.21 min | 36.95 min |
| Data-wait | 0.02 min (0.0%) | 0.02 min (0.0%) |
| GPU-compute | 37.87 min (99.1%) | 36.44 min (98.6%) |

**Speedup epoch 0 → 1: 1.03x — essentially none.** Ruled out: this is not a warm-up
artifact, it's sustained.

### 2. Is it data loading (disk I/O, DataLoader worker starvation)?

From the same instrumented run: data-wait was 0.02 minutes out of ~37 — **negligible**. The
memmap cache and 4 DataLoader workers were keeping up fine. Ruled out.

### 3. Is the GPU actually busy, or stalled/contended?

Sampled `nvidia-smi` utilization roughly once per second throughout both epochs (4,226
samples): **mean utilization 97.0%**, only 2.0% of samples below 50%, peak memory used
3.9GB. The GPU was genuinely, continuously busy — not idle, not waiting on something else.
Ruled out contention as the bottleneck.

### 4. Was a stray leftover process competing for the GPU during the measurement?

A legitimate concern raised mid-investigation: several throwaway smoke-test scripts had
been run earlier in the same session to validate the training pipeline before committing to
the real pre-flight. Checked directly via `Get-CimInstance Win32_Process` (full command
lines, not just process names): one process (`smoke_test.py`, from an earlier crashed
`num_workers=2` attempt) had been sitting alive for ~57 minutes despite its background task
having been reported "completed" hours earlier.

Investigated before assuming it mattered: its CPU time had not moved at all over a 5-second
sample (fully idle), and it did **not** appear in `nvidia-smi`'s GPU process list (held no
GPU memory). An idle process holding no GPU memory cannot meaningfully starve another
process of compute or VRAM. Killed for hygiene regardless — this immediately triggered the
harness's own stale completion notification for that earlier crashed task, confirming it
really was that leftover, not something new. **Ruled out** as the cause of the slowdown,
based on direct evidence, not assumption.

### Conclusion

All four checks point the same way: compute genuinely takes ~37 minutes/epoch on this GPU
for this architecture at this batch size. This is a real, structural hardware ceiling, not a
fixable local bug. One partially-explored, not-fully-verified lead: the built `ResNet50UNet`
has 71,876,484 parameters — noticeably more than the 43,864,834 the earlier throwaway timing
diagnostic measured, because this decoder mirrors the encoder's full channel width at every
stage (a heavier design than a typical lean U-Net decoder). Flagged as worth investigating
further given headroom, not acted on locally.

---

## A Real Bug Found Along the Way: `num_workers>0` Fails Under Windows + Jupyter

Before the real pre-flight could run at all, a smoke test surfaced a genuine, non-obvious
Windows/multiprocessing bug: a `Dataset` class defined directly in a notebook cell lives in
the kernel's `__main__` namespace. On Windows, `DataLoader` workers (`num_workers>0`) use
`multiprocessing.spawn`, and each worker process re-imports the Dataset class by module path
to unpickle it — a class living in `__main__` **cannot** be re-imported this way, and fails
with `AttributeError: Can't get attribute 'SliceCacheDataset' on <module '__main__'>`.
Confirmed directly both ways (failed inline, succeeded once moved) before settling on the
fix: `SliceCacheDataset` was extracted into its own importable module,
`src/imaging/slice_cache_dataset.py`, imported into the notebook rather than defined inline.
`NUM_WORKERS=4` was a spec requirement, not optional, so this had to be fixed rather than
silently dropped to `num_workers=0`.

---

## Decision: Move to a Rented GPU

Per the notebook's pre-agreed trigger, exceeding ~15 min/epoch after ruling out every
locally-fixable cause meant moving training off the RTX 3050 laptop. RunPod (RTX 6000 Ada,
48GB VRAM) was chosen — see `docs/Imaging_3Fold_Training_Results_documentation.md` for the
pod's own pre-flight (batch-size sweep, `torch.compile`) and the training results that
followed. The retrain-risk constraint this removed is also what changed the BOX=256 vs
BOX=320 decision — see `docs/Imaging_Slice_Cache_Measurement_documentation.md`'s "Task 2"
section for that repack.
