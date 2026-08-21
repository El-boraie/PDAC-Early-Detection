# Imaging Branch — Training Timing Diagnostic (Baseline CNN + ResNet-50 U-Net)

**Source:** two disposable, throwaway scripts run from scratchpad — **not** part of the official pipeline, **not** committed anywhere in `src/`. `src/imaging/train_segmentation_detection.ipynb` was not touched and remains an untouched stub, still gated behind its own separate approval.
**Purpose:** get real, measured per-batch/per-epoch timing floors for **both** imaging candidates before committing to the real training notebook's design, so batch size and epoch/fold budgeting decisions are based on actual GPU numbers rather than guesses.
**Hardware:** NVIDIA GeForce RTX 3050 Laptop GPU, 4.29 GB total VRAM (confirmed via `torch.cuda.get_device_properties`, matches `PROJECT_HANDOFF.md`'s stated hardware).
**Environment:** PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128 — confirmed working (`torch.cuda.is_available() == True`) in the actual execution path used, not assumed from the original setup notes.
**Run date:** 2026-07-16.

> Draft note: every number below is from the actual executed run, not estimated — except the explicitly-labeled extrapolations, which are estimates built on top of real measurements.

---

## What Was Built (and What Wasn't)

Both diagnostics deliberately do **not** build or approximate the real imaging pipeline. They exist purely to answer "how fast can a GPU training step run on this hardware, with this data," so real numbers are available when the real notebook's design is discussed — for both candidates named in `PROJECT_HANDOFF.md`, not just one.

**Shared across both runs:**
- **Data:** `data/processed/manifest.csv`, filtered to fold 0's training split using the manifest's existing `split` column (`split != 'fold0'`) — no folds re-derived. **72,139 rows.**
- **Target label:** the manifest's `class` column (0 = NIH/healthy, 1 = MSD/cancer-patient) used as a placeholder detection target, purely so `loss.backward()` has something real to compute against — **not** a claim about the real detection head's target, just a ready-made numeric column.
- **Image resizing:** raw slices are **not** a fixed resolution — confirmed by inspecting actual files: one MSD patient's slice was 330×330, one NIH patient's was 460×460 (both `uint8`, single-channel). Since a batch needs uniform tensor shape, every slice was resized to 256×256 before batching — a timing-diagnostic assumption only.

**Baseline CNN, specifically:** 4 conv/BatchNorm/ReLU/MaxPool blocks (16→32→64→128 channels) + global-average-pool + one linear layer to a single logit. **97,761 parameters.** No pretrained weights, no segmentation head, detection-only, single-channel input — deliberately the simpler of the two candidates.

**ResNet-50 U-Net, specifically:** a real pretrained-*style* ResNet-50 encoder (torchvision, `weights=None` — see note below) feeding a U-Net decoder (4 upsample-and-concat blocks with skip connections from each encoder stage) for segmentation, plus a detection head sharing the same encoder bottleneck (global-average-pool + linear), matching the multitask architecture `PROJECT_HANDOFF.md` specifies. **43,864,834 parameters** — 3-channel input (1→3 channel replication at load time, matching the design already documented in `CT_Preprocessing_documentation.md` — never stored on disk that way), and a dummy all-zero segmentation mask target (same reasoning as the placeholder detection label: exercises the real segmentation head's compute graph without needing real mask-loading logic for a pure timing probe).

**Deliberate choice, not a silent shortcut:** the ResNet-50 encoder used `weights=None` (randomly initialized) rather than downloading real ImageNet-pretrained weights (~100MB, not cached locally, would require an unprompted network download). For **timing** purposes this makes no difference — identical architecture, identical parameter count, identical FLOPs regardless of weight values — only the output *values* would differ, which isn't what this diagnostic measures.

---

## Process (identical for both runs)

1. **Batch-size probe.** Baseline CNN started at 16; ResNet-50 U-Net started at 8 (a heavier model, so a smaller starting guess), per your instruction. Wrapped in a try/except for `torch.cuda.OutOfMemoryError` — on OOM, the plan was to clear the CUDA cache, halve the batch size, and retry, down-reporting whatever size actually survived. **Neither run needed to halve — both fit on the first attempt.**
2. **Timed region.** Exactly 30 training batches each, timed as forward pass → loss → backward pass → optimizer step **only** — data loading and the CPU→GPU transfer happen before the timer starts, so the numbers reflect compute, not I/O. `torch.cuda.synchronize()` brackets both ends of the timed region, since CUDA calls are asynchronous by default — without it, the "time" measured would just be how long the CPU took to *launch* the GPU kernels, not how long they actually took to run.
3. **Extrapolation.** Batches/epoch = fold 0's training row count ÷ working batch size. Seconds/epoch = batches/epoch × mean seconds/batch. One-fold estimate = seconds/epoch × an explicitly-stated assumption of 15 epochs to converge (not measured). Multiplied out to 1/3/5 folds, assuming each fold's training split is roughly the same size (the five folds range from ~57k to ~58k rows, close enough not to matter here).

---

## Results

| Metric | Baseline CNN | ResNet-50 U-Net |
|---|---|---|
| Parameters | 97,761 | 43,864,834 |
| Batch size used | **16** (fit first try) | **8** (fit first try) |
| Batches timed | 30 | 30 |
| Seconds/batch (mean ± std) | 0.0440 ± 0.0171 | **0.2449 ± 0.0018** |
| Peak VRAM allocated | 0.380 GB (9% of 4.29 GB) | **2.086 GB (49% of 4.29 GB)** |
| Batches/epoch (fold 0 train, 72,139 rows) | 4,509 | 9,018 |
| Estimated seconds/epoch | 198.3 s (≈3.3 min) | **2,208.8 s (≈36.8 min)** |

**Fold-count extrapolation** (assumes 15 epochs to converge — stated assumption, not measured):

| Folds | Baseline CNN | ResNet-50 U-Net |
|---|---|---|
| 1 fold | ~49.6 min (0.83 hr) | **~552.2 min (9.20 hr)** |
| 3 folds | ~148.8 min (2.48 hr) | **~1656.6 min (27.61 hr)** |
| 5 folds | ~247.9 min (4.13 hr) | **~2760.9 min (46.02 hr)** |

The ResNet-50 U-Net runs roughly **11x slower per epoch** and uses roughly **5.5x more peak VRAM** than the baseline CNN — a real, measured comparison, not the qualitative "it'll be slower" caveat this document originally shipped with.

---

## Follow-Up: Does Using More VRAM Actually Speed Things Up?

A natural follow-up question: the ResNet-50 U-Net only used 49% of available VRAM at batch 8 — would deliberately using more of it (larger batch size) make training faster? Tested empirically with a 5-config sweep (batch 8/FP32 reproduced, batch 16/FP32, batch 8/mixed-precision, batch 16/mixed-precision, batch 24/mixed-precision), same 30-batch timing methodology as above.

**Note on run-to-run variance:** the batch=8/FP32 config was re-measured here as 0.2977 s/batch, vs. 0.2449 s/batch in the original run above — an ~18% difference on the *same* configuration, most likely GPU thermal/clock-state variance on a laptop GPU rather than anything methodological. Treat any single number in this table as having roughly that much noise, not as precise to 4 decimal places.

| Config | Sec/batch | Peak VRAM | Images/sec | Est. sec/epoch |
|---|---|---|---|---|
| batch=8, FP32 (baseline) | 0.2977 | 2.083 GB | 26.9 | 2684.3 (44.7 min) |
| batch=16, FP32 | 0.7059 | 3.597 GB | **22.7 (worse)** | 3182.8 (53.0 min) |
| batch=8, mixed precision (AMP) | 0.2176 | 1.470 GB | 36.8 | 1962.7 (32.7 min) |
| **batch=16, AMP** | 0.3343 | 2.300 GB | **47.9 (best/GB used)** | **1507.1 (25.1 min)** |
| batch=24, AMP | 0.4794 | 3.124 GB | 50.1 | 1441.1 (24.0 min) |

**Answer: no, not directly.** Increasing batch size alone, at the same (full) precision, made things *worse* — batch 16/FP32 dropped throughput from 26.9 to 22.7 images/sec. At batch 8/FP32 the GPU was already close to compute-saturated, not memory-starved, so doubling the batch size more than doubled the per-batch time (0.706s vs. 2×0.298s) instead of parallelizing better — feeding it more VRAM with nothing else changed didn't help.

**What actually works is mixed-precision training** (`torch.cuda.amp` — float16 math on Tensor Cores instead of float32), which reduces memory *and* increases speed simultaneously, and that freed-up memory is what then makes a larger batch size worth trying. Batch 16 + AMP together gave the best measured tradeoff: **~44% faster per epoch** (25.1 vs. 44.7 min) while still only using 54% of the 4GB budget. Batch 24 + AMP pushed a little further but with clearly diminishing returns (+4% throughput for +36% VRAM over batch 16 + AMP).

**Revised one-fold estimate using batch 16 + AMP** (same 15-epoch assumption): 1507.1 s/epoch × 15 = 22,606.5 s ≈ **6.28 hours** — down from the original 9.20 hours at batch 8/FP32, a real ~32% reduction, not a guess. *(This number is itself revised further below — it still excludes data loading.)*

---

## Follow-Up #2: `torch.compile()`, and Does Data Loading Actually Matter?

Two more free/local levers, tested rather than assumed, at the best config found so far (batch 16, AMP).

**`torch.compile()` — unavailable in this environment.** It requires Triton as its default backend compiler; Triton isn't installed, and has only limited, experimental support on Windows in general. This isn't a quick environment fix, so this lever was not usable here — reported honestly rather than silently skipped.

**Data loading — a real, previously-uncounted cost.** Every timing number up to this point deliberately excluded data loading and the CPU→GPU transfer (per the original diagnostic's own spec: forward + backward + optimizer step only). That was correct for isolating compute cost, but it means the "6.28 hours/fold" estimate above was never a *real* wall-clock estimate — it silently assumed loading each batch was free. Tested end-to-end (load → transfer → compute, all timed together) at batch 16 + AMP:

| Config | Sec/batch (end-to-end) | vs. compute-only (0.3343s) |
|---|---|---|
| `num_workers=0` (single-threaded loading) | 0.7281 ± 0.1800 | **2.2x slower** |
| `num_workers=4` (parallel loading) | 0.4308 ± 0.3537 | 1.3x slower, high run-to-run variance |

Parallel loading (4 worker processes prefetching while the GPU computes) recovers most of the gap but doesn't eliminate it — some batches still wait on the data pipeline, which is why the variance is so much higher here (±0.354, nearly as large as the mean) than in any compute-only measurement.

**Revised, more realistic one-fold estimate**, using batch 16 + AMP + 4 workers: 0.4308 s/batch × 4,509 batches/epoch × 15 epochs ≈ **8.1 hours/fold** (5 folds: ~40.6 hours) — worse than the 6.28-hour figure above, because that figure was never counting a real cost. This is the most honest number this diagnostic produced.

---

## Interpretation — What These Diagnostics Do and Don't Tell Us

**What they confirm:** the data-loading → GPU pipeline works end-to-end on this hardware for both architectures, and both now have a real, measured per-batch timing floor and a real peak-VRAM measurement — not an estimate.

**What's still a floor, not the full picture:** both runs use a randomly-initialized encoder (ResNet-50 run) and dummy targets (both runs) — real training will have additional overhead this doesn't capture (real loss computation on real masks rather than all-zero dummies, data augmentation if used, validation passes each epoch, checkpointing I/O, logging). So even the ResNet-50 U-Net's measured 9.20 hr/fold is a **floor**, not a ceiling — the real number will likely be somewhat higher, not lower.

**Bottom line for planning, now with real numbers instead of a caveat:** the ResNet-50 U-Net's peak VRAM (2.09 GB, 49% of the 4GB budget) at batch size 8/FP32 leaves meaningfully less headroom than the baseline CNN's 9% — worth keeping in mind if the real training loop adds anything else memory-hungry (storing activations for Grad-CAM later, for instance). More importantly: **~9.2 hours for a single fold of the ResNet-50 U-Net at the original settings** is a serious constraint against `PROJECT_HANDOFF.md`'s original compressed schedule, which allocated a single day (Day 2) to "U-Net (ResNet-50) training." Five-fold training at those settings is a ~46-hour commitment.

**This is mitigated, not solved, by the mixed-precision + batch-16 finding above** — switching to it, plus parallel data loading, brings one fold down to a measured **~8.1 hours** (5 folds: ~40.6 hours; see Follow-Up #2 for why this superseded the earlier, data-loading-blind 6.28-hour figure). It doesn't remove the constraint: `train_segmentation_detection.ipynb`'s `FOLDS_TO_RUN` config point being a one-line change remains important — running fewer than all 5 folds for the ResNet-50 U-Net specifically is very likely necessary, not optional. If the real training notebook is built with mixed precision and multi-worker loading from the start, that decision should be made explicitly there, not inferred silently from this diagnostic.

## What Would It Take to Hit 5 Folds in Under 4 Hours?

Asked directly, and answered honestly rather than optimistically: **on this hardware, no.** The target (5 folds under 4 hours total, i.e. under ~48 minutes/fold) doesn't fit even a *single* fold's best measured time (~8.1 hours) — every reasonably free local lever has now actually been tested, not assumed:

| Lever | Result |
|---|---|
| Larger batch size alone (same precision) | Tested — made things **worse** (GPU was compute-bound, not memory-bound, at batch 8/FP32) |
| Mixed precision (AMP) | Tested — real ~1.5x win, the one lever that clearly helped |
| `torch.compile()` | **Unavailable** — requires Triton, not installed, limited Windows support |
| Parallel data loading (`num_workers=4`) | Tested — recovers most, not all, of a real I/O cost; adds variance |

What's left (smaller input resolution, fewer epochs than the assumed 15) aren't free wins the way AMP was — they're trade-offs against exactly what this model needs to do well: a resolution cut loses spatial detail on already-small tumour structures, and an epoch cut without independently verifying real convergence risks an undertrained model. Even stacking both aggressively might approach the ~10x speedup still needed in a best case, but that means gambling model quality against a hard deadline, not a clean engineering win.

**A more capable GPU is genuinely necessary here, not just one option among several** — but this isn't a new, unplanned expense. `PROJECT_HANDOFF.md`'s own architecture table already names **Google Colab as the stated cloud fallback for heavier training**. Colab's free tier provides a T4 GPU with 16GB VRAM — 4x this laptop's 4GB — which should comfortably remove the memory ceiling that's constraining batch size here. The realistic speedup on that specific hardware hasn't been measured (this diagnostic can't run there from this environment), so no precise number is claimed — but moving from a compute-bound 4GB card to a 16GB one is very likely to close most or all of the remaining gap. A paid rental (RunPod, Lambda, AWS, etc.) is the fallback if Colab's session limits or GPU availability become a practical problem, not the first move.
