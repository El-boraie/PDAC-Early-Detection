"""SliceCacheDataset lives in its own importable module, not inline in a notebook cell.

On Windows, multiprocessing uses 'spawn', not 'fork' -- DataLoader workers (num_workers>0)
re-import the Dataset class by module path to unpickle it. A class defined directly in a
Jupyter cell lives in the kernel's __main__ namespace, which worker processes cannot
re-import, and fails with "AttributeError: Can't get attribute '...' on <module '__main__'>".
Defining it here, in a real module imported by the notebook, is what makes num_workers=4
actually work under ipykernel -- confirmed by testing both ways before settling on this.

augment=True added 2026-07-17 after imaging_confound_check.ipynb's occlusion analysis found
resnet50_unet's detection head relies significantly MORE on the BOX=320 packing's synthetic
padding than on a random patch of tissue, and significantly LESS on the real tumour region
(p=0.044, p=0.0075 respectively, paired Wilcoxon, n=59). See
docs/Imaging_Confound_Check_documentation.md, "Round 4". Default stays False -- every
existing caller (evaluation passes, the confound-check notebook itself) must keep seeing the
real, unaugmented images; only new training-time call sites opt in explicitly.

Mask-preserving random erase added 2026-07-17 (Round 5 fixed the padding shortcut, but the
same occlusion re-test found the tumour region was STILL moving the prediction significantly
LESS than a random patch of tissue post-fix, 0.62x control, p=0.0075 -- see
docs/Imaging_Confound_Check_documentation.md, "Round 5" / "If resuming, option 1"). Only ever
applied to rows with has_mask=True (MSD rows -- NIH has no ground-truth mask, so "outside the
tumour" isn't defined). Randomly blanks a patch of ordinary tissue that never overlaps the
real tumour mask, so every region except the tumour becomes an unreliable, randomly-vanishing
cue during training while the tumour stays the one signal that's always present.

**Outcome (Round 6, 2026-07-17): did not help, reverted.** Retrained and re-ran the same
occlusion test: tumour sensitivity was statistically unchanged (0.58x control, p=0.0148, vs.
0.62x/p=0.0065 before this change), and padding sensitivity moved from indistinguishable-from-
control (1.08x, p=0.917) to a numerically large but not-quite-significant 2.86x (p=0.135) --
ambiguous, but not a confirmed improvement, and not obviously safe either. Reverted:
`checkpoints/imaging/final/` and `checkpoints/imaging/candidates/` were restored to the Round 5
(padding-fix-only) state; the Round 6 checkpoints/results are archived at
`checkpoints/imaging/candidates_5fold_mask_erase_archive/` and
`results/imaging/archive/*_mask_erase_archive*`. This method (`_apply_mask_preserving_erase`)
is left in place, disabled by nothing special -- it activates automatically whenever a caller
passes `augment=True`, same as before -- since the code itself isn't wrong, just not yet proven
to help; a future attempt could try a higher `MASK_ERASE_PROB`, a larger erase area, or pairing
it with the attention-supervision loss (doc's option 2) instead of alone. See
docs/Imaging_Confound_Check_documentation.md, "Round 6" for the full writeup.
"""

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset


class SliceCacheDataset(Dataset):
    """Reads images.npy/masks.npy via np.memmap, opened lazily per-worker.

    df must contain img_row, mask_row, class, patient_id, slice_index, dataset.
    use_3channel=True replicates the single stored channel to 3 (for the ResNet-50
    encoder) at __getitem__ time -- never stored that way on disk.

    augment=True applies a random-resized-crop to (image, mask) together at train time only
    -- crops a random `RANDOM_CROP_SCALE_RANGE` fraction of the BOX x BOX canvas and resizes
    back to BOX x BOX. This is NOT plain translation: translation alone doesn't break a
    global-average-pooled head's ability to read "what fraction of the image is padding",
    since GAP is translation-invariant -- the fraction survives a shift unchanged. Varying
    the crop scale changes that fraction itself, randomly, per sample per epoch, so it can no
    longer function as a stable, learnable cue. Image uses bilinear interpolation; mask uses
    nearest-neighbor (bilinear would blend integer class labels into invalid fractional
    values).
    """

    RANDOM_CROP_SCALE_RANGE = (0.85, 1.0)  # fraction of BOX x BOX kept before resizing back up

    MASK_ERASE_PROB = 0.4          # per-sample probability of applying the erase (MSD rows only)
    MASK_ERASE_AREA_RANGE = (0.05, 0.15)  # erased patch area, as a fraction of BOX x BOX
    MASK_ERASE_MAX_TRIES = 30      # random placements tried before falling back to the last one

    def __init__(self, df: pd.DataFrame, images_path, masks_path, use_3channel: bool, box_size: int,
                 augment: bool = False):
        self.records = df.reset_index(drop=True).to_dict("records")
        self.images_path = images_path
        self.masks_path = masks_path
        self.use_3channel = use_3channel
        self.box_size = box_size
        self.augment = augment
        self._images = None
        self._masks = None

    def _ensure_open(self):
        if self._images is None:
            self._images = np.load(self.images_path, mmap_mode="r")
        if self._masks is None:
            self._masks = np.load(self.masks_path, mmap_mode="r")

    def __len__(self):
        return len(self.records)

    def _apply_random_resized_crop(self, img: torch.Tensor, mask: torch.Tensor) -> tuple:
        """img: (C,H,W) float tensor, mask: (H,W) int64 tensor -- same random crop box
        applied to both so they stay pixel-aligned, only the interpolation mode differs."""
        box = self.box_size
        scale = float(np.random.uniform(*self.RANDOM_CROP_SCALE_RANGE))
        crop_size = max(int(round(box * scale)), 32)
        top = int(np.random.randint(0, box - crop_size + 1)) if crop_size < box else 0
        left = int(np.random.randint(0, box - crop_size + 1)) if crop_size < box else 0

        img = TF.resized_crop(img, top, left, crop_size, crop_size, [box, box],
                               interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resized_crop(mask.unsqueeze(0), top, left, crop_size, crop_size, [box, box],
                                interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
        return img, mask

    def _apply_mask_preserving_erase(self, img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """img: (C,H,W) float tensor, mask: (H,W) int64 tensor (post-crop, already pixel-aligned
        with img). Blanks a random patch of tissue that never overlaps mask >= 1, filled with
        this image's own median intensity (same fill strategy validated in the occlusion test --
        avoids introducing a new artificial black/white cue)."""
        h, w = mask.shape
        tumor_mask = (mask >= 1).numpy()

        area = float(np.random.uniform(*self.MASK_ERASE_AREA_RANGE)) * h * w
        side = max(int(round(np.sqrt(area))), 1)

        patch = np.zeros((h, w), dtype=bool)
        for _ in range(self.MASK_ERASE_MAX_TRIES):
            y0 = int(np.random.randint(0, h - side + 1)) if side < h else 0
            x0 = int(np.random.randint(0, w - side + 1)) if side < w else 0
            candidate = np.zeros((h, w), dtype=bool)
            candidate[y0:y0 + side, x0:x0 + side] = True
            if not (candidate & tumor_mask).any():
                patch = candidate
                break
            patch = candidate  # fallback: last-tried placement if none avoided the mask

        fill_value = float(img[0].median())
        patch_t = torch.from_numpy(patch)
        erased = img.clone()
        for c in range(erased.shape[0]):
            erased[c][patch_t] = fill_value
        return erased

    def __getitem__(self, idx):
        self._ensure_open()
        row = self.records[idx]

        img = np.asarray(self._images[row["img_row"]], dtype=np.float32) / 255.0
        img = np.repeat(img[None, :, :], 3, axis=0) if self.use_3channel else img[None, :, :]

        mask_row = int(row["mask_row"])
        has_mask = mask_row >= 0
        if has_mask:
            mask = np.asarray(self._masks[mask_row], dtype=np.int64)
        else:
            mask = np.zeros((self.box_size, self.box_size), dtype=np.int64)

        img_t = torch.from_numpy(img.copy())
        mask_t = torch.from_numpy(mask.copy())

        if self.augment:
            img_t, mask_t = self._apply_random_resized_crop(img_t, mask_t)
            if has_mask and np.random.rand() < self.MASK_ERASE_PROB:
                img_t = self._apply_mask_preserving_erase(img_t, mask_t)

        return {
            "image": img_t,
            "label": torch.tensor(float(row["class"]), dtype=torch.float32),
            "mask": mask_t,
            "has_mask": torch.tensor(has_mask, dtype=torch.bool),
        }
