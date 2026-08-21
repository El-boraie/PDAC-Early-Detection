"""Dataset packing + measurement for the CT slice cache.

Task 1 (measurement): scan data/processed/manifest.csv and report the slice
dimension distribution plus mask-retention rates for candidate center-crop
box sizes, so the box size can be chosen from data rather than guessed.

Task 2 (pack): crop/pad every slice into flat uint8 memmaps under
data/processed/cache/, using the box size chosen from Task 1's numbers.

This script only READS data/processed/images, data/processed/masks and
data/processed/manifest.csv. It never writes into data/processed/ itself --
all outputs go under data/processed/cache/, a new sibling directory.
"""

import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless CLI run, no display available
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, r"C:\FYP\src")
from utils.config import QA_CT_DIR

# --- Config: box size, paths, pad values, budget ----------------------------
CANDIDATE_BOX_SIZES = [256, 320]  # Task 1 candidates
SELECTED_BOX_SIZE = 320  # repacked 2026-07-17: moved to a rented GPU, removing the retrain-risk
# constraint that made 256 the reluctant local choice -- 320 has ZERO measured mask-pixel loss.
PROCESSED_DIR = Path(r"C:\FYP\data\processed")
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
CACHE_DIR = PROCESSED_DIR / "cache"  # BOX=320 promoted to canonical data/processed/cache/ after
# the RunPod training run confirmed it; the superseded BOX=256 pack is archived at
# data/processed/cache_box256_archive/.
IMAGES_MEMMAP_PATH = CACHE_DIR / "images.npy"
MASKS_MEMMAP_PATH = CACHE_DIR / "masks.npy"
MANIFEST_CACHE_PATH = CACHE_DIR / "manifest_cache.csv"
CACHE_META_PATH = CACHE_DIR / "cache_meta.json"
PAD_VALUE_IMAGE = 0  # -150 HU post-normalization -- low end of the window, air/background
PAD_VALUE_MASK = 0  # background class
VERIFY_SAMPLE_COUNT = 6  # rows to visually spot-check after packing, mixed MSD/NIH
VERIFY_OUTPUT_PATH = QA_CT_DIR / f"pack_verification_box{SELECTED_BOX_SIZE}.png"
FREE_DISK_BUDGET_GB = 21.0
BYTES_PER_GB = 1024 ** 3
IO_THREADS = 16  # I/O-bound reads (mmap header parse, mask load) release the GIL, so threads help


def load_manifest() -> pd.DataFrame:
    """Load manifest.csv and print row-count sanity check (MSD vs NIH)."""
    df = pd.read_csv(MANIFEST_PATH)
    counts = df["dataset"].value_counts()
    print(f"Loaded manifest: {len(df)} rows -> {dict(counts)}")
    return df


def measure_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    """Read one slice's (height, width) per PATIENT via mmap header-only access,
    then broadcast that shape to every row of that patient.

    All slices of one patient come from the same resampled volume at the same
    1x1x1mm spacing, so they share one in-plane shape -- confirmed by sampling
    multiple slices per patient before relying on this. That cuts the scan
    from 90,693 file opens down to one per patient (~361).

    Uses mmap_mode='r' so pixel data is never actually loaded -- only the
    .npy header is parsed to get shape. Reads are threaded since this is
    I/O-bound (the GIL is released during file I/O). Row-level errors are
    logged and skipped rather than crashing the whole scan.
    """
    first_per_patient = df.drop_duplicates(subset="patient_id", keep="first")

    def read_shape(row):
        path = PROCESSED_DIR / row.image_path
        try:
            arr = np.load(path, mmap_mode="r")
            return row.patient_id, arr.shape, None
        except Exception as e:
            return row.patient_id, None, str(e)

    rows = list(first_per_patient.itertuples(index=False))
    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex:
        results = list(tqdm(
            ex.map(read_shape, rows), total=len(rows), desc="measuring dims (per patient)"
        ))

    shape_by_patient = {}
    errors = []
    for patient_id, shape, err in results:
        if err:
            errors.append((patient_id, err))
        else:
            shape_by_patient[patient_id] = shape

    if errors:
        print(f"\n{len(errors)} patient(s) failed to read during dimension measurement:")
        for patient_id, msg in errors[:20]:
            print(f"  patient={patient_id}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    out = df.copy()
    heights = out["patient_id"].map(lambda p: shape_by_patient.get(p, (-1, -1))[0])
    widths = out["patient_id"].map(lambda p: shape_by_patient.get(p, (-1, -1))[1])
    out["height"] = heights.astype("int64")
    out["width"] = widths.astype("int64")
    return out


def report_dimension_distribution(dims_df: pd.DataFrame) -> None:
    """Print min/max/median/value_counts of height & width, MSD vs NIH separately."""
    valid = dims_df[(dims_df["height"] > 0) & (dims_df["width"] > 0)]
    dropped = len(dims_df) - len(valid)
    if dropped:
        print(f"\n(dropped {dropped} unreadable rows from distribution stats)")

    for dataset_name, group in valid.groupby("dataset"):
        print(f"\n=== {dataset_name} slice dimensions (n={len(group)}) ===")
        for dim in ["height", "width"]:
            s = group[dim]
            print(f"  {dim}: min={s.min()}, median={s.median()}, max={s.max()}")
        print("  (height, width) value_counts (top 15):")
        combo = list(zip(group["height"], group["width"]))
        vc = pd.Series(combo).value_counts().head(15)
        for (h, w), count in vc.items():
            print(f"    {h}x{w}: {count}")


def _crop_lost_fraction(nonzero: np.ndarray, h: int, w: int, total_nonzero: int, box_size: int) -> float:
    """Fraction of nonzero pixels a center crop-or-pad to box_size would lose."""
    if total_nonzero == 0:
        return 0.0
    row_start = max(0, (h - box_size) // 2) if h > box_size else 0
    row_end = row_start + min(h, box_size)
    col_start = max(0, (w - box_size) // 2) if w > box_size else 0
    col_end = col_start + min(w, box_size)
    kept_mask = np.zeros_like(nonzero)
    kept_mask[row_start:row_end, col_start:col_end] = True
    retained = int((nonzero & kept_mask).sum())
    return 1.0 - (retained / total_nonzero)


def compute_crop_retention(dims_df: pd.DataFrame, box_sizes: list) -> dict:
    """For every MSD slice with a mask, load the mask ONCE and compute what
    fraction of nonzero mask pixels (pancreas=1, tumour=2) a center
    crop-or-pad would lose, for every candidate box size in one pass --
    avoids re-reading the same 72,077 masks once per box size.

    If a dimension is <= box_size, that dimension is padded (nothing lost).
    If a dimension is > box_size, that dimension is center-cropped and any
    nonzero mask pixel outside the crop window counts as lost.

    Mask loads are threaded since this is I/O-bound.
    """
    msd = dims_df[(dims_df["dataset"] == "MSD") & (dims_df["height"] > 0)]
    rows = list(msd.itertuples(index=False))
    results = {box_size: [] for box_size in box_sizes}
    errors = []

    def process_row(row):
        mask_path = PROCESSED_DIR / row.mask_path
        try:
            mask = np.load(mask_path)
            h, w = mask.shape
            nonzero = mask != 0
            total_nonzero = int(nonzero.sum())
            per_box = {
                box_size: _crop_lost_fraction(nonzero, h, w, total_nonzero, box_size)
                for box_size in box_sizes
            }
            return row.patient_id, row.slice_index, total_nonzero, per_box, None
        except Exception as e:
            return row.patient_id, row.slice_index, None, None, str(e)

    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex:
        for patient_id, slice_index, total_nonzero, per_box, err in tqdm(
            ex.map(process_row, rows), total=len(rows), desc="crop retention (all box sizes)"
        ):
            if err:
                errors.append((patient_id, err))
                continue
            for box_size in box_sizes:
                results[box_size].append((patient_id, slice_index, total_nonzero, per_box[box_size]))

    if errors:
        print(f"\n{len(errors)} mask row(s) failed during crop-retention check:")
        for patient_id, msg in errors[:20]:
            print(f"  patient={patient_id}: {msg}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")

    return {
        box_size: pd.DataFrame(rows_, columns=["patient_id", "slice_index", "total_nonzero", "lost_fraction"])
        for box_size, rows_ in results.items()
    }


def report_crop_retention(retention_df: pd.DataFrame, box_size: int) -> None:
    """Print % of MSD slices with zero mask-pixel loss, and the loss distribution for the rest."""
    with_mask_pixels = retention_df[retention_df["total_nonzero"] > 0]
    n = len(with_mask_pixels)
    zero_loss = (with_mask_pixels["lost_fraction"] == 0.0).sum()
    print(f"\n=== Crop retention @ {box_size}x{box_size} ===")
    print(f"MSD slices with nonzero mask pixels: {n}")
    print(f"  Zero pixel loss: {zero_loss} ({100 * zero_loss / n:.3f}%)")

    lossy = with_mask_pixels[with_mask_pixels["lost_fraction"] > 0.0]
    if len(lossy):
        print(f"  Slices with SOME loss: {len(lossy)} ({100 * len(lossy) / n:.3f}%)")
        print(f"  Lost-fraction distribution among lossy slices:")
        print(f"    min={lossy['lost_fraction'].min():.4f}, median={lossy['lost_fraction'].median():.4f}, "
              f"mean={lossy['lost_fraction'].mean():.4f}, max={lossy['lost_fraction'].max():.4f}")
        for q in [0.5, 0.9, 0.95, 0.99, 1.0]:
            print(f"    p{int(q*100)}: {lossy['lost_fraction'].quantile(q):.4f}")
    else:
        print("  No slices lost any mask pixels.")


def compute_disk_requirements(box_size: int, n_images: int, n_masks: int) -> dict:
    """Exact byte math for the packed cache: images = n_images*box^2, masks (MSD only) = n_masks*box^2."""
    image_bytes = n_images * box_size * box_size
    mask_bytes = n_masks * box_size * box_size
    total_bytes = image_bytes + mask_bytes
    return {
        "box_size": box_size,
        "image_bytes": image_bytes,
        "mask_bytes": mask_bytes,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / BYTES_PER_GB,
        "fits_budget": total_bytes / BYTES_PER_GB <= FREE_DISK_BUDGET_GB,
    }


def report_disk_requirements(req: dict) -> None:
    print(f"\n=== Disk requirement @ {req['box_size']}x{req['box_size']} ===")
    print(f"  images.npy: {req['image_bytes']:,} bytes ({req['image_bytes'] / BYTES_PER_GB:.3f} GB)")
    print(f"  masks.npy:  {req['mask_bytes']:,} bytes ({req['mask_bytes'] / BYTES_PER_GB:.3f} GB)")
    print(f"  total:      {req['total_bytes']:,} bytes ({req['total_gb']:.3f} GB)")
    verdict = "FITS" if req["fits_budget"] else "DOES NOT FIT"
    print(f"  budget: {FREE_DISK_BUDGET_GB} GB free -> {verdict}")


def run_task1_measurement() -> None:
    """Full Task 1 pipeline: measure dimensions, crop retention, disk cost. Read-only."""
    df = load_manifest()

    dims_df = measure_dimensions(df)
    report_dimension_distribution(dims_df)

    n_images = len(df)
    n_masks = int((df["dataset"] == "MSD").sum())

    retention_by_box = compute_crop_retention(dims_df, CANDIDATE_BOX_SIZES)
    for box_size in CANDIDATE_BOX_SIZES:
        report_crop_retention(retention_by_box[box_size], box_size)

    for box_size in CANDIDATE_BOX_SIZES:
        req = compute_disk_requirements(box_size, n_images, n_masks)
        report_disk_requirements(req)


# =============================================================================
# Task 2 -- pack
# =============================================================================

def ensure_cache_dir() -> None:
    """Create data/processed/cache/ if it doesn't already exist. Idempotent."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def crop_or_pad(arr: np.ndarray, box_size: int, fill_value: int) -> np.ndarray:
    """Center-crop or pad (never resize/interpolate) a 2D array to (box_size, box_size).

    Each axis is handled independently: if the source is >= box_size along
    that axis it is center-cropped; if smaller, the source is centered inside
    a fill_value-filled canvas. Pure index slicing -- no interpolation, so
    mask class values (0/1/2) are never altered or blended.
    """
    h, w = arr.shape
    out = np.full((box_size, box_size), fill_value, dtype=arr.dtype)

    if h >= box_size:
        src_r0 = (h - box_size) // 2
        src_r1 = src_r0 + box_size
        dst_r0, dst_r1 = 0, box_size
    else:
        src_r0, src_r1 = 0, h
        dst_r0 = (box_size - h) // 2
        dst_r1 = dst_r0 + h

    if w >= box_size:
        src_c0 = (w - box_size) // 2
        src_c1 = src_c0 + box_size
        dst_c0, dst_c1 = 0, box_size
    else:
        src_c0, src_c1 = 0, w
        dst_c0 = (box_size - w) // 2
        dst_c1 = dst_c0 + w

    out[dst_r0:dst_r1, dst_c0:dst_c1] = arr[src_r0:src_r1, src_c0:src_c1]
    return out


def assign_row_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add img_row (index into images.npy, = manifest row order) and mask_row
    (index into masks.npy for MSD rows, -1 for NIH rows with no mask)."""
    out = df.reset_index(drop=True).copy()
    out["img_row"] = np.arange(len(out), dtype=np.int64)

    is_msd = (out["dataset"] == "MSD").to_numpy()
    mask_row = np.full(len(out), -1, dtype=np.int64)
    mask_row[is_msd] = np.arange(int(is_msd.sum()), dtype=np.int64)
    out["mask_row"] = mask_row
    return out


def build_memmaps(df: pd.DataFrame, box_size: int) -> tuple:
    """Crop/pad every slice and write it into flat uint8 .npy memmaps.

    images.npy holds every manifest row in order; masks.npy holds MSD rows
    only, indexed by mask_row. Reads are threaded (I/O-bound); each thread
    only loads+crops -- writes into the memmaps happen in the main thread to
    keep row assignment simple and avoid any concurrent-write ambiguity.
    Row-level errors are logged and skipped (row is left at its fill value)
    rather than crashing the whole pack.
    """
    n_images = len(df)
    n_masks = int((df["dataset"] == "MSD").sum())

    images_mm = np.lib.format.open_memmap(
        IMAGES_MEMMAP_PATH, mode="w+", dtype=np.uint8, shape=(n_images, box_size, box_size)
    )
    images_mm[:] = PAD_VALUE_IMAGE
    masks_mm = np.lib.format.open_memmap(
        MASKS_MEMMAP_PATH, mode="w+", dtype=np.uint8, shape=(n_masks, box_size, box_size)
    )
    masks_mm[:] = PAD_VALUE_MASK

    rows = list(df.itertuples(index=False))
    image_errors = []
    mask_errors = []

    def load_and_crop_image(row):
        try:
            arr = np.load(PROCESSED_DIR / row.image_path)
            return row.img_row, crop_or_pad(arr, box_size, PAD_VALUE_IMAGE), None
        except Exception as e:
            return row.img_row, None, (row.patient_id, row.image_path, str(e))

    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex:
        for img_row, cropped, err in tqdm(
            ex.map(load_and_crop_image, rows), total=len(rows), desc=f"packing images @ {box_size}"
        ):
            if err:
                image_errors.append(err)
                continue
            images_mm[img_row] = cropped

    msd_rows = [row for row in rows if row.mask_row >= 0]

    def load_and_crop_mask(row):
        try:
            arr = np.load(PROCESSED_DIR / row.mask_path)
            return row.mask_row, crop_or_pad(arr, box_size, PAD_VALUE_MASK), None
        except Exception as e:
            return row.mask_row, None, (row.patient_id, row.mask_path, str(e))

    with ThreadPoolExecutor(max_workers=IO_THREADS) as ex:
        for mask_row, cropped, err in tqdm(
            ex.map(load_and_crop_mask, msd_rows), total=len(msd_rows), desc=f"packing masks @ {box_size}"
        ):
            if err:
                mask_errors.append(err)
                continue
            masks_mm[mask_row] = cropped

    images_mm.flush()
    masks_mm.flush()

    for label, errors in [("image", image_errors), ("mask", mask_errors)]:
        if errors:
            print(f"\n{len(errors)} {label} row(s) failed during packing:")
            for patient_id, path, msg in errors[:20]:
                print(f"  patient={patient_id} path={path}: {msg}")
            if len(errors) > 20:
                print(f"  ... and {len(errors) - 20} more")

    return images_mm, masks_mm


def compute_manifest_checksum(path: Path) -> str:
    """sha256 hex digest of the source manifest file, for provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_manifest_cache(df: pd.DataFrame) -> None:
    """Write the manifest plus img_row/mask_row columns to manifest_cache.csv."""
    df.to_csv(MANIFEST_CACHE_PATH, index=False)
    print(f"Wrote {MANIFEST_CACHE_PATH} ({len(df)} rows)")


def write_cache_meta(box_size: int, n_images: int, n_masks: int) -> None:
    """Write cache_meta.json: box size, dtype, row counts, pad values, and a
    checksum/mtime of the source manifest so the cache can be proven to match
    (or not match) its source later."""
    meta = {
        "box_size": box_size,
        "dtype": "uint8",
        "n_images": n_images,
        "n_masks": n_masks,
        "pad_value_image": PAD_VALUE_IMAGE,
        "pad_value_mask": PAD_VALUE_MASK,
        "source_manifest_path": str(MANIFEST_PATH),
        "source_manifest_sha256": compute_manifest_checksum(MANIFEST_PATH),
        "source_manifest_mtime": datetime.fromtimestamp(MANIFEST_PATH.stat().st_mtime).isoformat(),
        "created_at": datetime.now().isoformat(),
    }
    CACHE_META_PATH.write_text(json.dumps(meta, indent=2))
    print(f"Wrote {CACHE_META_PATH}")
    print(json.dumps(meta, indent=2))


def report_disk_usage() -> None:
    """Print actual on-disk size of data/processed/cache/ and confirm it fits the budget."""
    total_bytes = sum(f.stat().st_size for f in CACHE_DIR.iterdir() if f.is_file())
    total_gb = total_bytes / BYTES_PER_GB
    print(f"\n=== Cache disk usage ({CACHE_DIR}) ===")
    for f in sorted(CACHE_DIR.iterdir()):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size:,} bytes ({f.stat().st_size / BYTES_PER_GB:.3f} GB)")
    print(f"  total: {total_bytes:,} bytes ({total_gb:.3f} GB)")
    verdict = "FITS" if total_gb <= FREE_DISK_BUDGET_GB else "DOES NOT FIT"
    print(f"  budget: {FREE_DISK_BUDGET_GB} GB free -> {verdict}")


def verify_pack(df_cache: pd.DataFrame, box_size: int, sample_count: int) -> None:
    """Pick a mix of MSD/NIH rows, compare packed rows against a fresh crop of
    the original source file (exact array equality -- catches any off-by-one
    indexing bug that a shape check would miss), and save a side-by-side plot
    for a human sanity check of centering/alignment/clipping.
    """
    images_mm = np.load(IMAGES_MEMMAP_PATH, mmap_mode="r")
    masks_mm = np.load(MASKS_MEMMAP_PATH, mmap_mode="r")

    n_msd_sample = sample_count // 2
    n_nih_sample = sample_count - n_msd_sample
    msd_sample = df_cache[df_cache["dataset"] == "MSD"].sample(n=n_msd_sample, random_state=42)
    nih_sample = df_cache[df_cache["dataset"] == "NIH"].sample(n=n_nih_sample, random_state=42)
    sample = pd.concat([msd_sample, nih_sample]).reset_index(drop=True)

    fig, axes = plt.subplots(len(sample), 3, figsize=(12, 4 * len(sample)))
    if len(sample) == 1:
        axes = axes.reshape(1, 3)

    all_match = True
    for i, row in enumerate(sample.itertuples(index=False)):
        original_img = np.load(PROCESSED_DIR / row.image_path)
        expected_img = crop_or_pad(original_img, box_size, PAD_VALUE_IMAGE)
        packed_img = np.array(images_mm[row.img_row])
        img_match = np.array_equal(expected_img, packed_img)
        all_match &= img_match

        axes[i, 0].imshow(original_img, cmap="gray")
        axes[i, 0].set_title(f"{row.patient_id} slice{row.slice_index} ({row.dataset})\noriginal {original_img.shape}")
        axes[i, 1].imshow(packed_img, cmap="gray")
        axes[i, 1].set_title(f"packed img_row={row.img_row}\nmatches source crop: {img_match}")

        if row.mask_row >= 0:
            original_mask = np.load(PROCESSED_DIR / row.mask_path)
            expected_mask = crop_or_pad(original_mask, box_size, PAD_VALUE_MASK)
            packed_mask = np.array(masks_mm[row.mask_row])
            mask_match = np.array_equal(expected_mask, packed_mask)
            all_match &= mask_match

            axes[i, 2].imshow(packed_img, cmap="gray")
            axes[i, 2].imshow(np.ma.masked_where(packed_mask == 0, packed_mask), cmap="autumn", alpha=0.5)
            axes[i, 2].set_title(f"packed img+mask overlay\nmask_row={row.mask_row}, matches: {mask_match}")
        else:
            axes[i, 2].axis("off")
            axes[i, 2].set_title("NIH -- no mask")

        for ax in axes[i]:
            ax.axis("off")

    fig.tight_layout()
    QA_CT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(VERIFY_OUTPUT_PATH, dpi=100)
    plt.close(fig)

    print(f"\n=== Verification ({len(sample)} sampled rows) ===")
    print(f"All packed rows exactly match a fresh crop of their source file: {all_match}")
    print(f"Verification plot saved to {VERIFY_OUTPUT_PATH}")


def run_task2_pack(box_size: int = SELECTED_BOX_SIZE) -> None:
    """Full Task 2 pipeline: pack every slice into the cache, write sidecars, verify."""
    ensure_cache_dir()

    df = load_manifest()
    df = assign_row_indices(df)

    n_images = len(df)
    n_masks = int((df["dataset"] == "MSD").sum())

    build_memmaps(df, box_size)
    write_manifest_cache(df)
    write_cache_meta(box_size, n_images, n_masks)
    report_disk_usage()
    verify_pack(df, box_size, VERIFY_SAMPLE_COUNT)


if __name__ == "__main__":
    run_task2_pack()
