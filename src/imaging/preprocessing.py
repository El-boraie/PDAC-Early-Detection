import numpy as np
import SimpleITK as sitk

TARGET_SPACING = (1.0, 1.0, 1.0)  # mm, isotropic
HU_MIN, HU_MAX = -150, 250
BOX_SIZE = 320
PAD_VALUE_IMAGE = 0


def reorient_to_ras(image: sitk.Image) -> sitk.Image:
    return sitk.DICOMOrient(image, "RAS")


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


def clip_hu(array, hu_min, hu_max):
    return np.clip(array, hu_min, hu_max)


def normalize_to_unit_range(array, hu_min, hu_max):
    return (array - hu_min) / (hu_max - hu_min)


def crop_or_pad(arr: np.ndarray, box_size: int, fill_value: int) -> np.ndarray:
    """Identical to pack_slice_cache.py's crop_or_pad -- pure center-crop/pad, no
    interpolation, so it must be reproduced exactly, not approximated."""
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


def nifti_bytes_to_resampled_slices(nifti_bytes: bytes, suffix: str = ".nii.gz") -> np.ndarray:
    """Raw NIfTI file bytes -> (N, H, W) uint8 array, one row per resampled axial slice,
    at the volume's natural resampled size (NOT yet cropped/padded to BOX) -- this is the
    same stage as data/processed/images/*.npy, the stage nifti_to_slices() crops from next.
    SimpleITK has no in-memory NIfTI reader, so this writes to a temp file first."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp) / f"upload{suffix}"
        tmp_path.write_bytes(nifti_bytes)
        image = sitk.ReadImage(str(tmp_path))

    image = reorient_to_ras(image)

    image = resample_to_spacing(image, TARGET_SPACING, sitk.sitkLinear, default_value=HU_MIN)

    volume = sitk.GetArrayFromImage(image)  # (z, y, x) -- SimpleITK convention
    volume = clip_hu(volume, HU_MIN, HU_MAX)
    volume = normalize_to_unit_range(volume, HU_MIN, HU_MAX)
    return np.round(volume * 255.0).astype(np.uint8)


def nifti_bytes_to_slices(nifti_bytes: bytes, suffix: str = ".nii.gz") -> np.ndarray:
    """Raw NIfTI file bytes -> (N, BOX, BOX) uint8 array, center-cropped/padded to
    BOX_SIZE -- the same final space as data/processed/cache/images.npy."""
    resampled = nifti_bytes_to_resampled_slices(nifti_bytes, suffix)
    slices = [crop_or_pad(resampled[z], BOX_SIZE, PAD_VALUE_IMAGE) for z in range(resampled.shape[0])]
    return np.stack(slices)
