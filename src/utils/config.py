"""Central path configuration for the PDAC FYP.

Every NEW notebook created as part of the v3 folder-structure migration
(`FYP_Folder_Structure_Migration.md`) imports its paths from here instead of
hardcoding `C:\\FYP\\...` inline -- that's what makes the tree movable.

`clinical.ipynb`, `imaging.ipynb`, `CT EDA.ipynb`, and `Tabular EDA.ipynb`
predate this file and keep their own inline path constants deliberately --
they are done and verified, not to be rewritten just to import from here.
"""

from pathlib import Path

# --- Project root ---
ROOT = Path(r"C:\FYP")

# --- Raw data (never touched, ever) ---
RAW_DIR = ROOT / "data" / "raw"
MSD_DIR = RAW_DIR / "Task07_Pancreas"
MSD_IMG_DIR = MSD_DIR / "imagesTr"
MSD_MASK_DIR = MSD_DIR / "labelsTr"
NIH_DIR = RAW_DIR / "NIH_Pancreas_CT" / "0024_pancreas_ct"
NIH_IMG_DIR = NIH_DIR / "images"
TABULAR_RAW_PATH = RAW_DIR / "Debernardi et al 2020 data.csv"  # original filename kept, spaces and all

# --- Processed data (fixed-rule pipeline inputs only -- no figures, no metrics, no models) ---
PROCESSED_DIR = ROOT / "data" / "processed"
MANIFEST_PATH = PROCESSED_DIR / "manifest.csv"
MANIFEST_MSD_ONLY_BACKUP_PATH = PROCESSED_DIR / "manifest_msd_only_backup.csv"
TABULAR_CLEAN_PATH = PROCESSED_DIR / "tabular_clean.csv"  # written by tabular_clean.ipynb

# --- Packed slice cache (BOX=320, written by pack_slice_cache.py) -----------
# Repacked at BOX=320 on 2026-07-17 -- training moved to a rented RunPod GPU, removing the
# retrain-risk constraint that made 256 the reluctant local choice; 320 has zero measured
# mask-pixel loss on MSD masks (see docs/Imaging_Slice_Cache_Measurement_documentation.md).
# The superseded BOX=256 cache is archived at data/processed/cache_box256_archive/.
CACHE_DIR = PROCESSED_DIR / "cache"
CACHE_IMAGES_PATH = CACHE_DIR / "images.npy"  # (90693, 320, 320) uint8 memmap, all rows
CACHE_MASKS_PATH = CACHE_DIR / "masks.npy"  # (72077, 320, 320) uint8 memmap, MSD rows only
CACHE_MANIFEST_PATH = CACHE_DIR / "manifest_cache.csv"  # manifest.csv + img_row/mask_row
CACHE_META_PATH = CACHE_DIR / "cache_meta.json"

# --- Outputs: figures (.png), committed to git ---
OUTPUTS_DIR = ROOT / "outputs"
EDA_CT_DIR = OUTPUTS_DIR / "eda" / "ct"
EDA_TABULAR_DIR = OUTPUTS_DIR / "eda" / "tabular"
QA_CT_DIR = OUTPUTS_DIR / "qa" / "ct"
QA_TABULAR_DIR = OUTPUTS_DIR / "qa" / "tabular"
EVAL_IMAGING_DIR = OUTPUTS_DIR / "eval" / "imaging"
EVAL_CLINICAL_DIR = OUTPUTS_DIR / "eval" / "clinical"
EVAL_FUSION_DIR = OUTPUTS_DIR / "eval" / "fusion"

# --- Results: metrics (.csv), committed to git ---
RESULTS_DIR = ROOT / "results"
RESULTS_CLINICAL_DIR = RESULTS_DIR / "clinical"
RESULTS_IMAGING_DIR = RESULTS_DIR / "imaging"
RESULTS_FUSION_DIR = RESULTS_DIR / "fusion"

CLINICAL_MODEL_COMPARISON_PATH = RESULTS_CLINICAL_DIR / "model_comparison.csv"
CLINICAL_IMPUTER_BENCHMARK_PATH = RESULTS_CLINICAL_DIR / "imputer_benchmark.csv"
CLINICAL_OOF_PREDICTIONS_PATH = RESULTS_CLINICAL_DIR / "oof_predictions.csv"

IMAGING_MODEL_COMPARISON_PATH = RESULTS_IMAGING_DIR / "model_comparison.csv"
IMAGING_OOF_PREDICTIONS_PATH = RESULTS_IMAGING_DIR / "oof_predictions.csv"

FUSION_PAIR_COMPARISON_PATH = RESULTS_FUSION_DIR / "pair_comparison.csv"
# Written by fusion.ipynb, not by imaging_evaluation.ipynb -- lives under results/fusion/
# (not results/imaging/) because patient-level slice aggregation is a fusion/inference-time
# decision, measured here from imaging's OOF file; the imaging branch itself was trained and
# evaluated purely slice-level and is not retrofitted.
FUSION_IMAGING_AGGREGATION_PATH = RESULTS_FUSION_DIR / "imaging_slice_vs_volume_aggregation.csv"

# --- Checkpoints: trained things only, gitignored ---
CHECKPOINTS_DIR = ROOT / "checkpoints"
CHECKPOINTS_CLINICAL_FINAL_DIR = CHECKPOINTS_DIR / "clinical" / "final"
CHECKPOINTS_IMAGING_CANDIDATES_DIR = CHECKPOINTS_DIR / "imaging" / "candidates"
CHECKPOINTS_IMAGING_FINAL_DIR = CHECKPOINTS_DIR / "imaging" / "final"
# No model.pt/model.pkl here -- fusion.ipynb's combination rule is a fixed, hand-set rule,
# not a fitted/learned artifact (see FYP_Folder_Structure_Migration.md Section 8). Only a
# model_card.json lives here, documenting the rule + weights + both branches' caveats.
CHECKPOINTS_FUSION_FINAL_DIR = CHECKPOINTS_DIR / "fusion" / "final"

# --- Shared constants ---
RANDOM_SEED = 42  # matches clinical.ipynb / imaging.ipynb -- keep consistent for reproducibility


def ensure_dirs() -> None:
    """Create every output/results/checkpoints directory this config declares.
    Safe to call repeatedly (idempotent) -- notebooks call this once near the top."""
    for path in [
        EDA_CT_DIR, EDA_TABULAR_DIR, QA_CT_DIR, QA_TABULAR_DIR,
        EVAL_IMAGING_DIR, EVAL_CLINICAL_DIR, EVAL_FUSION_DIR,
        RESULTS_CLINICAL_DIR, RESULTS_IMAGING_DIR, RESULTS_FUSION_DIR,
        CHECKPOINTS_CLINICAL_FINAL_DIR, CHECKPOINTS_IMAGING_CANDIDATES_DIR, CHECKPOINTS_IMAGING_FINAL_DIR,
        CHECKPOINTS_FUSION_FINAL_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
