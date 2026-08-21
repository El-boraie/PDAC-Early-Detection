import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config import (
    TABULAR_CLEAN_PATH, CACHE_IMAGES_PATH, CACHE_MANIFEST_PATH,
    CHECKPOINTS_CLINICAL_FINAL_DIR, CHECKPOINTS_IMAGING_FINAL_DIR, CHECKPOINTS_FUSION_FINAL_DIR,
    RANDOM_SEED,
)
from imaging.models import ResNet50UNet, DetectionOnlyWrapper

# --- Fixed, hand-set rules (never fitted) -- mirrors checkpoints/fusion/final/model_card.json ---
W_IMAGING = 0.4
W_TABULAR = 0.6
IMAGING_SLICE_AGGREGATION = "mean"
IMAGING_VOLUME_BATCH_SIZE = 32

SLICE_AGGREGATORS = {"mean": np.mean, "median": np.median, "max": np.max}

TABULAR_FEATURES = ["creatinine", "LYVE1", "REG1B", "TFF1", "plasma_CA19_9", "age", "sex"]


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Clinical branch -------------------------------------------------------

class MICE_CA19_9Imputer:
    """Verbatim copy of the class pickled inside ca19_9_imputer.pkl (from a notebook's
    __main__). joblib.load() needs this exact class, by this exact name, to unpickle --
    copying it here is required, not a style choice. See fusion.ipynb / clinical_shap.ipynb."""

    PREDICTORS = ["creatinine", "LYVE1", "REG1B", "TFF1", "age"]
    TARGET = "plasma_CA19_9"

    def __init__(self, random_state=RANDOM_SEED):
        from sklearn.experimental import enable_iterative_imputer  # noqa: F401
        from sklearn.impute import IterativeImputer
        from sklearn.linear_model import BayesianRidge
        self.imputer = IterativeImputer(estimator=BayesianRidge(), random_state=random_state)

    def fit(self, train_df):
        self.imputer.fit(train_df[self.PREDICTORS + [self.TARGET]])
        return self

    def transform(self, target_df):
        out = target_df.copy()
        out[self.TARGET] = self.imputer.transform(out[self.PREDICTORS + [self.TARGET]])[:, -1]
        return out


def _patch_main_for_imputer_unpickling() -> None:
    """ca19_9_imputer.pkl was pickled from a notebook cell, which runs as __main__ -- so
    joblib looks for __main__.MICE_CA19_9Imputer specifically at unpickle time, regardless
    of which module actually defines the class. Called right before every joblib.load() of
    that file, not once at import time: Streamlit's multipage app runner swaps in a NEW
    __main__ module object on every page rerun (each st.Page file is exec'd as if it were
    __main__), so a one-time patch at inference.py's import time goes stale as soon as
    Streamlit replaces sys.modules['__main__'] for the next rerun -- confirmed by this
    failing on the second navigation into the dashboard's Predict page."""
    import sys as _sys
    _sys.modules["__main__"].MICE_CA19_9Imputer = MICE_CA19_9Imputer


class ClinicalBranch:
    def __init__(self, model, imputer, calibrator, model_card, feature_order, explainer, expected_value):
        self.model = model
        self.imputer = imputer
        self.calibrator = calibrator
        self.model_card = model_card
        self.feature_order = feature_order
        self.explainer = explainer
        self.expected_value = expected_value  # log-odds base value


def load_clinical_branch() -> ClinicalBranch:
    import shap
    from sklearn.experimental import enable_iterative_imputer  # noqa: F401

    model = joblib.load(CHECKPOINTS_CLINICAL_FINAL_DIR / "model.pkl")
    _patch_main_for_imputer_unpickling()
    imputer = joblib.load(CHECKPOINTS_CLINICAL_FINAL_DIR / "ca19_9_imputer.pkl")
    calibrator = joblib.load(CHECKPOINTS_CLINICAL_FINAL_DIR / "calibrator.pkl")
    with open(CHECKPOINTS_CLINICAL_FINAL_DIR / "model_card.json") as f:
        model_card = json.load(f)

    feature_order = list(model.feature_names_in_)
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    
    df = pd.read_csv(TABULAR_CLEAN_PATH)
    X_imputed = imputer.transform(df)[feature_order].reset_index(drop=True)
    shap_values = np.asarray(explainer.shap_values(X_imputed))
    if shap_values.ndim == 3:
        shap_values = shap_values[..., -1] if shap_values.shape[-1] == 2 else shap_values[-1]

    expected_value = explainer.expected_value
    if isinstance(expected_value, (list, np.ndarray)):
        expected_value = np.asarray(expected_value).reshape(-1)[-1]
    expected_value = float(expected_value)

    raw_margin_pred = model.predict(X_imputed, output_margin=True)
    reconstructed = expected_value + shap_values.sum(axis=1)
    max_abs_error = float(np.max(np.abs(reconstructed - raw_margin_pred)))
    assert max_abs_error < 1e-3, (
        f"SHAP additivity check FAILED at clinical branch load (max abs error {max_abs_error:.6f}) "
        "-- explanations would not sum to the model's actual prediction. Not safe to serve."
    )

    return ClinicalBranch(model, imputer, calibrator, model_card, feature_order, explainer, expected_value)


def run_tabular_branch(clinical: ClinicalBranch, patient_features: dict) -> float:
    """patient_features: dict keyed by TABULAR_FEATURES (plasma_CA19_9 may be NaN/missing).
    Returns the clinical branch's own calibrated PDAC probability."""
    df = pd.DataFrame([patient_features])[TABULAR_FEATURES]
    df_imputed = clinical.imputer.transform(df)
    raw_proba = clinical.model.predict_proba(df_imputed[TABULAR_FEATURES])[:, 1]
    calibrated = clinical.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]
    return float(calibrated[0])


def explain_tabular(clinical: ClinicalBranch, row_df: pd.DataFrame) -> tuple[dict, float]:
    """row_df: 1-row DataFrame with raw (pre-imputation) TABULAR_FEATURES columns.
    Returns (shap_per_feature, base_value), both in log-odds/margin space."""
    assert len(row_df) == 1, "explain_tabular expects exactly one row"
    row_imputed = clinical.imputer.transform(row_df)[clinical.feature_order]
    row_shap = clinical.explainer.shap_values(row_imputed)
    row_shap = np.asarray(row_shap).reshape(-1)
    return dict(zip(clinical.feature_order, row_shap.tolist())), clinical.expected_value


def ca19_9_was_imputed(patient_features: dict) -> bool:
    value = patient_features.get("plasma_CA19_9")
    return bool(value is None or (isinstance(value, float) and np.isnan(value)))


# --- Imaging branch ---------------------------------------------------------

class ImagingBranch:
    def __init__(self, model, detection_wrapper, calibrator, model_card, box_size, device, caveat):
        self.model = model
        self.detection_wrapper = detection_wrapper
        self.calibrator = calibrator
        self.model_card = model_card
        self.box_size = box_size
        self.device = device
        self.caveat = caveat


def load_imaging_branch(device: torch.device = None) -> ImagingBranch:
    device = device or get_device()

    with open(CHECKPOINTS_IMAGING_FINAL_DIR / "pod_training_run.json") as f:
        pod_run = json.load(f)
    box_size = pod_run["box_size"]

    state_dict = torch.load(CHECKPOINTS_IMAGING_FINAL_DIR / "model.pt", map_location=device, weights_only=True)
    bad_keys = [k for k in state_dict if k.startswith("_orig_mod.")]
    assert not bad_keys, f"torch.compile prefix leakage: {bad_keys[:5]}"

    model = ResNet50UNet(pretrained=False).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    detection_wrapper = DetectionOnlyWrapper(model).to(device).eval()

    calibrator = joblib.load(CHECKPOINTS_IMAGING_FINAL_DIR / "calibrator.pkl")
    with open(CHECKPOINTS_IMAGING_FINAL_DIR / "model_card.json") as f:
        model_card = json.load(f)

    caveat = (
        "KNOWN LIMITATION (checkpoints/imaging/final/model_card.json, confound-check investigation): "
        + model_card["known_limitations"]["confound_check_summary"]
    )

    return ImagingBranch(model, detection_wrapper, calibrator, model_card, box_size, device, caveat)


def _score_slices(imaging: ImagingBranch, slice_stack: torch.Tensor) -> np.ndarray:
    """slice_stack: (N, 3, BOX, BOX). Returns (N,) calibrated per-slice probabilities.
    Batched so a full ~250-slice volume never lands on the GPU in one go."""
    raw_batches = []
    with torch.no_grad():
        for start in range(0, len(slice_stack), IMAGING_VOLUME_BATCH_SIZE):
            batch = slice_stack[start:start + IMAGING_VOLUME_BATCH_SIZE].to(imaging.device)
            _, det_logit = imaging.model(batch)
            raw_batches.append(torch.sigmoid(det_logit).cpu().numpy().reshape(-1))
    raw_proba = np.concatenate(raw_batches)
    return imaging.calibrator.predict_proba(raw_proba.reshape(-1, 1))[:, 1]


def run_imaging_branch(imaging: ImagingBranch, image_tensor: torch.Tensor) -> float:
    """image_tensor: (3, BOX, BOX) float32 in [0, 1], 3-channel-replicated -- one
    preprocessed CT slice. Returns that slice's calibrated probability (the model's
    native, trained/evaluated granularity)."""
    return float(_score_slices(imaging, image_tensor.unsqueeze(0))[0])


def run_imaging_branch_volume(imaging: ImagingBranch, slice_stack: torch.Tensor, aggregation: str = None) -> dict:
    """slice_stack: (N, 3, BOX, BOX) -- a whole CT scan's preprocessed slices, in order.
    Returns the aggregated patient-level score AND the full per-slice vector (the dashboard
    needs the profile to show the near-flat within-patient response, the honest finding)."""
    aggregation = aggregation or IMAGING_SLICE_AGGREGATION
    if aggregation not in SLICE_AGGREGATORS:
        raise ValueError(f"Unknown aggregation '{aggregation}'; expected one of {sorted(SLICE_AGGREGATORS)}")
    if slice_stack.ndim != 4:
        raise ValueError(f"Expected a (N, 3, BOX, BOX) volume, got shape {tuple(slice_stack.shape)}")

    per_slice = _score_slices(imaging, slice_stack)
    return {
        "patient_score": float(SLICE_AGGREGATORS[aggregation](per_slice)),
        "per_slice_proba": per_slice,
        "n_slices": int(len(per_slice)),
        "aggregation": aggregation,
    }


def generate_gradcam(imaging: ImagingBranch, image_tensor: torch.Tensor) -> np.ndarray:
    """image_tensor: (3, BOX, BOX). Returns a (BOX, BOX) float heatmap in [0, 1] from the
    FINAL promoted model (not the per-fold candidates the confound-check notebook used),
    targeting layer4[-1] -- same target layer, same RawScoresOutputTarget as
    imaging_confound_check.ipynb. Caller must still show IMAGING_CAVEAT alongside this:
    Grad-CAM shows where the model looked, not confirmation that it looked at tumour."""
    from pytorch_grad_cam import GradCAM
    from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

    batch = image_tensor.unsqueeze(0).to(imaging.device)
    with GradCAM(model=imaging.detection_wrapper, target_layers=[imaging.model.layer4[-1]]) as cam:
        heatmap = cam(input_tensor=batch, targets=[RawScoresOutputTarget()])[0]
    return heatmap


# --- Fusion ------------------------------------------------------------------

def fuse(clinical: ClinicalBranch = None, imaging: ImagingBranch = None,
         imaging_input: torch.Tensor = None, tabular_input: dict = None) -> dict:
    """imaging_input may be a single slice (3, BOX, BOX) or a whole volume (N, 3, BOX, BOX)
    -- dispatched on tensor rank, recorded in the result. At least one of imaging_input /
    tabular_input must be given; never fabricates a joint score when only one branch's
    input is available. Pass the loaded branch(es) matching whichever input(s) you give."""
    if imaging_input is None and tabular_input is None:
        raise ValueError("fuse() requires at least one of imaging_input or tabular_input.")

    result = {"imaging_calibrated_proba": None, "imaging_granularity": None, "imaging_n_slices": None,
              "tabular_calibrated_proba": None, "fused_score": None, "mode": None}

    if imaging_input is not None:
        assert imaging is not None, "imaging_input given but no ImagingBranch passed"
        if imaging_input.ndim == 3:
            result["imaging_calibrated_proba"] = run_imaging_branch(imaging, imaging_input)
            result["imaging_granularity"] = "single-slice"
            result["imaging_n_slices"] = 1
        elif imaging_input.ndim == 4:
            volume_result = run_imaging_branch_volume(imaging, imaging_input)
            result["imaging_calibrated_proba"] = volume_result["patient_score"]
            result["imaging_granularity"] = f"volume ({volume_result['aggregation']} over slices)"
            result["imaging_n_slices"] = volume_result["n_slices"]
        else:
            raise ValueError(
                f"imaging_input must be (3, BOX, BOX) or (N, 3, BOX, BOX), got {tuple(imaging_input.shape)}"
            )

    if tabular_input is not None:
        assert clinical is not None, "tabular_input given but no ClinicalBranch passed"
        result["tabular_calibrated_proba"] = run_tabular_branch(clinical, tabular_input)

    if imaging_input is not None and tabular_input is not None:
        result["fused_score"] = (
            W_IMAGING * result["imaging_calibrated_proba"] + W_TABULAR * result["tabular_calibrated_proba"]
        )
        result["mode"] = "fused (both modalities)"
    elif imaging_input is not None:
        result["fused_score"] = result["imaging_calibrated_proba"]
        result["mode"] = "single-modality (imaging only)"
    else:
        result["fused_score"] = result["tabular_calibrated_proba"]
        result["mode"] = "single-modality (tabular only)"

    return result


def load_fusion_model_card() -> dict:
    with open(CHECKPOINTS_FUSION_FINAL_DIR / "model_card.json") as f:
        return json.load(f)


# --- Sample-patient picker (guaranteed-working demo path, real cached data) -----

def list_sample_imaging_patients() -> pd.DataFrame:
    """One row per patient in the packed cache: patient_id, dataset (MSD/NIH), class
    (1=cancer/0=healthy), n_slices. For the Predict page's sample-patient dropdown."""
    manifest = pd.read_csv(CACHE_MANIFEST_PATH)
    grouped = manifest.groupby("patient_id").agg(
        dataset=("dataset", "first"), pdac_class=("class", "first"), n_slices=("slice_index", "count"),
    ).reset_index()
    return grouped.sort_values(["dataset", "patient_id"]).reset_index(drop=True)


def load_sample_patient_volume(patient_id: str, box_size: int) -> torch.Tensor:
    """Returns (N, 3, BOX, BOX) float32 tensor for every cached slice of one patient, in
    slice order -- same preprocessing as SliceCacheDataset.__getitem__ (already applied at
    packing time; this just reads it back and 3-channel-replicates)."""
    manifest = pd.read_csv(CACHE_MANIFEST_PATH)
    rows = manifest[manifest["patient_id"] == patient_id].sort_values("slice_index")
    if rows.empty:
        raise ValueError(f"Unknown sample patient_id: {patient_id}")
    images_memmap = np.load(CACHE_IMAGES_PATH, mmap_mode="r")
    imgs = np.stack([
        np.asarray(images_memmap[int(r)], dtype=np.float32) / 255.0 for r in rows["img_row"]
    ])
    return torch.from_numpy(np.repeat(imgs[:, None, :, :], 3, axis=1).copy())


def load_sample_tabular_row(sample_id) -> dict:
    """Real Debernardi patient row (raw, pre-imputation) keyed by TABULAR_FEATURES, for the
    Predict page's sample-patient path on the clinical side."""
    df = pd.read_csv(TABULAR_CLEAN_PATH)
    row = df[df["sample_id"] == sample_id].iloc[0]
    return row[TABULAR_FEATURES].to_dict()


# --- Raw NIfTI upload path (validated against real cached patients; see dashboard build notes) ---

def nifti_upload_to_volume_tensor(nifti_bytes: bytes, box_size: int, suffix: str = ".nii.gz") -> torch.Tensor:
    """Raw .nii.gz file bytes -> (N, 3, BOX, BOX) float32 tensor in [0, 1], same space as
    load_sample_patient_volume(). Uses imaging.preprocessing, reimplemented from
    docs/CT_Preprocessing_documentation.md and verified byte-exact against real cached MSD
    and NIH patients before being trusted here."""
    from imaging.preprocessing import nifti_bytes_to_slices

    slices_uint8 = nifti_bytes_to_slices(nifti_bytes, suffix=suffix)  # (N, BOX, BOX)
    assert slices_uint8.shape[1] == box_size and slices_uint8.shape[2] == box_size
    imgs = slices_uint8.astype(np.float32) / 255.0
    return torch.from_numpy(np.repeat(imgs[:, None, :, :], 3, axis=1).copy())
