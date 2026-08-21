"""Local, inspectable storage for the dashboard's own runtime state -- patient register and
saved case reports. Deliberately separate from data/ (raw/processed ML inputs) and results/
(evaluation metrics): this is dashboard-generated state, not part of the research pipeline.

De-identification (handoff rule 6): patient names live ONLY in patients.csv. Every case
record and report keys on the generated PT- id, never the name, so reports/ and its index
can be shared or inspected without exposing names.
"""

import json
from datetime import date, datetime
from pathlib import Path

import pandas as pd

STORE_DIR = Path(__file__).resolve().parent / "store"
PATIENTS_CSV = STORE_DIR / "patients.csv"
REPORTS_DIR = STORE_DIR / "reports"
REPORTS_INDEX_CSV = STORE_DIR / "reports_index.csv"

PATIENTS_COLUMNS = ["patient_id", "name", "dob", "sex", "created_at"]
REPORTS_INDEX_COLUMNS = [
    "case_id", "patient_id", "patient_name", "created_at",
    "modalities", "fused_score", "risk_band", "mode",
]

# UI risk-banding convention -- a chosen display threshold for the risk-band pill, NOT a
# clinically validated cutoff (no such cutoff exists or is claimed anywhere in this project).
RISK_BAND_LOW_MAX = 0.30
RISK_BAND_MODERATE_MAX = 0.70


def ensure_store() -> None:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    if not PATIENTS_CSV.exists():
        pd.DataFrame(columns=PATIENTS_COLUMNS).to_csv(PATIENTS_CSV, index=False)
    else:
        # Migration: patients.csv from before dob/sex existed -- add the columns rather
        # than break on existing (real, already-registered) rows.
        patients = pd.read_csv(PATIENTS_CSV)
        missing = [c for c in PATIENTS_COLUMNS if c not in patients.columns]
        if missing:
            for col in missing:
                patients[col] = None
            patients = patients[PATIENTS_COLUMNS]
            patients.to_csv(PATIENTS_CSV, index=False)
    if not REPORTS_INDEX_CSV.exists():
        pd.DataFrame(columns=REPORTS_INDEX_COLUMNS).to_csv(REPORTS_INDEX_CSV, index=False)


def classify_risk_band(score: float) -> str:
    if score < RISK_BAND_LOW_MAX:
        return "Low"
    if score < RISK_BAND_MODERATE_MAX:
        return "Moderate"
    return "High"


def compute_age(dob_str: str) -> int | None:
    """Exact age in whole years from an ISO 'YYYY-MM-DD' string, computed live against
    today's date -- never stored, so it's always current ("calculate like a system")."""
    if not dob_str or pd.isna(dob_str):
        return None
    dob = date.fromisoformat(str(dob_str))
    today = date.today()
    return today.year - dob.year - ((today.month, today.day) < (dob.month, dob.day))


# --- Patients ----------------------------------------------------------------

def list_patients() -> pd.DataFrame:
    ensure_store()
    return pd.read_csv(PATIENTS_CSV)


def register_patient(name: str, dob: str = None, sex: str = None) -> str:
    """Issues the next sequential PT-#### id, appends to patients.csv, returns the id.
    dob: ISO 'YYYY-MM-DD' string. sex: 'Female' or 'Male'. Collected once at registration
    so Predict never has to ask the same patient for their own age/sex again."""
    ensure_store()
    patients = pd.read_csv(PATIENTS_CSV)
    next_n = len(patients) + 1
    patient_id = f"PT-{next_n:04d}"
    row = pd.DataFrame([{
        "patient_id": patient_id, "name": name, "dob": dob, "sex": sex,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }])
    patients = pd.concat([patients, row], ignore_index=True)
    patients.to_csv(PATIENTS_CSV, index=False)
    return patient_id


def get_patient_name(patient_id: str) -> str | None:
    patients = list_patients()
    match = patients[patients["patient_id"] == patient_id]
    return None if match.empty else str(match.iloc[0]["name"])


def get_patient_info(patient_id: str) -> dict | None:
    """Returns {name, dob, sex, created_at} for a patient, or None if unknown. dob/sex may
    be None for patients registered before this field existed (see ensure_store migration)."""
    patients = list_patients()
    match = patients[patients["patient_id"] == patient_id]
    if match.empty:
        return None
    row = match.iloc[0]
    dob = row.get("dob")
    sex = row.get("sex")
    return {
        "name": str(row["name"]),
        "dob": None if pd.isna(dob) else str(dob),
        "sex": None if pd.isna(sex) else str(sex),
        "created_at": str(row["created_at"]),
    }


def update_patient_dob_sex(patient_id: str, dob: str, sex: str) -> None:
    """Backfills dob/sex for a patient registered before these fields existed."""
    ensure_store()
    patients = pd.read_csv(PATIENTS_CSV)
    mask = patients["patient_id"] == patient_id
    patients.loc[mask, "dob"] = dob
    patients.loc[mask, "sex"] = sex
    patients.to_csv(PATIENTS_CSV, index=False)


def find_patients_by_name(name: str) -> pd.DataFrame:
    """Case-insensitive, whitespace-trimmed name match against existing patients -- used by
    Register to catch the same person being registered twice under separate PT- ids."""
    patients = list_patients()
    if patients.empty:
        return patients
    normalized = name.strip().lower()
    return patients[patients["name"].astype(str).str.strip().str.lower() == normalized]


# --- Reports / saved cases ----------------------------------------------------

def _next_case_id() -> str:
    ensure_store()
    index = pd.read_csv(REPORTS_INDEX_CSV)
    return f"C-{len(index) + 1:04d}"


def save_case(patient_id: str, case: dict) -> str:
    """case: arbitrary JSON-serializable dict (inputs summary, per-branch scores, fused
    score, band, SHAP, etc.) -- written verbatim to reports/<case_id>.json. A flattened
    summary row is appended to the reports_index.csv the Reports page reads."""
    ensure_store()
    case_id = _next_case_id()
    created_at = datetime.now().isoformat(timespec="seconds")

    record = {"case_id": case_id, "patient_id": patient_id, "created_at": created_at, **case}
    (REPORTS_DIR / f"{case_id}.json").write_text(json.dumps(record, indent=2, default=str))

    index = pd.read_csv(REPORTS_INDEX_CSV)
    summary_row = pd.DataFrame([{
        "case_id": case_id,
        "patient_id": patient_id,
        "patient_name": get_patient_name(patient_id) or "",
        "created_at": created_at,
        "modalities": case.get("modalities", ""),
        "fused_score": case.get("fused_score"),
        "risk_band": case.get("risk_band", ""),
        "mode": case.get("mode", ""),
    }])
    index = pd.concat([index, summary_row], ignore_index=True)
    index.to_csv(REPORTS_INDEX_CSV, index=False)
    return case_id


def list_cases() -> pd.DataFrame:
    ensure_store()
    return pd.read_csv(REPORTS_INDEX_CSV)


def load_case(case_id: str) -> dict:
    path = REPORTS_DIR / f"{case_id}.json"
    return json.loads(path.read_text())
