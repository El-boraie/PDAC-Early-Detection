import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import inject_css, research_use_badge, caveat_note
from storage import register_patient, find_patients_by_name

inject_css()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("PancraDX")
    st.caption("Register a patient to begin")
with top_right:
    research_use_badge()

st.divider()

st.subheader("Register")
st.caption("Demo data -- not real patients. Age and sex are collected once here and reused "
           "automatically on Predict -- no need to re-enter them for the same patient.")

PREDICT_STATE_KEYS = [
    "predict_result", "uploaded_ct_key", "uploaded_ct_volume", "last_saved_case_id",
    "patient_features_cache",
]


def _start_case(patient_id: str, patient_name: str) -> None:
    # A fresh patient (or a fresh visit for an existing one) should never show the
    # previous patient's uploaded scan / results -- clear Predict's working state.
    for key in PREDICT_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["current_patient_id"] = patient_id
    st.session_state["current_patient_name"] = patient_name
    st.session_state.pop("pending_name", None)
    st.session_state.pop("pending_dob", None)
    st.session_state.pop("pending_sex", None)
    st.switch_page("pages/predict.py")


with st.form("register_form", clear_on_submit=True):
    name = st.text_input("Patient name")
    c1, c2 = st.columns(2)
    with c1:
        dob = st.date_input("Date of birth", value=date(1970, 1, 1),
                             min_value=date(1900, 1, 1), max_value=date.today())
    with c2:
        sex_label = st.selectbox("Sex", ["Female", "Male"])
    submitted = st.form_submit_button("Register & start", type="primary")

if submitted:
    if not name.strip():
        st.error("Enter a name to register.")
    else:
        st.session_state["pending_name"] = name.strip()
        st.session_state["pending_dob"] = dob.isoformat()
        st.session_state["pending_sex"] = sex_label

pending_name = st.session_state.get("pending_name")
if pending_name:
    pending_dob = st.session_state.get("pending_dob")
    pending_sex = st.session_state.get("pending_sex")
    matches = find_patients_by_name(pending_name)
    if matches.empty:
        patient_id = register_patient(pending_name, dob=pending_dob, sex=pending_sex)
        st.success(f"Registered **{pending_name}** as `{patient_id}`. Taking you to Predict...")
        _start_case(patient_id, pending_name)
    else:
        caveat_note(
            f"<b>A patient named '{pending_name}' already exists.</b> Choose whether this is "
            "the same person (continue their existing record) or a different person who "
            "happens to share a name (register as a new patient)."
        )
        st.dataframe(matches[["patient_id", "name", "dob", "sex", "created_at"]], hide_index=True, width="stretch")

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Same person**")
            existing_id = st.selectbox("Existing record", matches["patient_id"].tolist(), label_visibility="collapsed")
            if st.button("Continue with this existing patient"):
                existing_name = matches[matches["patient_id"] == existing_id].iloc[0]["name"]
                _start_case(existing_id, str(existing_name))
        with c2:
            st.markdown("**Different person, same name**")
            if st.button("Register as a new patient anyway"):
                patient_id = register_patient(pending_name, dob=pending_dob, sex=pending_sex)
                _start_case(patient_id, pending_name)
