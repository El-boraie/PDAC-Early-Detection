import base64
import sys
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    inf, classify_risk_band, inject_css, read_box, risk_pill, research_use_badge,
    get_clinical_branch, get_imaging_branch, get_fusion_model_card, render_gradcam_overlay,
    score_ring_svg, branch_row, shap_bar_row, BIOMARKER_INFO, PRIMARY, ACCENT_HIGH_RISK,
)
from storage import save_case, get_patient_info, update_patient_dob_sex, compute_age
from report_pdf import generate_case_pdf

inject_css()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("Predict")
with top_right:
    research_use_badge()

patient_id = st.session_state.get("current_patient_id")
patient_name = st.session_state.get("current_patient_name")

if not patient_id:
    st.info("No patient selected. Go to Register to start a new case.")
    if st.button("Go to Register"):
        st.switch_page("pages/register.py")
    st.stop()

# Belt-and-suspenders patient-switch reset: Register already clears this on its own route
# into Predict, but this catches switching patients via direct sidebar navigation too --
# without it, a new patient would silently see the previous patient's uploaded scan/result.
PREDICT_STATE_KEYS = ["predict_result", "uploaded_ct_key", "uploaded_ct_volume", "last_saved_case_id",
                      "pending_pdf_bytes", "pending_pdf_name"]
if st.session_state.get("_predict_last_patient_id") != patient_id:
    for key in PREDICT_STATE_KEYS:
        st.session_state.pop(key, None)
    st.session_state["_predict_last_patient_id"] = patient_id

st.caption(f"Patient: **{patient_name}** ({patient_id})")

# =============================================================================
# Age / sex -- collected once at Register, never asked again for the same patient
# =============================================================================
patient_info = get_patient_info(patient_id)
if not patient_info.get("dob") or not patient_info.get("sex"):
    st.warning(f"{patient_name}'s date of birth / sex weren't recorded at registration "
               "(an older record). Enter them once here -- they'll be remembered from now on.")
    with st.form("backfill_dob_sex"):
        b1, b2 = st.columns(2)
        with b1:
            backfill_dob = st.date_input("Date of birth", value=None, min_value=date(1900, 1, 1),
                                          max_value=date.today())
        with b2:
            backfill_sex = st.selectbox("Sex", ["Female", "Male"])
        if st.form_submit_button("Save", type="primary"):
            if backfill_dob is None:
                st.error("Enter a date of birth.")
            else:
                update_patient_dob_sex(patient_id, backfill_dob.isoformat(), backfill_sex)
                st.rerun()
    st.stop()

patient_age = compute_age(patient_info["dob"])
patient_sex = patient_info["sex"]
st.caption(f"Age {patient_age} · {patient_sex}")
st.divider()

# =============================================================================
# CT scan input
# =============================================================================
imaging_tensor = None          # (3,BOX,BOX) or (N,3,BOX,BOX) fed to fuse()
imaging_display_slice = None   # (3,BOX,BOX) single slice tensor used for Grad-CAM display
imaging_source_label = None

with st.container(border=True):
    st.markdown(
        "<h3>CT scan</h3>"
        '<div class="lede">Upload a raw CT volume (.nii.gz). Optional -- provide a scan, biomarkers, or both.</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader("Upload a .nii.gz CT volume", type=["gz", "nii"], key=f"ct_uploader_{patient_id}")
    if uploaded is not None:
        cache_key = f"{uploaded.name}:{uploaded.size}"
        if st.session_state.get("uploaded_ct_key") != cache_key:
            with st.spinner("Preprocessing scan (reorient, resample, HU-window)..."):
                imaging_branch_for_load = get_imaging_branch()
                volume = inf.nifti_upload_to_volume_tensor(uploaded.getvalue(), imaging_branch_for_load.box_size)
            st.session_state["uploaded_ct_key"] = cache_key
            st.session_state["uploaded_ct_volume"] = volume
        volume = st.session_state["uploaded_ct_volume"]
        n_slices = volume.shape[0]
        imaging_source_label = uploaded.name
        st.success(f"Processed {n_slices} slices.")

        granularity = st.radio("Granularity", ["Full volume (mean over slices)", "Single slice"],
                                horizontal=True, key=f"ct_granularity_{patient_id}")
        if granularity == "Full volume (mean over slices)":
            imaging_tensor = volume
            imaging_display_slice = volume[n_slices // 2]
        else:
            slice_idx = st.slider("Slice", 0, n_slices - 1, n_slices // 2, key=f"ct_slice_{patient_id}")
            imaging_tensor = volume[slice_idx]
            imaging_display_slice = imaging_tensor

# =============================================================================
# Biomarker input
# =============================================================================
patient_features = None
tabular_source_label = None

with st.container(border=True):
    st.markdown(
        "<h3>Urinary biomarkers</h3>"
        f'<div class="lede">Optional -- provide a scan, biomarkers, or both. Age ({patient_age}) and sex '
        f'({patient_sex}) are already on file for {patient_name}. Hover the (?) on each field for what it means.</div>',
        unsafe_allow_html=True,
    )

    include_biomarkers = st.checkbox("Enter biomarkers for this case", key=f"include_biomarkers_{patient_id}")

    if include_biomarkers:
        tabular_source_label = "manual entry"
        c1, c2 = st.columns(2)
        with c1:
            creatinine = st.number_input(
                BIOMARKER_INFO["creatinine"]["label"], min_value=BIOMARKER_INFO["creatinine"]["min"],
                max_value=BIOMARKER_INFO["creatinine"]["max"], value=BIOMARKER_INFO["creatinine"]["default"],
                step=BIOMARKER_INFO["creatinine"]["step"], help=BIOMARKER_INFO["creatinine"]["help"],
                key=f"creatinine_{patient_id}",
            )
            lyve1 = st.number_input(
                BIOMARKER_INFO["LYVE1"]["label"], min_value=BIOMARKER_INFO["LYVE1"]["min"],
                max_value=BIOMARKER_INFO["LYVE1"]["max"], value=BIOMARKER_INFO["LYVE1"]["default"],
                step=BIOMARKER_INFO["LYVE1"]["step"], help=BIOMARKER_INFO["LYVE1"]["help"],
                key=f"lyve1_{patient_id}",
            )
            reg1b = st.number_input(
                BIOMARKER_INFO["REG1B"]["label"], min_value=BIOMARKER_INFO["REG1B"]["min"],
                max_value=BIOMARKER_INFO["REG1B"]["max"], value=BIOMARKER_INFO["REG1B"]["default"],
                step=BIOMARKER_INFO["REG1B"]["step"], help=BIOMARKER_INFO["REG1B"]["help"],
                key=f"reg1b_{patient_id}",
            )
        with c2:
            tff1 = st.number_input(
                BIOMARKER_INFO["TFF1"]["label"], min_value=BIOMARKER_INFO["TFF1"]["min"],
                max_value=BIOMARKER_INFO["TFF1"]["max"], value=BIOMARKER_INFO["TFF1"]["default"],
                step=BIOMARKER_INFO["TFF1"]["step"], help=BIOMARKER_INFO["TFF1"]["help"],
                key=f"tff1_{patient_id}",
            )
            ca19_9 = st.number_input(
                BIOMARKER_INFO["plasma_CA19_9"]["label"], min_value=BIOMARKER_INFO["plasma_CA19_9"]["min"],
                max_value=BIOMARKER_INFO["plasma_CA19_9"]["max"], value=BIOMARKER_INFO["plasma_CA19_9"]["default"],
                step=BIOMARKER_INFO["plasma_CA19_9"]["step"], help=BIOMARKER_INFO["plasma_CA19_9"]["help"],
                key=f"ca199_{patient_id}",
            )

        patient_features = {
            "creatinine": creatinine, "LYVE1": lyve1, "REG1B": reg1b, "TFF1": tff1,
            "plasma_CA19_9": ca19_9, "age": patient_age, "sex": 1 if patient_sex == "Male" else 0,
        }

# =============================================================================
# Run
# =============================================================================
can_run = imaging_tensor is not None or patient_features is not None
if st.button("Run assessment", type="primary", disabled=not can_run):
    with st.spinner("Scoring..."):
        clinical_branch = get_clinical_branch() if patient_features is not None else None
        imaging_branch = get_imaging_branch() if imaging_tensor is not None else None
        result = inf.fuse(clinical=clinical_branch, imaging=imaging_branch,
                           imaging_input=imaging_tensor, tabular_input=patient_features)

        gradcam_png = None
        if imaging_branch is not None and imaging_display_slice is not None:
            heatmap = inf.generate_gradcam(imaging_branch, imaging_display_slice)
            gradcam_png = render_gradcam_overlay(imaging_display_slice, heatmap)

        per_slice = None
        if imaging_tensor is not None and imaging_tensor.ndim == 4:
            per_slice = inf._score_slices(imaging_branch, imaging_tensor).tolist()

        shap_per_feature, base_value = (None, None)
        if clinical_branch is not None:
            row_df = pd.DataFrame([patient_features])[inf.TABULAR_FEATURES]
            shap_per_feature, base_value = inf.explain_tabular(clinical_branch, row_df)

        modalities = []
        if imaging_tensor is not None:
            modalities.append("Imaging")
        if patient_features is not None:
            modalities.append("Clinical")

        st.session_state["predict_result"] = {
            "result": result,
            "gradcam_png": gradcam_png,
            "per_slice": per_slice,
            "shap_per_feature": {k: float(v) for k, v in shap_per_feature.items()} if shap_per_feature else None,
            "base_value": base_value,
            "patient_features": patient_features,
            "modalities": " + ".join(modalities),
            "imaging_source_label": imaging_source_label,
            "tabular_source_label": tabular_source_label,
        }

# =============================================================================
# Results
# =============================================================================
pred = st.session_state.get("predict_result")
if pred:
    st.divider()
    result = pred["result"]
    fused_score = result["fused_score"]
    band = classify_risk_band(fused_score)
    band_words = {"Low": "low risk", "Moderate": "moderate risk", "High": "elevated risk"}

    # --- Hero verdict card ---
    with st.container(border=True):
        if result["mode"] == "fused (both modalities)":
            headline = f"Findings point to {band_words[band]} of early PDAC."
            subtext = ("The combined score blends both branches using a fixed rule -- "
                       "0.4 &times; imaging + 0.6 &times; clinical -- reasoned, not fitted, "
                       "since no patient in the training data has both a scan and a urine sample.")
            eyebrow = "Combined assessment"
        else:
            which = "Imaging" if result["imaging_calibrated_proba"] is not None else "Clinical"
            headline = f"{which} assessment points to {band_words[band]}."
            subtext = f"Only the {which.lower()} branch was provided -- this is that branch's own score, not a combined score."
            eyebrow = "Single-branch assessment"

        # Built as one unindented string -- st.markdown's renderer treats 4+ leading spaces
        # on a line as a Markdown code block, which silently breaks HTML parsing partway
        # through an indented multi-line f-string (confirmed: it leaked a literal '</div>').
        hero_html = (
            f'<div class="pdx-hero">'
            f'<div class="text">'
            f'<span class="eyebrow">{eyebrow}</span>'
            f'<h1>{headline}</h1>'
            f'<p>{subtext}</p>'
            f'</div>'
            f'{score_ring_svg(fused_score, band)}'
            f'</div>'
        )
        st.markdown(hero_html, unsafe_allow_html=True)

    # --- Branch contribution card ---
    if result["mode"] == "fused (both modalities)":
        with st.container(border=True):
            rows_html = (
                branch_row("Imaging", "CT · ResNet-50 U-Net", result["imaging_calibrated_proba"], "#8FBDC4")
                + branch_row("Clinical", "Urine · XGBoost", result["tabular_calibrated_proba"], PRIMARY)
                + f'<div class="pdx-weightline">Blended as <b>0.40 &times; {result["imaging_calibrated_proba"]:.2f} '
                  f'+ 0.60 &times; {result["tabular_calibrated_proba"]:.2f} = {fused_score:.2f}</b>.</div>'
            )
            st.markdown(
                "<h3>How the two branches contributed</h3>"
                '<div class="lede">Each branch is scored separately, then blended.</div>'
                + rows_html,
                unsafe_allow_html=True,
            )

    # --- Imaging panel ---
    if result["imaging_calibrated_proba"] is not None:
        with st.container(border=True):
            st.markdown(
                "<h3>Imaging</h3>"
                f'<div class="lede">{result["imaging_granularity"]}, {result["imaging_n_slices"]} slice(s) '
                f'-- {pred["imaging_source_label"]}</div>',
                unsafe_allow_html=True,
            )
            i1, i2 = st.columns([1, 1])
            with i1:
                if pred["gradcam_png"]:
                    st.image(pred["gradcam_png"], caption="Grad-CAM overlay (where the model looked)", width="stretch")
            with i2:
                if pred["per_slice"]:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(y=pred["per_slice"], mode="lines", line=dict(color=PRIMARY)))
                    fig.update_layout(
                        title="Per-slice calibrated probability", xaxis_title="Slice index",
                        yaxis_title="P(PDAC)", yaxis_range=[0, 1], height=300, margin=dict(t=40, b=20),
                    )
                    st.plotly_chart(fig, width="stretch")

            confidence_word = "high" if result["imaging_calibrated_proba"] >= 0.7 else (
                "moderate" if result["imaging_calibrated_proba"] >= 0.3 else "low")
            pattern_word = ""
            if pred["per_slice"]:
                spread = float(np.std(pred["per_slice"]))
                pattern_word = " with a fairly even response across the whole volume" if spread < 0.1 else \
                    " with the response varying noticeably across slices"
            read_box(
                f"The model reads this input as <b>{confidence_word} confidence</b> "
                f"(score {result['imaging_calibrated_proba']:.2f}){pattern_word}."
            )

    # --- Clinical panel ---
    if result["tabular_calibrated_proba"] is not None:
        with st.container(border=True):
            st.markdown(
                "<h3>Clinical markers</h3>"
                '<div class="lede">Which biomarkers moved this patient\'s score, and by how much '
                f'-- {pred["tabular_source_label"]}.</div>',
                unsafe_allow_html=True,
            )
            shap_items = sorted(pred["shap_per_feature"].items(), key=lambda kv: -abs(kv[1]))
            max_abs = max(abs(v) for _, v in shap_items) if shap_items else 1.0
            shap_html = "".join(shap_bar_row(feat, val, max_abs) for feat, val in shap_items)
            st.markdown(shap_html, unsafe_allow_html=True)

            top_feat, top_val = shap_items[0]
            direction = "pushing the score higher" if top_val >= 0 else "pushing the score lower"
            read_box(f"<b>{top_feat}</b> is the strongest driver here, {direction}.")

    # --- Actions ---
    # Shared by both buttons below -- a PDF's Case ID must always reference a real, saved
    # case (never a literal "(unsaved)" placeholder), so "Generate report" saves the case
    # itself first when it hasn't been saved yet, using this exact same record.
    case = {
        "patient_name": patient_name,
        "modalities": pred["modalities"],
        "mode": result["mode"],
        "fused_score": fused_score,
        "risk_band": band,
        "imaging": ({
            "calibrated_proba": result["imaging_calibrated_proba"],
            "granularity": result["imaging_granularity"],
            "n_slices": result["imaging_n_slices"],
            "per_slice_proba": pred["per_slice"],
            # Stored so a PDF regenerated later from Reports (no live session
            # state, no original scan on disk) can still show the same overlay.
            "gradcam_png_b64": (base64.b64encode(pred["gradcam_png"]).decode("ascii")
                                if pred["gradcam_png"] else None),
        } if result["imaging_calibrated_proba"] is not None else None),
        "clinical": ({
            "calibrated_proba": result["tabular_calibrated_proba"],
            "shap_per_feature": pred["shap_per_feature"],
            "base_value": pred["base_value"],
            "raw_features": pred["patient_features"],
        } if result["tabular_calibrated_proba"] is not None else None),
    }

    with st.container(border=True):
        a1, a2 = st.columns(2)
        with a1:
            if st.button("Save case"):
                case_id = save_case(patient_id, case)
                st.session_state["last_saved_case_id"] = case_id
                st.success(f"Saved as {case_id}.")
        with a2:
            if st.button("Generate report (PDF)"):
                case_id_for_pdf = st.session_state.get("last_saved_case_id")
                if case_id_for_pdf is None:
                    case_id_for_pdf = save_case(patient_id, case)
                    st.session_state["last_saved_case_id"] = case_id_for_pdf
                pdf_case = {
                    "case_id": case_id_for_pdf, "patient_id": patient_id,
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    **case,
                }
                st.session_state["pending_pdf_bytes"] = generate_case_pdf(
                    pdf_case, gradcam_png_bytes=pred["gradcam_png"])
                st.session_state["pending_pdf_name"] = f"{case_id_for_pdf}_report.pdf"
            if st.session_state.get("pending_pdf_bytes"):
                st.download_button("Download report (PDF)", data=st.session_state["pending_pdf_bytes"],
                                    file_name=st.session_state["pending_pdf_name"], mime="application/pdf")
        st.caption("Research and educational use only. Not a diagnostic device and not a substitute for clinical judgement.")
