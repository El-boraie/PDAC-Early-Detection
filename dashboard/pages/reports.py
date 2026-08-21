import base64
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import inject_css, research_use_badge, risk_pill, PRIMARY, RISK_BAND_COLORS
from storage import list_cases, load_case, get_patient_name
from report_pdf import generate_case_pdf


def _case_pdf_bytes(case_id: str, patient_name_fallback: str) -> bytes:
    """Builds a case dict fit for generate_case_pdf() from a saved case JSON, which (unlike
    Predict's own live in-session PDF path) has no session state to draw on -- patient_name
    and the Grad-CAM image have to come from the saved record itself, with a fallback for
    cases saved before those fields existed."""
    case = load_case(case_id)
    case.setdefault("patient_name", patient_name_fallback or get_patient_name(case["patient_id"]) or "")
    gradcam_png_bytes = None
    imaging = case.get("imaging")
    if imaging and imaging.get("gradcam_png_b64"):
        gradcam_png_bytes = base64.b64decode(imaging["gradcam_png_b64"])
    return generate_case_pdf(case, gradcam_png_bytes=gradcam_png_bytes)

inject_css()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("Reports")
with top_right:
    research_use_badge()

st.divider()

cases = list_cases()

if cases.empty:
    st.info("No cases yet -- run one from Predict.")
    if st.button("Go to Predict"):
        st.switch_page("pages/predict.py")
    st.stop()

# =============================================================================
# Filter / sort rail
# =============================================================================
rail, main = st.columns([1, 3])

with rail:
    st.subheader("Filter & sort")
    search = st.text_input("Search by name or ID")

    all_bands = sorted(cases["risk_band"].dropna().unique().tolist())
    band_filter = st.multiselect("Risk band", all_bands, default=all_bands)

    all_modalities = sorted(cases["modalities"].dropna().unique().tolist())
    modality_filter = st.multiselect("Modalities", all_modalities, default=all_modalities)

    sort_choice = st.selectbox("Sort by", ["Date (newest)", "Score (high to low)", "Name (A-Z)"])

filtered = cases.copy()
if search.strip():
    q = search.strip().lower()
    filtered = filtered[
        filtered["patient_name"].astype(str).str.lower().str.contains(q)
        | filtered["patient_id"].astype(str).str.lower().str.contains(q)
    ]
filtered = filtered[filtered["risk_band"].isin(band_filter) & filtered["modalities"].isin(modality_filter)]

if sort_choice == "Date (newest)":
    filtered = filtered.sort_values("created_at", ascending=False)
elif sort_choice == "Score (high to low)":
    filtered = filtered.sort_values("fused_score", ascending=False)
else:
    filtered = filtered.sort_values("patient_name", ascending=True)

# =============================================================================
# Main table
# =============================================================================
with main:
    st.subheader(f"Cases ({len(filtered)} of {len(cases)})")

    if filtered.empty:
        st.info("No cases match the current filters.")
    else:
        header = st.columns([1.2, 2, 1.4, 1.6, 1, 1, 1])
        for col, label in zip(header, ["Case ID", "Patient", "Date", "Modalities", "Score", "Risk", "Report"]):
            col.markdown(f"**{label}**")

        for _, row in filtered.iterrows():
            c1, c2, c3, c4, c5, c6, c7 = st.columns([1.2, 2, 1.4, 1.6, 1, 1, 1])
            c1.write(row["case_id"])
            c2.write(f"{row['patient_name']} ({row['patient_id']})")
            c3.write(str(row["created_at"])[:16])
            c4.write(row["modalities"])
            c5.write(f"{row['fused_score']:.3f}" if pd.notna(row["fused_score"]) else "--")
            c6.markdown(risk_pill(row["risk_band"]), unsafe_allow_html=True)
            with c7:
                pdf_bytes = _case_pdf_bytes(row["case_id"], row["patient_name"])
                st.download_button("PDF", data=pdf_bytes, file_name=f"{row['case_id']}.pdf",
                                    mime="application/pdf", key=f"pdf_{row['case_id']}")

# =============================================================================
# Compare cases for a patient
# =============================================================================
st.divider()
st.subheader("Compare cases for a patient")

case_counts = cases.groupby("patient_id").size()
comparable_patients = case_counts[case_counts >= 2].index.tolist()

if not comparable_patients:
    st.info("Compare needs at least 2 recorded cases for the same patient -- run another "
            "assessment for an existing patient from Predict to unlock this.")
else:
    labels = {
        pid: f"{cases[cases['patient_id'] == pid]['patient_name'].iloc[0]} ({pid}) -- "
             f"{case_counts[pid]} cases"
        for pid in comparable_patients
    }
    chosen_pid = st.selectbox("Patient", comparable_patients, format_func=lambda pid: labels[pid])

    patient_cases = cases[cases["patient_id"] == chosen_pid].sort_values("created_at")
    case_ids = st.multiselect("Cases to compare", patient_cases["case_id"].tolist(),
                               default=patient_cases["case_id"].tolist())

    if len(case_ids) < 2:
        st.info("Select at least 2 cases to compare.")
    else:
        compare_df = patient_cases[patient_cases["case_id"].isin(case_ids)].sort_values("created_at")

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("**Score over time**")
            fig = go.Figure(go.Scatter(
                x=compare_df["created_at"], y=compare_df["fused_score"], mode="lines+markers",
                line=dict(color=PRIMARY),
                marker=dict(size=10, color=[RISK_BAND_COLORS.get(b, PRIMARY) for b in compare_df["risk_band"]]),
            ))
            fig.update_layout(xaxis_title="Date", yaxis_title="Score", yaxis_range=[0, 1],
                               height=320, margin=dict(t=20, b=20))
            fig.update_xaxes(type="category")
            st.plotly_chart(fig, width="stretch")
        with c2:
            st.markdown("**Case summary**")
            st.dataframe(
                compare_df[["case_id", "created_at", "modalities", "fused_score", "risk_band"]],
                hide_index=True, width="stretch",
            )

        # --- SHAP comparison across cases, when 2+ have clinical data ---
        loaded = {cid: load_case(cid) for cid in case_ids}
        clinical_cases = {cid: c["clinical"] for cid, c in loaded.items() if c.get("clinical")}
        if len(clinical_cases) >= 2:
            st.markdown("**Biomarker contribution (SHAP) across cases**")
            all_features = sorted({f for c in clinical_cases.values() for f in c["shap_per_feature"]})
            fig = go.Figure()
            for cid, clinical in clinical_cases.items():
                fig.add_trace(go.Bar(
                    name=cid, x=all_features,
                    y=[clinical["shap_per_feature"].get(f, 0) for f in all_features],
                ))
            fig.update_layout(barmode="group", xaxis_title="Feature", yaxis_title="SHAP value (log-odds)",
                               height=360, margin=dict(t=20, b=20))
            st.plotly_chart(fig, width="stretch")
        elif len(clinical_cases) == 1:
            st.caption("Only one selected case has clinical (SHAP) data -- need 2+ to compare biomarker contributions.")
