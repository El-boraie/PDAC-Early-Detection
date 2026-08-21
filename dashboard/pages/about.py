import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    inject_css, research_use_badge, PRIMARY, PRIMARY_DEEP, ACCENT_HIGH_RISK,
    get_sample_tabular_df, load_manifest_cache_df, get_clinical_model_card, get_imaging_model_card,
    get_fusion_model_card,
)

inject_css()

top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("About")
with top_right:
    research_use_badge()

st.caption(
    "A research tool that estimates early pancreatic-cancer risk from two independent signals "
    "-- a CT scan and a urinary biomarker panel -- and shows its reasoning on every result."
)
st.divider()

# =============================================================================
# Architecture
# =============================================================================
st.subheader("System architecture")
st.caption("Two branches run independently, each is calibrated to output honest probabilities, "
           "then a fixed rule blends them into one score.")

ARCH_SVG = f"""
<svg viewBox="0 0 720 300" style="width:100%;height:auto;">
  <defs></defs>
  <rect x="20" y="40" width="150" height="56" rx="12" fill="#F0F8F8" stroke="#C4E0E2"/>
  <text x="95" y="64" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">CT scan</text>
  <text x="95" y="80" text-anchor="middle" font-size="10" fill="#5F7278">slice or full volume</text>

  <rect x="20" y="204" width="150" height="56" rx="12" fill="#F0F8F8" stroke="#C4E0E2"/>
  <text x="95" y="228" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Urinary biomarkers</text>
  <text x="95" y="244" text-anchor="middle" font-size="10" fill="#5F7278">7 features</text>

  <rect x="230" y="40" width="160" height="56" rx="12" fill="#FFFFFF" stroke="#E0EAEC"/>
  <text x="310" y="64" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Imaging model</text>
  <text x="310" y="80" text-anchor="middle" font-size="10" fill="#5F7278">ResNet-50 U-Net</text>

  <rect x="230" y="204" width="160" height="56" rx="12" fill="#FFFFFF" stroke="#E0EAEC"/>
  <text x="310" y="228" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Clinical model</text>
  <text x="310" y="244" text-anchor="middle" font-size="10" fill="#5F7278">XGBoost</text>

  <rect x="430" y="40" width="120" height="56" rx="12" fill="#FFFFFF" stroke="#E0EAEC"/>
  <text x="490" y="64" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Calibrate</text>
  <text x="490" y="80" text-anchor="middle" font-size="10" fill="#5F7278">honest probs</text>

  <rect x="430" y="204" width="120" height="56" rx="12" fill="#FFFFFF" stroke="#E0EAEC"/>
  <text x="490" y="228" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Calibrate</text>
  <text x="490" y="244" text-anchor="middle" font-size="10" fill="#5F7278">honest probs</text>

  <rect x="430" y="122" width="120" height="56" rx="12" fill="#EAF3F4" stroke="{PRIMARY}"/>
  <text x="490" y="146" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Fusion rule</text>
  <text x="490" y="162" text-anchor="middle" font-size="10" fill="#5F7278">0.4 img &#183; 0.6 clin</text>

  <rect x="600" y="122" width="100" height="56" rx="12" fill="#FCEEEC" stroke="#F1CFCC"/>
  <text x="650" y="146" text-anchor="middle" font-size="12" font-weight="600" fill="{PRIMARY_DEEP}">Combined</text>
  <text x="650" y="162" text-anchor="middle" font-size="10" fill="#5F7278">risk score</text>

  <path d="M170 68 H230" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M170 232 H230" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M390 68 H430" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M390 232 H430" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M550 68 C575 68 555 150 550 150" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M550 232 C575 232 555 150 550 150" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
  <path d="M550 150 H600" stroke="#9FB4B8" stroke-width="1.6" fill="none"/>
</svg>
"""
st.markdown(ARCH_SVG, unsafe_allow_html=True)
st.caption(
    "Either input works alone -- with only one branch, the system shows that branch's score "
    "and labels it as such, never a fused number."
)

# =============================================================================
# How it works
# =============================================================================
st.subheader("How a case is assessed")
st.write(
    "A patient is registered by name and issued an ID. On the Predict screen you provide a CT "
    "scan, a biomarker row, or both. Each available branch produces a calibrated probability; "
    "Grad-CAM shows where the imaging model looked, and SHAP shows which biomarkers moved the "
    "clinical score. If both branches ran, the fixed 0.4/0.6 rule combines them. The result -- "
    "with its explanations -- can be saved and exported as a PDF report."
)

st.divider()

# =============================================================================
# Intended use
# =============================================================================
st.subheader("Intended use")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Designed for**")
    st.markdown(
        "- Research and educational demonstration\n"
        "- Exploring multimodal risk-model behaviour\n"
        "- Showing explainability and calibration in practice\n"
        "- Use by people who understand its limits"
    )
with c2:
    st.markdown("**Never use for**")
    st.markdown(
        "- Real diagnosis or clinical decisions\n"
        "- Screening actual patients\n"
        "- Replacing a radiologist or clinician\n"
        "- Any setting where a person is affected by the output"
    )

st.divider()

# =============================================================================
# Data provenance
# =============================================================================
st.subheader("Data provenance & ethics")
manifest_df = load_manifest_cache_df()
tab_df = get_sample_tabular_df()
msd_n = manifest_df[manifest_df["dataset"] == "MSD"]["patient_id"].nunique()
nih_n = manifest_df[manifest_df["dataset"] == "NIH"]["patient_id"].nunique()

st.table({
    "Source": ["MSD (Task07)", "NIH Pancreas-CT", "Debernardi et al."],
    "Description": [
        f"{msd_n} pancreatic-cancer CT patients. Public research dataset.",
        f"{nih_n} healthy CT patients. Public research dataset.",
        f"{len(tab_df)} patients with urinary biomarkers. Published cohort.",
    ],
})
st.caption(
    "Patient names entered in the register are stored separately from all results and never "
    "leave the local machine. Every downstream record uses the generated ID. This tool "
    "processes no real patient data."
)

st.divider()

# =============================================================================
# Behind the scenes -- backend / model snapshot
# =============================================================================
st.subheader("Behind the scenes")
st.caption("A brief look at how each branch was built and how it performs, for context on where "
           "the numbers on this dashboard come from.")

clinical_card = get_clinical_model_card()
imaging_card = get_imaging_model_card()
fusion_card = get_fusion_model_card()

b1, b2 = st.columns(2)
with b1:
    st.markdown("**Imaging branch**")
    imaging_metrics = imaging_card["comparison_metrics_that_justified_this_model"][imaging_card["candidate"]]
    st.markdown(
        f"- Architecture: {imaging_card['architecture']}\n"
        f"- Trained on: {imaging_card['n_patients']} patients, {imaging_card['n_rows']:,} CT slices\n"
        f"- ROC-AUC: {imaging_metrics['roc_auc_mean']:.3f} &middot; Recall: {imaging_metrics['recall_mean']:.3f} "
        f"(5-fold CV)\n"
        f"- Calibration: {imaging_card['calibrator']}",
        unsafe_allow_html=True,
    )
with b2:
    st.markdown("**Clinical branch**")
    model_display_name = {"xgboost": "XGBoost"}.get(clinical_card["model"], clinical_card["model"])
    clinical_metrics = clinical_card["comparison_metrics_that_justified_this_model"]["Repeated 5x20 CV"][model_display_name]
    st.markdown(
        f"- Architecture: {model_display_name}\n"
        f"- Trained on: {clinical_card['n_patients']} patients ({clinical_card['n_pdac']} PDAC)\n"
        f"- ROC-AUC: {clinical_metrics['auc_mean']:.3f} &middot; Recall: {clinical_metrics['recall_mean']:.3f} "
        f"(repeated 5x20 CV)\n"
        f"- Calibration: {clinical_card['calibrator']}",
        unsafe_allow_html=True,
    )

weights = fusion_card["combination_rule"]["weights"]
st.markdown(
    f"**Fusion rule:** fused = {weights['W_IMAGING']} &times; imaging + {weights['W_TABULAR']} &times; clinical "
    "-- a fixed, hand-set rule rather than a trained model, since no patient in the data has both "
    "a CT scan and a urine sample to fit or validate a blend against.",
    unsafe_allow_html=True,
)

st.divider()

# =============================================================================
# Glossary
# =============================================================================
st.subheader("Glossary")
g1, g2 = st.columns(2)
with g1:
    st.markdown("**PDAC**")
    st.caption("Pancreatic ductal adenocarcinoma -- the most common pancreatic cancer.")
    st.markdown("**Grad-CAM**")
    st.caption("A heatmap showing which parts of an image a model responded to.")
with g2:
    st.markdown("**SHAP**")
    st.caption("A method attributing a prediction to each input feature's contribution.")
    st.markdown("**Calibration**")
    st.caption("Adjusting scores so a stated probability matches real-world frequency.")

st.divider()
st.caption("PancraDX -- research build -- models and datasets versioned in the project repository. "
           "Not a medical device.")
