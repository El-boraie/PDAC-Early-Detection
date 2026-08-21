"""Shared, cached resources and small UI helpers used across every dashboard page.

Custom CSS here covers the card/hero/ring/read components used to give Predict a more
considered, less "empty" look (closer to pdac_dashboard_mockup_v2.html) than bare
st.* components alone provide, per direct user feedback on the first build.
"""

import math
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import inference as inf
from storage import classify_risk_band

# --- Palette (locked, from the handoff) ---------------------------------------
PRIMARY = "#17879B"
PRIMARY_DEEP = "#0F5F6E"
ACCENT_HIGH_RISK = "#E5564C"
LOW_RISK = "#4FB08A"
AMBER_CAVEAT = "#C98A2E"
INK = "#1B2C31"
INK_SECONDARY = "#5F7278"
HAIRLINE = "#E0EAEC"

RISK_BAND_COLORS = {"Low": LOW_RISK, "Moderate": AMBER_CAVEAT, "High": ACCENT_HIGH_RISK}

# --- Biomarker reference info --------------------------------------------------
# CA19-9's <37 U/mL cutoff is a well-established clinical reference (standard oncology
# tumour-marker literature). Creatinine's ~0.6-1.3 mg/dL is standard clinical chemistry.
# LYVE1 / REG1B / TFF1 are NOT routine clinical lab tests -- they are research urinary
# biomarkers specific to pancreatic-cancer studies (Debernardi et al. 2020 and related
# work), with no standardized clinical reference range published anywhere. For those
# three, min/max/help are grounded in this project's own real cohort
# (data/processed/tabular_clean.csv describe()), not an invented "normal range" --
# labelled as such below rather than silently presented as if they were routine labs.
BIOMARKER_INFO = {
    "creatinine": {
        "label": "Creatinine (mg/dL)",
        "help": "Waste product filtered by the kidneys; used here as a general marker of "
                "renal function. Standard clinical reference range: ~0.6-1.3 mg/dL. Values "
                "outside that range can occur with reduced kidney function.",
        "min": 0.0, "max": 10.0, "default": 0.72, "step": 0.01,
    },
    "LYVE1": {
        "label": "LYVE1 (ng/mL)",
        "help": "Lymphatic vessel endothelial hyaluronan receptor 1 -- an experimental "
                "urinary biomarker studied for pancreatic cancer. Not a routine clinical lab "
                "test, so there is no standardized 'normal' reference range. The bounds shown "
                "reflect the range observed in this project's research cohort (n=590).",
        "min": 0.0, "max": 30.0, "default": 1.65, "step": 0.01,
    },
    "REG1B": {
        "label": "REG1B (ng/mL)",
        "help": "Regenerating protein 1 beta -- an experimental urinary biomarker linked to "
                "pancreatic tissue regeneration/pathology. Not a routine clinical lab test, so "
                "there is no standardized reference range. The bounds shown reflect the range "
                "observed in this project's research cohort (n=590).",
        "min": 0.0, "max": 1500.0, "default": 34.3, "step": 1.0,
    },
    "TFF1": {
        "label": "TFF1 (ng/mL)",
        "help": "Trefoil factor 1 -- an experimental urinary biomarker linked to mucosal "
                "protection, studied for pancreatic cancer. Not a routine clinical lab test, "
                "so there is no standardized reference range. The bounds shown reflect the "
                "range observed in this project's research cohort (n=590).",
        "min": 0.0, "max": 15000.0, "default": 259.9, "step": 1.0,
    },
    "plasma_CA19_9": {
        "label": "Plasma CA19-9 (U/mL)",
        "help": "Carbohydrate antigen 19-9 -- the standard clinical tumour marker used to "
                "monitor pancreatic cancer. Normal reference: <37 U/mL, though it is not "
                "specific or sensitive enough alone for diagnosis. Markedly elevated levels "
                "(hundreds to tens of thousands) are seen in advanced disease.",
        "min": 0.0, "max": 40000.0, "default": 26.5, "step": 1.0,
    },
    "age": {
        "label": "Age",
        "help": "Patient age in years.",
        "min": 18, "max": 100, "default": 60, "step": 1,
    },
}

_CSS = f"""
<style>
.pdx-note {{
    display: flex;
    gap: 0.6rem;
    background: #FBF4E9;
    border: 1px solid #EFDFC4;
    border-radius: 10px;
    padding: 0.85rem 1.05rem;
    margin: 0.6rem 0;
    font-size: 0.92rem;
    color: #5C4416;
}}
.pdx-note b {{ color: #4A3712; }}
.pdx-note .icon {{ flex: 0 0 auto; color: {AMBER_CAVEAT}; font-weight: 700; }}

.pdx-read {{
    font-size: 0.92rem;
    color: {INK};
    background: #F4F9F9;
    border-radius: 10px;
    padding: 0.75rem 0.95rem;
    margin: 0.5rem 0;
}}
.pdx-read b {{ color: {PRIMARY_DEEP}; }}

.pdx-pill {{
    display: inline-block;
    padding: 0.15rem 0.7rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: white;
}}

.pdx-research-badge {{
    display: inline-block;
    font-size: 0.72rem;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: {ACCENT_HIGH_RISK};
    border: 1px solid #F1CFCC;
    background: #FCEEEC;
    padding: 0.2rem 0.6rem;
    border-radius: 999px;
    font-weight: 600;
}}

.pdx-card {{
    background: #FFFFFF;
    border: 1px solid {HAIRLINE};
    border-radius: 18px;
    box-shadow: 0 1px 2px rgba(27,44,49,.04), 0 10px 30px rgba(27,44,49,.05);
    padding: 1.6rem 1.9rem;
    margin-bottom: 1rem;
}}
.pdx-card h3 {{
    font-size: 1.15rem;
    font-weight: 600;
    color: {INK};
    margin: 0 0 0.2rem 0;
}}
.pdx-card .lede {{
    font-size: 0.85rem;
    color: {INK_SECONDARY};
    margin-bottom: 1rem;
}}

.pdx-hero {{
    display: flex;
    align-items: center;
    gap: 2rem;
}}
.pdx-hero .text {{ flex: 1; }}
.pdx-hero .eyebrow {{
    font-size: 0.72rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: {PRIMARY};
    font-weight: 600;
    display: block;
    margin-bottom: 0.5rem;
}}
.pdx-hero h1 {{
    font-size: 1.7rem;
    font-weight: 600;
    line-height: 1.2;
    color: {INK};
    margin: 0 0 0.5rem 0;
}}
.pdx-hero p {{ font-size: 0.92rem; color: {INK_SECONDARY}; max-width: 46ch; }}

.pdx-branch-row {{
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 0.8rem 0;
    border-bottom: 1px solid {HAIRLINE};
}}
.pdx-branch-row:last-of-type {{ border-bottom: none; }}
.pdx-branch-row .nm {{ width: 110px; flex: 0 0 110px; }}
.pdx-branch-row .nm .t {{ font-weight: 600; font-size: 0.88rem; color: {INK}; }}
.pdx-branch-row .nm .s {{ font-size: 0.72rem; color: {INK_SECONDARY}; }}
.pdx-branch-row .bar-t {{ flex: 1; height: 10px; background: #EAF1F2; border-radius: 999px; overflow: hidden; }}
.pdx-branch-row .bar-t i {{ display: block; height: 100%; border-radius: 999px; }}
.pdx-branch-row .val {{ font-size: 1.15rem; font-weight: 600; width: 54px; text-align: right; color: {INK}; }}

.pdx-weightline {{
    font-size: 0.82rem;
    color: {INK_SECONDARY};
    margin-top: 0.8rem;
    padding-top: 0.8rem;
    border-top: 1px dashed {HAIRLINE};
}}
.pdx-weightline b {{ color: {INK}; }}

.pdx-shap-row {{ display: flex; align-items: center; gap: 0.8rem; margin-bottom: 0.55rem; }}
.pdx-shap-row .f {{ font-size: 0.82rem; width: 110px; flex: 0 0 110px; color: {INK}; }}
.pdx-shap-row .b {{ flex: 1; height: 15px; background: #EAF1F2; border-radius: 5px; position: relative; overflow: hidden; }}
.pdx-shap-row .b i {{ position: absolute; top: 0; bottom: 0; }}
.pdx-shap-row .v {{ font-size: 0.78rem; width: 50px; text-align: right; color: {INK_SECONDARY}; }}
</style>
"""


def inject_css() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)


def caveat_note(html_body: str) -> None:
    """Amber 'note on trust' box. Not used on Predict/PDF/About anymore per user request
    (removed the imaging-caveat / known-limitations framing there) -- still available for
    genuinely load-bearing warnings elsewhere (e.g. Register's duplicate-name check)."""
    st.markdown(f'<div class="pdx-note"><span class="icon">&#9888;</span><div>{html_body}</div></div>',
                unsafe_allow_html=True)


def read_box(html_body: str) -> None:
    """Light-teal plain-language interpretation callout (mockup's '.read' box) -- a plain
    summary sentence, not a warning, so styled calmer than caveat_note."""
    st.markdown(f'<div class="pdx-read">{html_body}</div>', unsafe_allow_html=True)


def risk_pill(band: str) -> str:
    color = RISK_BAND_COLORS.get(band, INK_SECONDARY)
    return f'<span class="pdx-pill" style="background:{color}">{band} risk</span>'


def research_use_badge() -> None:
    st.markdown('<span class="pdx-research-badge">Research use only</span>', unsafe_allow_html=True)


def score_ring_svg(score: float, band: str, size: int = 140) -> str:
    """Circular progress ring (mockup style) showing the score 0-1, colored by risk band.
    Built as one unindented line -- Streamlit's markdown renderer treats 4+ leading spaces
    on a line as a Markdown code block, which silently breaks HTML parsing partway through
    (a real bug found here: indented multi-line f-strings leaked a literal '</div>' as
    visible text instead of closing the tag)."""
    color = RISK_BAND_COLORS.get(band, PRIMARY)
    r = size * 0.405
    circumference = 2 * math.pi * r
    offset = circumference * (1 - max(0.0, min(1.0, score)))
    c = size / 2
    return (
        f'<div style="position:relative;width:{size}px;height:{size}px;flex:0 0 {size}px;">'
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="#EAF1F2" stroke-width="13"/>'
        f'<circle cx="{c}" cy="{c}" r="{r}" fill="none" stroke="{color}" stroke-width="13" '
        f'stroke-linecap="round" stroke-dasharray="{circumference:.1f}" stroke-dashoffset="{offset:.1f}" '
        f'transform="rotate(-90 {c} {c})"/>'
        f'</svg>'
        f'<div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;">'
        f'<div style="font-size:1.6rem;font-weight:600;color:{INK};line-height:1;">{score:.2f}</div>'
        f'<div style="font-size:0.68rem;letter-spacing:0.08em;text-transform:uppercase;color:{color};font-weight:600;margin-top:2px;">{band}</div>'
        f'</div>'
        f'</div>'
    )


def branch_row(name: str, subtitle: str, value: float, color: str) -> str:
    pct = max(0.0, min(1.0, value)) * 100
    return (
        f'<div class="pdx-branch-row">'
        f'<div class="nm"><div class="t">{name}</div><div class="s">{subtitle}</div></div>'
        f'<div class="bar-t"><i style="width:{pct:.1f}%;background:{color}"></i></div>'
        f'<div class="val">{value:.2f}</div>'
        f'</div>'
    )


def shap_bar_row(feature: str, value: float, max_abs: float) -> str:
    pct = 0 if max_abs == 0 else min(1.0, abs(value) / max_abs) * 46
    color = ACCENT_HIGH_RISK if value >= 0 else PRIMARY
    side = f"left:50%;width:{pct:.1f}%;background:{color};" if value >= 0 else \
           f"right:50%;width:{pct:.1f}%;background:{color};"
    return (
        f'<div class="pdx-shap-row">'
        f'<span class="f">{feature}</span>'
        f'<div class="b"><i style="{side}"></i></div>'
        f'<span class="v">{value:+.2f}</span>'
        f'</div>'
    )


# --- Cached model loaders ------------------------------------------------------

@st.cache_resource(show_spinner="Loading clinical model...")
def get_clinical_branch() -> inf.ClinicalBranch:
    return inf.load_clinical_branch()


@st.cache_resource(show_spinner="Loading imaging model...")
def get_imaging_branch() -> inf.ImagingBranch:
    return inf.load_imaging_branch()


@st.cache_resource(show_spinner=False)
def get_fusion_model_card() -> dict:
    return inf.load_fusion_model_card()


@st.cache_data(show_spinner=False)
def get_sample_imaging_patients():
    return inf.list_sample_imaging_patients()


@st.cache_data(show_spinner=False)
def get_sample_tabular_df():
    import pandas as pd
    from utils.config import TABULAR_CLEAN_PATH
    return pd.read_csv(TABULAR_CLEAN_PATH)


@st.cache_data(show_spinner=False)
def load_results_csv(relative_path: str):
    import pandas as pd
    from utils.config import RESULTS_DIR
    return pd.read_csv(RESULTS_DIR / relative_path)


@st.cache_data(show_spinner=False)
def load_manifest_cache_df():
    import pandas as pd
    from utils.config import CACHE_MANIFEST_PATH
    return pd.read_csv(CACHE_MANIFEST_PATH)


@st.cache_data(show_spinner=False)
def get_clinical_model_card() -> dict:
    import json
    from utils.config import CHECKPOINTS_CLINICAL_FINAL_DIR
    return json.loads((CHECKPOINTS_CLINICAL_FINAL_DIR / "model_card.json").read_text())


@st.cache_data(show_spinner=False)
def get_imaging_model_card() -> dict:
    import json
    from utils.config import CHECKPOINTS_IMAGING_FINAL_DIR
    return json.loads((CHECKPOINTS_IMAGING_FINAL_DIR / "model_card.json").read_text())


def render_gradcam_overlay(slice_tensor, heatmap) -> bytes:
    """slice_tensor: (3, BOX, BOX) float in [0,1] (channel 0 used, they're replicated).
    heatmap: (BOX, BOX) float in [0,1] from inf.generate_gradcam. Returns PNG bytes of a
    grayscale-image + jet-heatmap overlay, matching imaging_confound_check.ipynb's own
    visualization convention."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    img = slice_tensor[0].detach().cpu().numpy()
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img, cmap="gray")
    ax.imshow(heatmap, cmap="jet", alpha=0.45)
    ax.axis("off")
    fig.tight_layout(pad=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    return buf.getvalue()


__all__ = [
    "inf", "classify_risk_band", "inject_css", "caveat_note", "read_box", "risk_pill",
    "research_use_badge", "score_ring_svg", "branch_row", "shap_bar_row", "BIOMARKER_INFO",
    "get_clinical_branch", "get_imaging_branch", "get_fusion_model_card", "get_sample_imaging_patients",
    "get_sample_tabular_df", "load_results_csv", "load_manifest_cache_df",
    "get_clinical_model_card", "get_imaging_model_card", "render_gradcam_overlay",
    "PRIMARY", "PRIMARY_DEEP", "ACCENT_HIGH_RISK", "LOW_RISK", "AMBER_CAVEAT", "INK", "INK_SECONDARY",
    "HAIRLINE", "RISK_BAND_COLORS",
]
