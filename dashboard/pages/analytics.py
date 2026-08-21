import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from common import (
    inject_css, research_use_badge,
    PRIMARY, PRIMARY_DEEP, ACCENT_HIGH_RISK, LOW_RISK, AMBER_CAVEAT, RISK_BAND_COLORS,
)
from storage import list_cases, load_case

inject_css()


def chart_title(text: str, help_text: str) -> None:
    """Chart heading with a hover (?) explaining what it shows -- same pattern as the
    biomarker input tooltips on Predict."""
    st.markdown(f"**{text}**", help=help_text)


top_left, top_right = st.columns([5, 1])
with top_left:
    st.title("Analytics")
with top_right:
    research_use_badge()

st.caption("Analytics over the patients and cases recorded in this dashboard -- not the training dataset.")

st.markdown(
    "Jump to: [1. Outcome & modality mix](#1-outcome-and-modality-mix) &nbsp;&middot;&nbsp; "
    "[2. Risk profiling](#2-clinical-risk-profiling) &nbsp;&middot;&nbsp; "
    "[3. Model confidence](#3-model-confidence-analysis) &nbsp;&middot;&nbsp; "
    "[4. Temporal trends](#4-temporal-trends) &nbsp;&middot;&nbsp; "
    "[5. Biomarkers](#5-biomarker-measurements-analysis) &nbsp;&middot;&nbsp; "
    "[6. Reliability disclosure](#6-model-limitation-and-reliability-disclosure) &nbsp;&middot;&nbsp; "
    "[7. Summary table](#7-summary-statistics-table)"
)
st.divider()

cases_index = list_cases()

if cases_index.empty:
    st.info("No cases recorded yet -- run one from Predict to see analytics here.")
    if st.button("Go to Predict"):
        st.switch_page("pages/predict.py")
    st.stop()

# =============================================================================
# Load full case detail (imaging/clinical sub-dicts) and derive per-case fields
# =============================================================================
IMAGING_COLOR = "#8FBDC4"
CLINICAL_COLOR = PRIMARY


def _confidence(score: float) -> float:
    """0 (right at the decision boundary) to 1 (maximally confident either way)."""
    return abs(score - 0.5) * 2.0


records = []
for _, row in cases_index.iterrows():
    try:
        full = load_case(row["case_id"])
    except FileNotFoundError:
        continue

    imaging = full.get("imaging")
    clinical = full.get("clinical")

    predicted_labels = []
    if imaging:
        predicted_labels.append("Imaging: Cancer" if imaging["calibrated_proba"] >= 0.5 else "Imaging: Healthy")
    if clinical:
        predicted_labels.append("Clinical: PDAC" if clinical["calibrated_proba"] >= 0.5 else "Clinical: Not PDAC")

    records.append({
        "case_id": full["case_id"],
        "patient_id": full["patient_id"],
        "patient_name": row["patient_name"],
        "created_at": pd.to_datetime(full["created_at"]),
        "modalities": full["modalities"],
        "mode": full["mode"],
        "fused_score": full["fused_score"],
        "risk_band": full["risk_band"],
        "confidence": _confidence(full["fused_score"]),
        "predicted_labels": predicted_labels,
        "imaging": imaging,
        "clinical": clinical,
    })

cases = pd.DataFrame(records)
cases["date"] = cases["created_at"].dt.date

# =============================================================================
# Filters (apply to everything below at once)
# =============================================================================
f1, f2, f3 = st.columns(3)
with f1:
    modality_opts = sorted(cases["modalities"].unique().tolist())
    modality_filter = st.multiselect("Modality", modality_opts, default=modality_opts)
with f2:
    class_opts = sorted({label for labels in cases["predicted_labels"] for label in labels})
    class_filter = st.multiselect("Predicted class", class_opts, default=class_opts)
with f3:
    band_opts = [b for b in ["Low", "Moderate", "High"] if b in cases["risk_band"].unique()]
    band_filter = st.multiselect("Risk band", band_opts, default=band_opts)

mask = (
    cases["modalities"].isin(modality_filter)
    & cases["risk_band"].isin(band_filter)
    & cases["predicted_labels"].apply(lambda labels: not labels or any(l in class_filter for l in labels))
)
fcases = cases[mask].copy()

st.caption(f"Showing {len(fcases)} of {len(cases)} recorded cases.")
st.divider()

if fcases.empty:
    st.info("No cases match the current filters.")
    st.stop()

# =============================================================================
# Section 1 -- Prediction outcome and modality mix
# =============================================================================
st.header("1. Outcome and modality mix")

c1, c2 = st.columns(2)
with c1:
    label_counts: dict[str, int] = {}
    for labels in fcases["predicted_labels"]:
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1
    colors = [IMAGING_COLOR if lbl.startswith("Imaging") else CLINICAL_COLOR for lbl in label_counts]
    chart_title("Predicted-class counts (by branch)",
                "Counts of what each branch predicted -- Cancer/Healthy for imaging, PDAC/Not PDAC "
                "for clinical -- across your filtered cases.")
    fig = go.Figure(go.Bar(x=list(label_counts.keys()), y=list(label_counts.values()), marker_color=colors))
    fig.update_layout(height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
with c2:
    modality_counts = fcases["modalities"].value_counts()
    chart_title("Modality mix",
                "Share of recorded cases that used imaging only, clinical only, or both combined.")
    fig = go.Figure(go.Pie(labels=modality_counts.index, values=modality_counts.values, hole=0.55,
                            marker=dict(colors=[PRIMARY, IMAGING_COLOR, PRIMARY_DEEP])))
    fig.update_layout(height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

chart_title("Risk band by modality",
            "How Low/Moderate/High outcomes break down across each modality combination. A direct "
            "'modality x predicted class' breakdown isn't well-defined for fused cases (imaging and "
            "clinical can predict different classes), so risk band -- always a single value per case "
            "-- is used here instead.")
band_by_modality = pd.crosstab(fcases["modalities"], fcases["risk_band"])
fig = go.Figure()
for band in ["Low", "Moderate", "High"]:
    if band in band_by_modality.columns:
        fig.add_trace(go.Bar(name=band, x=band_by_modality.index, y=band_by_modality[band],
                              marker_color=RISK_BAND_COLORS[band]))
fig.update_layout(barmode="stack", height=340, margin=dict(t=20, b=20))
st.plotly_chart(fig, width="stretch")

st.divider()

# =============================================================================
# Section 2 -- Clinical risk profiling
# =============================================================================
st.header("2. Clinical risk profiling")

c1, c2 = st.columns(2)
with c1:
    band_counts = fcases["risk_band"].value_counts()
    band_order = [b for b in ["Low", "Moderate", "High"] if b in band_counts.index]
    chart_title("Risk-band counts", "How many recorded cases fall into each risk band.")
    fig = go.Figure(go.Bar(x=band_order, y=[band_counts[b] for b in band_order],
                            marker_color=[RISK_BAND_COLORS[b] for b in band_order]))
    fig.update_layout(height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
with c2:
    chart_title("Score distribution",
                "Spread of combined (or single-branch) scores across your filtered cases, with the "
                "average marked.")
    fig = go.Figure(go.Histogram(x=fcases["fused_score"], marker_color=PRIMARY, nbinsx=20))
    mean_score = fcases["fused_score"].mean()
    fig.add_vline(x=mean_score, line_dash="dash", line_color=ACCENT_HIGH_RISK,
                  annotation_text=f"mean {mean_score:.2f}")
    fig.update_layout(xaxis_title="Combined / branch score", xaxis_range=[0, 1],
                      height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

chart_title("CA19-9 vs. age",
            "Each point is one case's biomarker reading, colored by its risk band. CA19-9 is shown on "
            "a log scale since it spans a very wide range (single digits to tens of thousands).")
scatter_rows = []
for _, r in fcases.iterrows():
    if r["clinical"] and r["clinical"].get("raw_features"):
        rf = r["clinical"]["raw_features"]
        scatter_rows.append({"age": rf["age"], "ca19_9": rf["plasma_CA19_9"], "risk_band": r["risk_band"],
                              "case_id": r["case_id"]})
if scatter_rows:
    sdf = pd.DataFrame(scatter_rows)
    sdf = sdf[sdf["ca19_9"] > 0]  # log scale can't plot zero/negative
    fig = go.Figure()
    for band in ["Low", "Moderate", "High"]:
        sub = sdf[sdf["risk_band"] == band]
        if not sub.empty:
            fig.add_trace(go.Scatter(x=sub["age"], y=sub["ca19_9"], mode="markers", name=band,
                                      marker=dict(color=RISK_BAND_COLORS[band], size=10)))
    fig.update_layout(xaxis_title="Age", yaxis_title="Plasma CA19-9 (U/mL, log scale)",
                      yaxis_type="log", height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No cases with saved biomarker values yet -- this needs clinical cases run after this "
            "feature was added (older cases only stored the score, not the raw values).")

st.divider()

# =============================================================================
# Section 3 -- Model confidence analysis
# =============================================================================
st.header("3. Model confidence analysis")
st.caption("Confidence here means how far a score sits from the 0.5 decision boundary "
           "(0 = right at the boundary, 1 = maximally confident) -- computed from your recorded "
           "cases, not the same thing as the training-time calibration Brier scores shown in About.")

c1, c2 = st.columns(2)
with c1:
    chart_title("Confidence distribution",
                "How far each case's score sits from the 0.5 boundary, with the 25th/median/75th "
                "percentile marked.")
    fig = go.Figure(go.Histogram(x=fcases["confidence"], marker_color=PRIMARY, nbinsx=20))
    p25, p50, p75 = fcases["confidence"].quantile([0.25, 0.5, 0.75])
    for p, label, color in [(p25, "p25", AMBER_CAVEAT), (p50, "median", ACCENT_HIGH_RISK), (p75, "p75", AMBER_CAVEAT)]:
        fig.add_vline(x=p, line_dash="dash", line_color=color, annotation_text=f"{label} {p:.2f}")
    fig.update_layout(xaxis_title="Confidence", xaxis_range=[0, 1], height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
with c2:
    chart_title("Average confidence by modality",
                "Mean decisiveness per branch across your recorded cases, with the spread (error bars) "
                "shown too -- this dashboard's own usage, not the training-time Brier scores.")
    branch_conf = {"Imaging": [], "Clinical": []}
    for _, r in fcases.iterrows():
        if r["imaging"]:
            branch_conf["Imaging"].append(_confidence(r["imaging"]["calibrated_proba"]))
        if r["clinical"]:
            branch_conf["Clinical"].append(_confidence(r["clinical"]["calibrated_proba"]))
    branches = [b for b in branch_conf if branch_conf[b]]
    means = [np.mean(branch_conf[b]) for b in branches]
    stds = [np.std(branch_conf[b]) for b in branches]
    fig = go.Figure(go.Bar(x=branches, y=means, error_y=dict(type="data", array=stds),
                            marker_color=[IMAGING_COLOR if b == "Imaging" else CLINICAL_COLOR for b in branches]))
    fig.update_layout(yaxis_title="Mean confidence", yaxis_range=[0, 1], height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

st.divider()

# =============================================================================
# Section 4 -- Temporal trends
# =============================================================================
st.header("4. Temporal trends")

by_date = fcases.groupby("date").agg(cases=("case_id", "count"), mean_conf=("confidence", "mean")).reset_index()
by_date["date_str"] = by_date["date"].astype(str)
by_date["rolling_avg"] = by_date["cases"].rolling(3, min_periods=1).mean()

c1, c2 = st.columns(2)
with c1:
    chart_title("Daily case volume",
                "How many cases were run each day, with a 3-day rolling average to smooth day-to-day "
                "noise. Click a bar to see that day's cases below.")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=by_date["date_str"], y=by_date["cases"], name="Cases", marker_color=PRIMARY))
    fig.add_trace(go.Scatter(x=by_date["date_str"], y=by_date["rolling_avg"], name="Rolling avg (3)",
                              mode="lines", line=dict(color=ACCENT_HIGH_RISK)))
    fig.update_xaxes(type="category")
    fig.update_layout(height=380, margin=dict(t=20, b=20))
    event = st.plotly_chart(fig, width="stretch", on_select="rerun", selection_mode="points", key="volume_chart")
with c2:
    chart_title("Mean confidence over time", "Average decisiveness of cases run on each day.")
    fig = go.Figure(go.Scatter(x=by_date["date_str"], y=by_date["mean_conf"], mode="lines+markers",
                                line=dict(color=PRIMARY)))
    fig.update_xaxes(type="category")
    fig.update_layout(yaxis_range=[0, 1], height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

clicked_points = event.selection.points if event and event.selection else []
if clicked_points:
    clicked_date = clicked_points[0]["x"]
    st.markdown(f"**Cases on {clicked_date}**")
    day_cases = fcases[fcases["date"].astype(str) == clicked_date]
    st.dataframe(day_cases[["case_id", "patient_name", "modalities", "fused_score", "risk_band"]],
                 hide_index=True, width="stretch")
else:
    st.caption("Click a bar on the left chart to drill into that day's cases.")

st.divider()

# =============================================================================
# Section 5 -- Biomarker measurements analysis
# =============================================================================
st.header("5. Biomarker measurements analysis")

BIOMARKER_KEYS = ["creatinine", "LYVE1", "REG1B", "TFF1", "plasma_CA19_9"]
box_rows = []
for _, r in fcases.iterrows():
    if r["clinical"] and r["clinical"].get("raw_features"):
        rf = r["clinical"]["raw_features"]
        pred_label = "PDAC" if r["clinical"]["calibrated_proba"] >= 0.5 else "Not PDAC"
        for feat in BIOMARKER_KEYS:
            box_rows.append({"feature": feat, "value": rf[feat], "predicted_class": pred_label})

if box_rows:
    box_df = pd.DataFrame(box_rows)
    chart_title("Biomarker distributions by predicted class",
                "Spread of each biomarker's real measured value, split by what the clinical branch "
                "predicted for that case.")
    cols = st.columns(3)
    for i, feat in enumerate(BIOMARKER_KEYS):
        with cols[i % 3]:
            sub = box_df[box_df["feature"] == feat]
            fig = go.Figure()
            for cls, color in [("Not PDAC", LOW_RISK), ("PDAC", ACCENT_HIGH_RISK)]:
                vals = sub[sub["predicted_class"] == cls]["value"]
                if not vals.empty:
                    fig.add_trace(go.Box(y=vals, name=cls, marker_color=color))
            fig.update_layout(title=feat, height=300, margin=dict(t=30, b=20), showlegend=False)
            st.plotly_chart(fig, width="stretch")

    chart_title("Age vs. CA19-9, sized and colored by combined score",
                "Same two biomarkers as the scatter above, but each point's size and color reflect the "
                "combined score for that case -- bigger and redder means higher risk.")
    bubble_rows = []
    for _, r in fcases.iterrows():
        if r["clinical"] and r["clinical"].get("raw_features"):
            rf = r["clinical"]["raw_features"]
            bubble_rows.append({"age": rf["age"], "ca19_9": rf["plasma_CA19_9"],
                                 "score": r["fused_score"], "case_id": r["case_id"]})
    bdf = pd.DataFrame(bubble_rows)
    fig = go.Figure(go.Scatter(
        x=bdf["age"], y=bdf["ca19_9"], mode="markers",
        marker=dict(size=bdf["score"] * 40 + 8, color=bdf["score"], colorscale=[[0, LOW_RISK], [1, ACCENT_HIGH_RISK]],
                    showscale=True, colorbar=dict(title="Score")),
        text=bdf["case_id"],
    ))
    fig.update_layout(xaxis_title="Age", yaxis_title="Plasma CA19-9 (U/mL)", height=420, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No cases with saved biomarker values yet -- run and save a clinical case from Predict "
            "to populate these charts.")

chart_title("Per-slice risk profile (volume-based imaging cases)",
            "The calibrated probability for every individual slice in one volume scan -- a near-flat "
            "line is the expected, honest pattern given the imaging caveat in Section 6, not a "
            "rendering issue.")
volume_cases = fcases[fcases["imaging"].apply(
    lambda im: bool(im) and bool(im.get("per_slice_proba")) and "volume" in im.get("granularity", "")
)]
if volume_cases.empty:
    st.caption("No saved volume-based imaging cases with a per-slice profile yet.")
else:
    options = volume_cases["case_id"].tolist()
    chosen = st.selectbox("Case", options)
    chosen_row = volume_cases[volume_cases["case_id"] == chosen].iloc[0]
    per_slice = chosen_row["imaging"]["per_slice_proba"]
    fig = go.Figure(go.Scatter(y=per_slice, mode="lines", line=dict(color=IMAGING_COLOR)))
    fig.update_layout(xaxis_title="Slice index", yaxis_title="P(cancer)", yaxis_range=[0, 1],
                      height=340, margin=dict(t=20, b=20))
    st.plotly_chart(fig, width="stretch")

st.divider()

# =============================================================================
# Section 6 -- Model limitation & reliability disclosure
# =============================================================================
st.header("6. Model limitation and reliability disclosure")

granularity_labels = []
for _, r in fcases.iterrows():
    if not r["imaging"]:
        granularity_labels.append("Not used")
    elif "volume" in r["imaging"].get("granularity", ""):
        granularity_labels.append("Volume (mean)")
    else:
        granularity_labels.append("Single-slice")
gcounts = pd.Series(granularity_labels).value_counts()
chart_title("How imaging contributed across recorded cases",
            "Whether imaging cases used a single slice or the whole-volume average, versus cases "
            "where imaging wasn't used at all.")
fig = go.Figure(go.Pie(labels=gcounts.index, values=gcounts.values, hole=0.55,
                        marker=dict(colors=[PRIMARY, IMAGING_COLOR, "#C9D3D5"])))
fig.update_layout(height=380, margin=dict(t=20, b=20))
st.plotly_chart(fig, width="stretch")

st.divider()

# =============================================================================
# Section 7 -- Summary statistics table
# =============================================================================
st.header("7. Summary statistics table")

summary_rows = []
for _, r in fcases.iterrows():
    predicted = ", ".join(r["predicted_labels"]) if r["predicted_labels"] else "--"
    if r["imaging"]:
        key_info = f"{r['imaging']['n_slices']} slice(s)"
    elif r["clinical"] and r["clinical"].get("shap_per_feature"):
        top_feat = max(r["clinical"]["shap_per_feature"].items(), key=lambda kv: abs(kv[1]))[0]
        key_info = f"top biomarker: {top_feat}"
    else:
        key_info = "--"
    summary_rows.append({
        "Case ID": r["case_id"], "Patient": r["patient_name"], "Modality": r["modalities"],
        "Predicted class": predicted, "Risk band": r["risk_band"], "Fused score": round(r["fused_score"], 3),
        "Key info": key_info, "Timestamp": r["created_at"],
    })
st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")
