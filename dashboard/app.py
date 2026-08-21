"""PancraDX -- multimodal early-PDAC risk dashboard. Entry point.

Consumes the project's already-trained, already-calibrated artifacts (checkpoints/{clinical,
imaging,fusion}/final/) via src/inference.py. Never retrains anything.
"""

import streamlit as st

st.set_page_config(page_title="PancraDX", page_icon="\U0001FA7A", layout="wide")

register_page = st.Page("pages/register.py", title="Register", icon=":material/person_add:")
predict_page = st.Page("pages/predict.py", title="Predict", icon=":material/monitor_heart:")
analytics_page = st.Page("pages/analytics.py", title="Analytics", icon=":material/query_stats:")
reports_page = st.Page("pages/reports.py", title="Reports", icon=":material/description:")
about_page = st.Page("pages/about.py", title="About", icon=":material/info:")

nav = st.navigation([register_page, predict_page, analytics_page, reports_page, about_page])
nav.run()
