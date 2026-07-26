import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="PaperGrade AI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

from dashboard.views import (
    inject_css,
    render_input_and_prediction,
    render_digital_twin_sandbox,
    render_feedback_audit,
)

inject_css()

with st.sidebar:
    st.markdown("""
    <div style="padding:12px 0 16px 0;margin-bottom:14px;border-bottom:1px solid #e0e0e0">
      <div style="font-size:17px;font-weight:800;color:#111;letter-spacing:-0.02em">PAPERGRADE AI</div>
      <div style="font-size:11px;color:#666;margin-top:2px">Grade Transition Decision Support</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        label="nav",
        options=[
            "Prediction & Recommendations",
            "Digital Twin Sandbox",
            "Feedback & Audit Log",
        ],
        label_visibility="collapsed",
    )

if "Prediction" in page:
    render_input_and_prediction()
elif "Digital Twin" in page:
    render_digital_twin_sandbox()
elif "Feedback" in page:
    render_feedback_audit()
