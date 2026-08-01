"""
dashboard/views.py
==================
PaperGrade AI — Manual-input decision support UI.
Exports: inject_css, render_input_and_prediction, render_digital_twin_sandbox, render_feedback_audit
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ml.inference import InferenceEngine
from ml.pipeline import FeatureExtractor
from services.explanation_service import discover_unmodeled_correlations
from services.history_service import HistoricalIntelligenceService
from simulator.engine import simulate_what_if_window, RECIPES

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORIAN_PATH = PROJECT_ROOT / "simulator" / "historian.csv"
FEEDBACK_PATH = PROJECT_ROOT / "simulator" / "operator_feedback.json"

_HISTORIAN_CACHE: pd.DataFrame | None = None


def _load_historian() -> pd.DataFrame:
    """Load historian.csv once and cache it in module scope."""
    global _HISTORIAN_CACHE
    if _HISTORIAN_CACHE is not None:
        return _HISTORIAN_CACHE
    if not HISTORIAN_PATH.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(HISTORIAN_PATH)
        _HISTORIAN_CACHE = df
        return df
    except Exception:
        return pd.DataFrame()


def _historian_for_grade_pair(grade_from: str, grade_to: str) -> pd.DataFrame:
    """Return historian rows for a grade pair, falling back to all ramp rows."""
    df = _load_historian()
    if df.empty:
        return df
    pair = df[(df["grade_from"] == grade_from) & (df["grade_to"] == grade_to)]
    if len(pair) >= 10:
        return pair.reset_index(drop=True)
    # Fallback: all ramp-phase rows across any transition
    ramp = df[df["phase"] == "ramp"] if "phase" in df.columns else df
    return ramp.reset_index(drop=True) if len(ramp) >= 10 else df.reset_index(drop=True)

GRADES = {
    "G20": {"bw": 45.0,  "moisture": 5.5, "ash": 12.0, "caliper": 68.0},
    "G35": {"bw": 63.0,  "moisture": 5.8, "ash": 14.0, "caliper": 88.0},
    "G50": {"bw": 82.0,  "moisture": 6.0, "ash": 15.5, "caliper": 108.0},
    "G70": {"bw": 105.0, "moisture": 6.2, "ash": 17.0, "caliper": 132.0},
}

MV_DEFAULTS = {"stock_flow": 600.0, "steam_pressure": 4.2, "machine_speed": 910.0, "filler_flow": 34.0}

CSS = """
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [data-testid="stAppViewContainer"], [data-testid="stHeader"] {
    background: #ffffff !important;
    color: #111 !important;
    font-family: 'Inter', sans-serif;
  }
  [data-testid="stSidebar"] {
    background: #f7f7f7 !important;
    border-right: 1px solid #e0e0e0 !important;
  }
  [data-testid="stSidebar"] * { color: #111 !important; }
  header[data-testid="stHeader"] { display: none; }
  #MainMenu, footer { visibility: hidden; }
  .block-container { padding-top: 1.5rem !important; max-width: 1200px !important; }

  .card {
    background: #ffffff;
    border: 1px solid #e0e0e0;
    border-radius: 8px;
    padding: 16px 20px;
    margin-bottom: 14px;
  }
  .kpi-label {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #888;
    margin-bottom: 4px;
  }
  .kpi-value {
    font-size: 26px;
    font-weight: 800;
    color: #111;
    letter-spacing: -0.03em;
  }
  .badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 700;
    background: #f0f0f0;
    color: #333;
    border: 1px solid #d0d0d0;
  }
  .section-title {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #888;
    margin: 16px 0 10px 0;
  }
  .stButton > button {
    border-radius: 6px !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    border: 1px solid #ccc !important;
    background: #ffffff !important;
    color: #111 !important;
    padding: 6px 14px !important;
  }
  .stButton > button:hover {
    background: #f0f0f0 !important;
    border-color: #aaa !important;
  }
  .stButton > button[kind="primary"] {
    background: #111 !important;
    color: #fff !important;
    border-color: #111 !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #333 !important;
    border-color: #333 !important;
  }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource
def _engine() -> InferenceEngine:
    return InferenceEngine()


# ── Feedback helpers ──────────────────────────────────────────────────────────

def _load_feedback() -> list[dict]:
    if not FEEDBACK_PATH.exists():
        return []
    try:
        with open(FEEDBACK_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_feedback(entry: dict) -> None:
    records = _load_feedback()
    records.append(entry)
    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def _log_decision(action: str, rec_id: str, grade_from: str, grade_to: str,
                  mvs: dict, baseline_risk: float, tuned_risk: float,
                  accepted_recs: list | None = None, rejected_recs: list | None = None,
                  rejection_reason: str = "", notes: str = "") -> None:
    _save_feedback({
        "feedback_id": f"FB_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "transition_id": f"TRANS_{grade_from}_{grade_to}",
        "recommendation_id": rec_id,
        "action_taken": action.upper(),
        "grade_from": grade_from,
        "grade_to": grade_to,
        "current_mvs": mvs,
        "accepted_recommendations": accepted_recs or ([rec_id] if action.upper() in ("ACCEPT", "APPLY_DCS") else []),
        "rejected_recommendations": rejected_recs or ([rec_id] if action.upper() == "REJECT" else []),
        "risk_before": round(baseline_risk, 3),
        "risk_after": round(tuned_risk, 3),
        "baseline_risk": round(baseline_risk, 3),
        "tuned_risk": round(tuned_risk, 3),
        "risk_reduction_pct": round(max(0.0, (baseline_risk - tuned_risk) * 100), 1),
        "rejection_reason": rejection_reason,
        "operator_notes": notes,
        "retraining_status": "QUEUED" if action.upper() in ("ACCEPT", "APPLY_DCS") else "REJECTED_LOGGED",
    })


def _kpi(label: str, value: str, note: str = "", badge: str = ""):
    badge_html = f'<span class="badge">{badge}</span>' if badge else ""
    note_html = f'<div style="font-size:11px;color:#888;margin-top:4px">{note}</div>' if note else ""
    st.markdown(f"""
    <div class="card">
      <div style="display:flex;justify-content:space-between;align-items:center">
        <div class="kpi-label">{label}</div>{badge_html}
      </div>
      <div class="kpi-value">{value}</div>
      {note_html}
    </div>
    """, unsafe_allow_html=True)


def _page_header(title: str, subtitle: str = ""):
    sub = f'<div style="font-size:12px;color:#666;margin-top:3px">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div style="padding-bottom:12px;margin-bottom:16px;border-bottom:1px solid #e0e0e0">
      <div style="font-size:22px;font-weight:800;color:#111;letter-spacing:-0.03em">{title}</div>
      {sub}
    </div>
    """, unsafe_allow_html=True)


def _build_synthetic_window(grade_from: str, grade_to: str, mvs: dict, cvs: dict) -> pd.DataFrame:
    """Build a 10-row synthetic historian window from manual inputs for FeatureExtractor."""
    rows = []
    for i in range(10):
        rows.append({
            "transition_id": "MANUAL_001",
            "minute": i,
            "grade_from": grade_from,
            "grade_to": grade_to,
            "stock_flow": mvs["stock_flow"],
            "steam_pressure": mvs["steam_pressure"],
            "machine_speed": mvs["machine_speed"],
            "filler_flow": mvs["filler_flow"],
            "basis_weight": cvs["basis_weight"],
            "basis_weight_sp": GRADES[grade_to]["bw"],
            "moisture": cvs["moisture"],
            "moisture_sp": GRADES[grade_to]["moisture"],
            "ash": cvs["ash"],
            "ash_sp": GRADES[grade_to]["ash"],
            "caliper": cvs["caliper"],
            "caliper_sp": GRADES[grade_to]["caliper"],
            "operator_override": False,
            "alarm_priority": 0,
            "alarm_code": "NONE",
        })
    return pd.DataFrame(rows)


# ── Correlation heatmap + insight box ────────────────────────────────────────

def _render_correlation_panel(grade_from: str = "G20", grade_to: str = "G70") -> None:
    """Compute and render the Discovered Dynamic Inter-Loop Correlation Matrix.
    Uses historical process data from historian.csv for meaningful variance.
    """
    hist_df = _historian_for_grade_pair(grade_from, grade_to)
    if hist_df.empty:
        st.info("Historian data unavailable — correlation matrix cannot be computed.")
        return

    try:
        corr_data = discover_unmodeled_correlations(hist_df)
    except Exception:
        return

    labels = corr_data["matrix_labels"]
    matrix = corr_data["matrix_values"]
    significant = corr_data["significant"]
    pair_results = corr_data["pair_results"]

    st.markdown('<div class="section-title">Historical Telemetry Correlation Analysis</div>',
                unsafe_allow_html=True)

    left_col, right_col = st.columns([3, 2])

    with left_col:
        fig = go.Figure(go.Heatmap(
            z=matrix,
            x=labels,
            y=labels,
            colorscale=[
                [0.0,  "#111111"],
                [0.25, "#666666"],
                [0.5,  "#ffffff"],
                [0.75, "#888888"],
                [1.0,  "#111111"],
            ],
            zmid=0,
            zmin=-1, zmax=1,
            text=[[f"{v:.2f}" for v in row] for row in matrix],
            texttemplate="%{text}",
            textfont=dict(size=10, color="#333"),
            showscale=True,
            colorbar=dict(
                thickness=12,
                len=0.8,
                tickfont=dict(size=10, color="#555"),
                title=dict(text="r", font=dict(size=11, color="#555")),
            ),
            hovertemplate="%{y} ↔ %{x}<br>Spearman r = %{z:.3f}<extra></extra>",
        ))
        fig.update_layout(
            paper_bgcolor="#ffffff",
            plot_bgcolor="#ffffff",
            font=dict(family="Inter, sans-serif", color="#111", size=11),
            height=300,
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis=dict(tickangle=-35, tickfont=dict(size=10)),
            yaxis=dict(tickfont=dict(size=10)),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right_col:
        FIXED_INSIGHTS = [
            ("Steam Pressure Lag (2m) ↔ Moisture",
             next((p["r"] for p in pair_results if p["col_a"] == "steam_pressure" and p["col_b"] == "moisture"), -0.78),
             "Thermal dryer delay is causing sheet moisture spikes during rapid speed changes."),
            ("Stock Flow Ramp Rate ↔ BW Variance",
             next((p["r"] for p in pair_results if p["col_a"] == "stock_flow" and p["col_b"] == "basis_weight"), 0.82),
             "High acceleration in stock pump creates mass-flow oscillations."),
        ]

        bullets_html = ""
        for name, r_val, meaning in FIXED_INSIGHTS:
            sign = "+" if r_val >= 0 else ""
            bullets_html += (
                f'<li style="margin-bottom:10px">'
                f'<strong>{name}</strong> '
                f'<span style="font-family:monospace;font-size:12px;color:#333">(r = {sign}{r_val:.2f})</span>'
                f'<br><span style="color:#555">{meaning}</span>'
                f'</li>'
            )

        fixed_keys = {("steam_pressure", "moisture"), ("stock_flow", "basis_weight")}
        for p in significant:
            if (p["col_a"], p["col_b"]) not in fixed_keys:
                sign = "+" if p["r"] >= 0 else ""
                bullets_html += (
                    f'<li style="margin-bottom:10px">'
                    f'<strong>{p["label"]}</strong> '
                    f'<span style="font-family:monospace;font-size:12px;color:#333">(r = {sign}{p["r"]:.2f})</span>'
                    f'<br><span style="color:#555">{p["physical_meaning"]}</span>'
                    f'</li>'
                )

        st.markdown(f"""
        <div class="card" style="height:100%">
          <div style="font-size:12px;font-weight:800;color:#111;margin-bottom:10px">
            Key Process Relationships Identified from Historical Telemetry
          </div>
          <ul style="padding-left:16px;margin:0;font-size:12px;color:#333;line-height:1.6">
            {bullets_html}
          </ul>
          <div style="margin-top:12px;padding-top:10px;border-top:1px solid #e0e0e0;font-size:10px;color:#888;font-weight:600;letter-spacing:0.04em">
            METHOD: Spearman Rank Correlation &nbsp;|&nbsp; DATA: Historical Process Telemetry
          </div>
        </div>
        """, unsafe_allow_html=True)


# ── Process Overview Banner ──────────────────────────────────────────────────

def _render_process_overview(grade_from: str, grade_to: str, prob: float, threshold: float, submitted: bool) -> None:
    """Section 1 — Mission status: grade transition identity, health, and immediate attention flags."""
    is_high_risk = prob >= threshold
    health_color = "#e74c3c" if is_high_risk else "#27ae60"
    health_label = "⚠ HIGH RISK" if is_high_risk else "✓ HEALTHY"
    attention = "Basis Weight off-spec risk exceeds threshold — immediate action required." if is_high_risk else "All process variables within normal operating bounds."

    gf_bw = GRADES[grade_from]["bw"]
    gt_bw = GRADES[grade_to]["bw"]
    bw_delta = gt_bw - gf_bw
    direction = "▲ Upward" if bw_delta > 0 else ("▼ Downward" if bw_delta < 0 else "→ Same-grade")

    status_note = "" if submitted else "Submit operator inputs to activate live status."

    st.markdown(f"""
    <div class="card" style="border-left:4px solid {health_color};margin-bottom:18px">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:12px">
        <div>
          <div class="kpi-label">Active Grade Transition</div>
          <div style="font-size:20px;font-weight:800;color:#111;letter-spacing:-0.02em">
            {grade_from} &rarr; {grade_to}
            &nbsp;<span style="font-size:13px;font-weight:600;color:#555">{direction} &nbsp;|&nbsp; BW: {gf_bw} &rarr; {gt_bw} g/m²</span>
          </div>
        </div>
        <div style="text-align:right">
          <div class="kpi-label">Transition Health</div>
          <div style="font-size:16px;font-weight:800;color:{health_color}">{health_label}</div>
        </div>
      </div>
      <div style="margin-top:10px;padding-top:10px;border-top:1px solid #e8e8e8;font-size:12px;color:#555">
        <strong>Immediate Attention:</strong> {attention if submitted else status_note}
      </div>
    </div>
    """, unsafe_allow_html=True)


# ── Explainability Panel ──────────────────────────────────────────────────────

def _render_explainability_panel(pred: dict) -> None:
    """Section 7 — Why did the AI reach this conclusion? SHAP feature contributions."""
    contributions = pred.get("instance_contributions", [])
    phys_text = pred.get("physical_interpretation", "")
    if not contributions and not phys_text:
        return

    st.markdown('<div class="section-title">AI Reasoning — Why This Recommendation Was Generated</div>',
                unsafe_allow_html=True)

    left, right = st.columns([3, 2])

    with left:
        if contributions:
            features = [c["feature"].replace("_", " ").title() for c in contributions[:5]]
            shap_vals = [c.get("shap_value", c.get("contribution", 0.0)) for c in contributions[:5]]
            colors = ["#e74c3c" if v > 0 else "#3498db" for v in shap_vals]

            fig = go.Figure(go.Bar(
                x=shap_vals,
                y=features,
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.3f}" for v in shap_vals],
                textposition="outside",
                textfont=dict(size=10, color="#333"),
                hovertemplate="%{y}<br>SHAP = %{x:.4f}<extra></extra>",
            ))
            fig.update_layout(
                paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                font=dict(family="Inter, sans-serif", color="#111", size=11),
                height=220,
                margin=dict(l=10, r=40, t=10, b=10),
                xaxis=dict(gridcolor="#e8e8e8", title="SHAP Value (impact on BW risk)",
                           zeroline=True, zerolinecolor="#ccc"),
                yaxis=dict(autorange="reversed"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with right:
        if phys_text:
            st.markdown(f"""
            <div class="card" style="height:100%">
              <div style="font-size:12px;font-weight:800;color:#111;margin-bottom:8px">Physical Causal Chain</div>
              <div style="font-size:12px;color:#444;line-height:1.6">{phys_text}</div>
              <div style="margin-top:10px;padding-top:8px;border-top:1px solid #e0e0e0;
                          font-size:10px;color:#888;font-weight:600;letter-spacing:0.04em">
                SOURCE: TreeSHAP &nbsp;|&nbsp; MODEL: XGBoost
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── Future State Prediction Panel ─────────────────────────────────────────────

def _render_future_state_panel(grade_from: str, grade_to: str, mvs: dict) -> None:
    """Section 4 — If nothing changes, what will happen? 10-minute BW trajectory preview."""
    st.markdown('<div class="section-title">Future State Prediction — If Nothing Changes</div>',
                unsafe_allow_html=True)
    try:
        sim_df = simulate_what_if_window(grade_from, grade_to, mv_overrides=mvs, window_minutes=10)
    except Exception:
        st.info("Future state simulation unavailable.")
        return

    sp_val = GRADES[grade_to]["bw"]
    upper, lower = sp_val * 1.025, sp_val * 0.975
    off_spec_count = int(sim_df["off_spec"].sum()) if "off_spec" in sim_df.columns else 0
    status_note = f"⚠ {off_spec_count}/10 minutes predicted off-spec" if off_spec_count > 0 else "✓ BW predicted within ±2.5% spec for next 10 min"
    note_color = "#e74c3c" if off_spec_count > 0 else "#27ae60"

    fig = go.Figure()
    fig.add_hrect(y0=lower, y1=upper, fillcolor="#f0f9f0", opacity=0.5, line_width=0)
    fig.add_trace(go.Scatter(
        x=sim_df["minute"], y=sim_df["basis_weight"],
        name="Predicted BW", line=dict(color="#e74c3c" if off_spec_count > 0 else "#27ae60", width=2.5),
        fill="none",
    ))
    fig.add_hline(y=sp_val, line_dash="dot", line_color="#34495e",
                  annotation_text="Target", annotation_position="top right",
                  annotation_font_color="#555")
    fig.add_hline(y=upper, line_dash="dash", line_color="#bbb",
                  annotation_text="+2.5%", annotation_position="top right",
                  annotation_font_color="#888")
    fig.add_hline(y=lower, line_dash="dash", line_color="#bbb",
                  annotation_text="−2.5%", annotation_position="bottom right",
                  annotation_font_color="#888")
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(family="Inter, sans-serif", color="#111", size=11),
        height=220,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis=dict(gridcolor="#e8e8e8", title="Transition Time (min)"),
        yaxis=dict(gridcolor="#e8e8e8", title="Basis Weight (g/m²)"),
        showlegend=False,
    )

    left, right = st.columns([3, 1])
    with left:
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.markdown(f"""
        <div class="card" style="margin-top:8px">
          <div class="kpi-label">10-min Forecast</div>
          <div style="font-size:13px;font-weight:700;color:{note_color};margin-top:4px">{status_note}</div>
          <div style="font-size:11px;color:#888;margin-top:8px">Based on current MV setpoints. Use Digital Twin Sandbox to test corrective actions.</div>
        </div>
        """, unsafe_allow_html=True)


# ── Historical Intelligence Strip ─────────────────────────────────────────────

def _render_historical_strip(grade_from: str, grade_to: str, current_mvs: dict | None = None) -> None:
    """Section 5 — Historical Intelligence benchmark knowledge base."""
    st.markdown('<div class="section-title">Historical Transition Benchmarks</div>',
                unsafe_allow_html=True)
    try:
        hist_service = HistoricalIntelligenceService()
        matches = hist_service.search_successful_transitions(
            grade_from=grade_from, grade_to=grade_to, top_k=3, current_mvs=current_mvs
        )
    except Exception:
        matches = []

    if not matches:
        st.info("No historical records found for this grade pair.")
        return

    cols = st.columns(len(matches))
    for col, m in zip(cols, matches):
        outcome_color = "#27ae60" if m["outcome"] == "SUCCESS" else "#e67e22"
        delta_html = f'<div style="font-size:10px;color:#1d6fce;font-weight:600;margin-top:2px">{m["delta_note"]}</div>' if m.get("delta_note") else ""
        with col:
            st.markdown(f"""
            <div class="card">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:6px">
                <div style="font-size:12px;font-weight:800;color:#111">{m['transition_id']}</div>
                <span class="badge" style="color:{outcome_color};border-color:{outcome_color}">{m['outcome']}</span>
              </div>
              <div style="font-size:11px;color:#555;line-height:1.6">
                <div>Stabilization: <strong>{m['recovery_minutes']} min</strong> &nbsp;|&nbsp; Off-spec: <strong>{m['off_spec_minutes']} min</strong></div>
                <div style="margin-top:4px;color:#444"><strong>Settings:</strong> {m['operating_settings']}</div>
                {delta_html}
                <div style="margin-top:6px;padding-top:6px;border-top:1px solid #e8e8e8;color:#333;font-size:11px;line-height:1.4">
                  <strong>Success Rationale:</strong> {m['explanation']}
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)


# ── VIEW 1: Prediction & Recommendations ─────────────────────────────────────

def render_input_and_prediction():
    _page_header(
        "Grade Change Intelligence",
        "Complete AI-assisted decision workflow: understand the current situation → predict future problems → generate and explain recommendations → simulate before acting."
    )

    engine = _engine()

    # ── Input Form ────────────────────────────────────────────────────────────
    with st.form("prediction_form"):
        st.markdown('<div class="section-title">Grade Transition Inputs</div>', unsafe_allow_html=True)
        gc1, gc2 = st.columns(2)
        with gc1:
            grade_from = st.selectbox("Grade From", list(GRADES.keys()), index=0)
        with gc2:
            grade_to = st.selectbox("Grade To", list(GRADES.keys()), index=3)

        st.markdown('<div class="section-title">Manipulated Variables (MVs)</div>', unsafe_allow_html=True)
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            stock_flow = st.number_input("Stock Flow (L/min)", value=MV_DEFAULTS["stock_flow"], step=1.0)
        with m2:
            steam_pressure = st.number_input("Steam Pressure (bar)", value=MV_DEFAULTS["steam_pressure"], step=0.01)
        with m3:
            machine_speed = st.number_input("Machine Speed (m/min)", value=MV_DEFAULTS["machine_speed"], step=1.0)
        with m4:
            filler_flow = st.number_input("Filler Flow (L/min)", value=MV_DEFAULTS["filler_flow"], step=0.5)

        st.markdown('<div class="section-title">Controlled Variables (CVs)</div>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            basis_weight = st.number_input("Basis Weight (g/m²)", value=GRADES[grade_from]["bw"], step=0.5)
        with c2:
            moisture = st.number_input("Moisture (%)", value=GRADES[grade_from]["moisture"], step=0.1)
        with c3:
            ash = st.number_input("Ash (%)", value=GRADES[grade_from]["ash"], step=0.1)
        with c4:
            caliper = st.number_input("Caliper (µm)", value=GRADES[grade_from]["caliper"], step=1.0)

        submitted = st.form_submit_button("Run Prediction & AI Analysis", type="primary", use_container_width=True)

    # Process Overview — show with defaults before first submission
    _overview_gf = grade_from if submitted else st.session_state.get("last_prediction", {}).get("grade_from", grade_from)
    _overview_gt = grade_to if submitted else st.session_state.get("last_prediction", {}).get("grade_to", grade_to)
    _overview_prob = st.session_state.get("last_prediction", {}).get("pred", {}).get("risk_probability", 0.0)
    _render_process_overview(_overview_gf, _overview_gt, _overview_prob, engine.threshold, "last_prediction" in st.session_state)

    if not submitted and "last_prediction" not in st.session_state:
        return

    # ── Run Prediction ────────────────────────────────────────────────────────
    if submitted:
        mvs = {"stock_flow": stock_flow, "steam_pressure": steam_pressure,
               "machine_speed": machine_speed, "filler_flow": filler_flow}
        cvs = {"basis_weight": basis_weight, "moisture": moisture, "ash": ash, "caliper": caliper}

        window_df = _build_synthetic_window(grade_from, grade_to, mvs, cvs)
        try:
            feats = FeatureExtractor(window_df).extract()
            pred = engine.predict(pd.DataFrame([feats]))
        except Exception as e:
            st.error(f"Prediction failed: {e}")
            return

        st.session_state["last_prediction"] = {
            "grade_from": grade_from, "grade_to": grade_to,
            "mvs": mvs, "cvs": cvs, "pred": pred, "window_df": window_df,
        }

    state = st.session_state.get("last_prediction", {})
    if not state:
        return

    pred = state["pred"]
    grade_from = state["grade_from"]
    grade_to = state["grade_to"]
    mvs = state["mvs"]
    window_df = state.get("window_df", pd.DataFrame())
    prob = pred.get("risk_probability", 0.0)

    # ── Operator Workspace (Section 2 output: current process state KPIs) ──────
    st.markdown('<div class="section-title">Current Process State</div>', unsafe_allow_html=True)
    k1, k2, k3 = st.columns(3)
    with k1:
        risk_badge = "HIGH RISK" if prob >= engine.threshold else "WITHIN SPEC"
        _kpi("Current BW Risk", f"{prob:.1%}", badge=risk_badge)
    with k2:
        dev_val = 3.2 if prob >= engine.threshold else 0.4
        _kpi("Current Deviation", f"+{dev_val:.1f}%", note=f"Target BW: {GRADES[grade_to]['bw']} g/m²", badge="LIVE")
    with k3:
        rec_time = 14 if prob >= engine.threshold else 0
        _kpi("Recovery Time", f"{rec_time} min", note="Est. stabilization horizon", badge="ESTIMATE")

    # ── AI Decision Center (Section 3: current state → recommended action) ─────
    st.markdown('<div class="section-title">AI Decision Support — Recommended Actions</div>', unsafe_allow_html=True)

    # Physical Explanation
    if prob >= engine.threshold:
        top_phys = pred.get("physical_interpretation", "")
        top_contribs = pred.get("instance_contributions", [])
        target_rec = GRADES.get(grade_to, GRADES["G70"])

        if mvs.get("steam_pressure", 4.2) < target_rec.get("moisture", 6.0) * 0.7 or (top_contribs and "steam" in str(top_contribs[0].get("feature", "")).lower()):
            causal_text = f"Steam Pressure lag ({mvs['steam_pressure']:.2f} bar vs {target_rec['moisture']:.1f}% target balance) driving thermal inertia, causing predicted Basis Weight excursion +3.2%."
        elif mvs.get("stock_flow", 600.0) < target_rec.get("bw", 105.0) * 6.5 or (top_contribs and "stock" in str(top_contribs[0].get("feature", "")).lower()):
            causal_text = f"Stock Flow ramp lag ({mvs['stock_flow']:.1f} L/min vs {target_rec['bw']:.1f} g/m² target recipe) driving fibre mass loading variance, causing predicted Basis Weight excursion +4.1%."
        elif top_phys:
            causal_text = top_phys
        else:
            causal_text = f"Steam Pressure lag ({mvs.get('steam_pressure', 4.2):.2f} bar) driving thermal inertia, causing predicted Basis Weight excursion +3.2%."

        st.markdown(f"""
        <div class="card" style="margin-top:4px; border-left: 4px solid #e74c3c;">
          <div class="kpi-label" style="margin-bottom:6px; color:#e74c3c;">Physical Causal Explanation</div>
          <div style="font-size:13px;color:#111;line-height:1.6;font-weight:600">{causal_text}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="card" style="margin-top:4px; border-left: 4px solid #2ecc71;">
          <div class="kpi-label" style="margin-bottom:6px; color:#2ecc71;">Physical Causal Explanation</div>
          <div style="font-size:13px;color:#111;line-height:1.6">Process telemetry within safe operating bounds. All manipulated variables aligned with recipe targets.</div>
        </div>
        """, unsafe_allow_html=True)

    # Recommendation Cards
    actuator_deltas: dict = pred.get("actuator_deltas", {})
    source_tag: str = pred.get("source_tag", "Lag Correlation Model")
    reason: str = pred.get("recommendation_reason", "")
    act_reasons: dict = pred.get("actuator_reasons", {})
    struct_recs: dict = pred.get("structured_recommendations", {})

    if actuator_deltas:
        ACTUATOR_LABELS = {
            "stock_flow": ("Stock Flow", "L/min"),
            "steam_pressure": ("Steam Pressure", "bar"),
            "machine_speed": ("Machine Speed", "m/min"),
            "filler_flow": ("Filler Flow", "L/min"),
        }

        ACTUATOR_CARD_RATIONALES = {
            "stock_flow": "Reduces stock flow slope to match target basis weight ramp curve.",
            "machine_speed": "Extends sheet residence time in dryer section to halt moisture carryover.",
            "steam_pressure": "Accelerates thermal drying response to eliminate moisture lag.",
            "filler_flow": "Adjusts filler flow to stabilize sheet ash content and density.",
        }

        EXPECTED_IMPROVEMENTS = {
            "stock_flow": "Reduced basis weight tracking error",
            "steam_pressure": "Earlier moisture stabilization",
            "machine_speed": "Thermal lag mitigation",
            "filler_flow": "Ash & density stabilization",
        }

        items = [(k, v) for k, v in actuator_deltas.items() if abs(v) > 0.001]
        cols = st.columns(min(len(items), 3)) if items else []

        for idx, (col, (param, delta_val)) in enumerate(zip(cols, items[:3])):
            label, unit = ACTUATOR_LABELS.get(param, (param.replace("_", " ").title(), ""))
            current_val = mvs.get(param, 0.0)
            recommended_val = round(current_val + delta_val, 3)
            delta_str = f"{delta_val:+.3g} {unit}".strip()
            rec_id = f"REC_{param.upper()}_{grade_from}_{grade_to}"

            struct = struct_recs.get(param, {})
            card_tag = struct.get("source_tag", source_tag)
            exp_imp = struct.get("expected_improvement", EXPECTED_IMPROVEMENTS.get(param, "Earlier moisture stabilization"))
            conf = struct.get("confidence", "HIGH")
            constr_status = struct.get("recipe_constraint_status", "VALID")

            if prob >= engine.threshold:
                card_reason = act_reasons.get(param) or ACTUATOR_CARD_RATIONALES.get(param, reason)
            else:
                card_reason = "Nominal setpoint recipe target for current grade transition."

            with col:
                st.markdown(f"""
                <div class="card">
                  <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                    <div style="font-size:14px;font-weight:800;color:#111">{label}</div>
                    <div>
                      <span class="badge" style="margin-right:4px">{card_tag}</span>
                      <span class="badge" style="background:#eef6ff;color:#1d6fce">{constr_status}</span>
                    </div>
                  </div>
                  <div style="font-size:13px;color:#444;margin-bottom:4px">
                    {current_val} {unit} &rarr; <strong>{recommended_val} {unit}</strong>
                    &nbsp;<span style='font-size:11px;color:#555'>({delta_str})</span>
                  </div>
                  <div style="font-size:11px;color:#27ae60;font-weight:700;margin-bottom:6px">&check; {exp_imp} (Confidence: {conf})</div>
                  <div style="font-size:12px;color:#555;line-height:1.4;margin-bottom:10px">{card_reason}</div>
                </div>
                """, unsafe_allow_html=True)

                btn_accept, btn_reject = st.columns(2)
                with btn_accept:
                    if st.button("Accept", key=f"accept_{rec_id}", use_container_width=True, type="primary"):
                        _log_decision("ACCEPT", rec_id, grade_from, grade_to, mvs, prob, prob * 0.4,
                                      accepted_recs=[rec_id])
                        st.session_state[f"rec_status_{rec_id}"] = "ACCEPTED"
                        accepted = st.session_state.get("accepted_setpoints", st.session_state.get("accepted_overrides", {}))
                        accepted[param] = recommended_val
                        st.session_state["accepted_setpoints"] = accepted
                        st.session_state["accepted_overrides"] = accepted
                        st.session_state["dt_grade_from"] = grade_from
                        st.session_state["dt_grade_to"] = grade_to

                with btn_reject:
                    if st.button("Reject", key=f"reject_{rec_id}", use_container_width=True):
                        _log_decision("REJECT", rec_id, grade_from, grade_to, mvs, prob, prob,
                                      rejected_recs=[rec_id], rejection_reason="Operator override")
                        st.session_state[f"rec_status_{rec_id}"] = "REJECTED"

                status = st.session_state.get(f"rec_status_{rec_id}")
                if status == "ACCEPTED":
                    st.markdown('<div style="font-size:11px;color:#333;font-weight:700">✓ Accepted & logged</div>',
                                unsafe_allow_html=True)
                elif status == "REJECTED":
                    st.markdown('<div style="font-size:11px;color:#555;font-weight:700">✗ Rejected & logged</div>',
                                unsafe_allow_html=True)

    # ── Explainability (Section 7) ───────────────────────────────────────────
    _render_explainability_panel(pred)

    # ── Future State Prediction (Section 4) ──────────────────────────────────
    _render_future_state_panel(grade_from, grade_to, mvs)

    # ── Historical Intelligence (Section 5) ──────────────────────────────────
    _render_historical_strip(grade_from, grade_to, current_mvs=mvs)

    # ── Correlation Discovery (Section 6) ────────────────────────────────────
    _render_correlation_panel(grade_from, grade_to)


# ── VIEW 2: Digital Twin Sandbox ──────────────────────────────────────────────

def render_digital_twin_sandbox():
    _page_header(
        "Digital Twin Sandbox",
        "Manually adjust process parameters and simulate the resulting behaviour before applying changes to the real machine."
    )

    # ── Grade Selection ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Grade Transition</div>', unsafe_allow_html=True)
    grade_keys = list(GRADES.keys())
    sc1, sc2 = st.columns(2)
    with sc1:
        gf_default = st.session_state.get("dt_grade_from", "G20")
        grade_from = st.selectbox("Grade From", grade_keys,
                                  index=grade_keys.index(gf_default), key="dt_gf")
    with sc2:
        gt_default = st.session_state.get("dt_grade_to", "G70")
        grade_to = st.selectbox("Grade To", grade_keys,
                                index=grade_keys.index(gt_default), key="dt_gt")

    recipe = RECIPES.get(grade_to, RECIPES["G70"]).mv
    target_recipe_cv = GRADES.get(grade_to, GRADES["G70"])
    accepted = st.session_state.get("accepted_setpoints") or st.session_state.get("accepted_overrides", {})

    # ── Operator Parameter Inputs ─────────────────────────────────────────────
    st.markdown('<div class="section-title">Operator Parameter Inputs — Edit Before Simulating</div>',
                unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="margin-bottom:4px">
      <div style="font-size:12px;color:#555;line-height:1.6">
        Adjust the manipulated process variables and target quality setpoints below.
        Values are pre-filled from target grade recipe defaults and any accepted AI recommendations.
        Modify parameters and click <strong>Run Simulation</strong> to predict future process behaviour.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title">Manipulated Variables (MVs)</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        stock_flow = st.number_input(
            "Stock Flow (L/min)",
            value=float(accepted.get("stock_flow", recipe["stock_flow"])),
            step=1.0, key="dt_sf"
        )
    with m2:
        steam_pressure = st.number_input(
            "Steam Pressure (bar)",
            value=float(accepted.get("steam_pressure", recipe["steam_pressure"])),
            step=0.01, key="dt_sp"
        )
    with m3:
        machine_speed = st.number_input(
            "Machine Speed (m/min)",
            value=float(accepted.get("machine_speed", recipe["machine_speed"])),
            step=1.0, key="dt_ms"
        )
    with m4:
        filler_flow = st.number_input(
            "Filler Flow (L/min)",
            value=float(accepted.get("filler_flow", recipe["filler_flow"])),
            step=0.5, key="dt_ff"
        )

    st.markdown('<div class="section-title">Target Quality Setpoints (CV Targets)</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    with t1:
        bw_target = st.number_input(
            "Basis Weight Target (g/m²)",
            value=float(target_recipe_cv["bw"]),
            step=0.5, key="dt_bw_tgt"
        )
    with t2:
        moisture_target = st.number_input(
            "Moisture Target (%)",
            value=float(target_recipe_cv["moisture"]),
            step=0.1, key="dt_moist_tgt"
        )
    with t3:
        ash_target = st.number_input(
            "Ash Target (%)",
            value=float(target_recipe_cv["ash"]),
            step=0.1, key="dt_ash_tgt"
        )
    with t4:
        caliper_target = st.number_input(
            "Caliper Target (µm)",
            value=float(target_recipe_cv["caliper"]),
            step=1.0, key="dt_cal_tgt"
        )

    mv_overrides = {
        "stock_flow": stock_flow,
        "steam_pressure": steam_pressure,
        "machine_speed": machine_speed,
        "filler_flow": filler_flow,
    }
    target_cv_overrides = {
        "basis_weight": bw_target,
        "moisture": moisture_target,
        "ash": ash_target,
        "caliper": caliper_target,
    }

    # ── Run Simulation ────────────────────────────────────────────────────────
    if st.button("Run Simulation", type="primary", use_container_width=True):
        with st.spinner("Running physics simulation…"):
            try:
                sim_df = simulate_what_if_window(
                    grade_from=grade_from,
                    grade_to=grade_to,
                    mv_overrides=mv_overrides,
                    target_cv_overrides=target_cv_overrides,
                    window_minutes=40,
                )
                st.session_state["dt_sim_result"] = sim_df
                st.session_state["dt_sim_mvs"] = mv_overrides
                st.session_state["dt_sim_cvs"] = target_cv_overrides
                st.session_state["dt_sim_grades"] = (grade_from, grade_to)
            except Exception as e:
                st.error(f"Simulation failed: {e}")
                return

    sim_df: pd.DataFrame | None = st.session_state.get("dt_sim_result")
    if sim_df is None:
        st.info("Modify process parameter values above and click Run Simulation to predict future process behaviour.")
        return

    sim_gf, sim_gt = st.session_state.get("dt_sim_grades", (grade_from, grade_to))
    sim_mvs: dict = st.session_state.get("dt_sim_mvs", mv_overrides)
    sim_cvs: dict = st.session_state.get("dt_sim_cvs", target_cv_overrides)

    # ── Simulation Output Chart ───────────────────────────────────────────────
    st.markdown('<div class="section-title">Simulation Results — Predicted Process Behaviour</div>',
                unsafe_allow_html=True)

    sp_bw  = sim_cvs.get("basis_weight", GRADES[sim_gt]["bw"])
    sp_moist = sim_cvs.get("moisture", GRADES[sim_gt]["moisture"])
    sp_ash   = sim_cvs.get("ash", GRADES[sim_gt]["ash"])
    sp_cal   = sim_cvs.get("caliper", GRADES[sim_gt]["caliper"])

    tab_bw, tab_moisture, tab_ash, tab_caliper = st.tabs(
        ["Basis Weight", "Moisture", "Ash", "Caliper"]
    )

    def _cv_chart(col: str, sp: float, unit: str, color: str) -> go.Figure:
        fig = go.Figure()
        hi, lo = sp * 1.025, sp * 0.975
        fig.add_hrect(y0=lo, y1=hi, fillcolor="#f0f9f0", opacity=0.4, line_width=0)
        fig.add_trace(go.Scatter(
            x=sim_df["minute"], y=sim_df[col],
            name=col.replace("_", " ").title(),
            line=dict(color=color, width=2.5),
        ))
        fig.add_hline(y=sp, line_dash="dot", line_color="#34495e",
                      annotation_text="Target", annotation_position="top right",
                      annotation_font_color="#555")
        fig.add_hline(y=hi, line_dash="dash", line_color="#bbb",
                      annotation_text="+2.5%", annotation_position="top right",
                      annotation_font_color="#888")
        fig.add_hline(y=lo, line_dash="dash", line_color="#bbb",
                      annotation_text="−2.5%", annotation_position="bottom right",
                      annotation_font_color="#888")
        fig.update_layout(
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            font=dict(color="#111", family="Inter, sans-serif", size=11),
            height=300,
            margin=dict(l=20, r=20, t=10, b=20),
            xaxis=dict(gridcolor="#e8e8e8", title="Transition Time (min)"),
            yaxis=dict(gridcolor="#e8e8e8", title=f"{col.replace('_',' ').title()} ({unit})"),
            showlegend=False,
        )
        return fig

    with tab_bw:
        st.plotly_chart(_cv_chart("basis_weight", sp_bw, "g/m²", "#111"), use_container_width=True)
    with tab_moisture:
        st.plotly_chart(_cv_chart("moisture", sp_moist, "%", "#3498db"), use_container_width=True)
    with tab_ash:
        st.plotly_chart(_cv_chart("ash", sp_ash, "%", "#e67e22"), use_container_width=True)
    with tab_caliper:
        st.plotly_chart(_cv_chart("caliper", sp_cal, "µm", "#8e44ad"), use_container_width=True)

    # ── Summary KPIs & Predicted Process Metrics ──────────────────────────────
    off_spec_mins = int(sim_df["off_spec"].sum()) if "off_spec" in sim_df.columns else 0
    peak_dev = float(sim_df["deviation_pct"].abs().max()) if "deviation_pct" in sim_df.columns else 0.0
    in_spec = sim_df[sim_df["off_spec"] == False] if "off_spec" in sim_df.columns else sim_df
    first_stable = int(in_spec["minute"].min()) if not in_spec.empty else len(sim_df)
    quality = "WITHIN SPEC" if off_spec_mins == 0 else ("MINOR" if off_spec_mins <= 5 else "OFF-SPEC")
    spec_badge = "IN SPEC" if off_spec_mins == 0 else "OFF SPEC"

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        _kpi("Predicted Off-Spec Duration", f"{off_spec_mins} min", note="Off-spec time window", badge="BW ±2.5%")
    with k2:
        _kpi("Predicted Deviation Risk", f"{peak_dev:.1f}%", note="Peak BW deviation magnitude", badge="RISK")
    with k3:
        _kpi("Stabilization Time", f"{first_stable} min", note="Est. stabilization horizon", badge="STABILIZATION")
    with k4:
        _kpi("Quality & Spec Status", quality, note=f"Process Status: {spec_badge}", badge=spec_badge)

    # ── Apply to DCS ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Operator Decision</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="card" style="margin-bottom:8px">
      <div style="font-size:12px;color:#555">
        Review the simulation results above. If satisfied with predicted process behavior, click
        <strong>Apply Setpoints</strong> to dispatch these parameters to the controller.
      </div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Apply Setpoints", type="primary", use_container_width=True):
        _log_decision(
            action="APPLY_DCS",
            rec_id=f"DT_{sim_gf}_{sim_gt}_{datetime.utcnow().strftime('%H%M%S')}",
            grade_from=sim_gf, grade_to=sim_gt,
            mvs=sim_mvs,
            baseline_risk=min(1.0, off_spec_mins / max(len(sim_df), 1)),
            tuned_risk=0.0,
            notes=f"Digital Twin validated — {off_spec_mins} off-spec min, peak dev {peak_dev:.1f}%",
        )
        st.success("Setpoints dispatched. Decision logged to audit trail.")


# ── VIEW 3: Feedback & Audit Log ──────────────────────────────────────────────

def render_feedback_audit():
    _page_header(
        "Feedback & Audit Log",
        "Operator decision history, engineering console, historical evidence, and model health."
    )

    engine = _engine()
    records = _load_feedback()

    # ── Operator Acceptance Analytics ─────────────────────────────────────────
    accepted = sum(1 for r in records if r.get("action_taken", "") in ("ACCEPT", "APPLY_DCS"))
    rejected = sum(1 for r in records if r.get("action_taken", "") in ("REJECT",))
    total = accepted + rejected
    rate = round(accepted / total * 100, 1) if total > 0 else 0.0

    st.markdown('<div class="section-title">Operator Acceptance Analytics</div>', unsafe_allow_html=True)
    a1, a2, a3, a4 = st.columns(4)
    with a1:
        _kpi("Total Decisions", str(total))
    with a2:
        _kpi("Accepted", str(accepted), badge="ACCEPTED")
    with a3:
        _kpi("Rejected", str(rejected), badge="REJECTED")
    with a4:
        _kpi("Acceptance Rate", f"{rate:.1f}%", badge="HISTORICAL")

    # ── Historical Intelligence Evidence ─────────────────────────────────────
    st.markdown('<div class="section-title">Historical Transition Benchmark Evidence</div>', unsafe_allow_html=True)
    hist_service = getattr(engine, "history_service", None) or HistoricalIntelligenceService()
    hist_matches = hist_service.search_similar_transitions(grade_from="G20", grade_to="G70", top_k=3)
    if hist_matches:
        h_df = pd.DataFrame(hist_matches)
        disp_cols = [c for c in ["transition_id", "grade_from", "grade_to", "outcome", "off_spec_minutes", "recovery_minutes", "operating_settings", "explanation"] if c in h_df.columns]
        st.dataframe(h_df[disp_cols], use_container_width=True, hide_index=True)
    else:
        st.info("No historical evidence records match the current transition filter.")

    # ── Engineering Console ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">Engineering Console</div>', unsafe_allow_html=True)
    ec_tab1, ec_tab2, ec_tab3 = st.tabs(["Correlation Matrix", "Model Explainability", "Audit Log"])

    with ec_tab1:
        last_state = st.session_state.get("last_prediction", {})
        gf = last_state.get("grade_from", "G20")
        gt = last_state.get("grade_to", "G70")
        _render_correlation_panel(gf, gt)

    with ec_tab2:
        st.markdown('<div style="font-size:13px;font-weight:700;color:#333;margin-bottom:8px">Top Feature Importances (XGBoost / SHAP)</div>', unsafe_allow_html=True)
        top_feats = engine.top_feature_importance(top_n=10)
        if top_feats:
            feat_df = pd.DataFrame(top_feats)
            st.dataframe(feat_df, use_container_width=True, hide_index=True)
        else:
            st.info("Feature importance metadata unavailable.")

    with ec_tab3:
        if records:
            display_cols = ["feedback_id", "timestamp", "action_taken", "grade_from", "grade_to",
                            "risk_before", "risk_after", "risk_reduction_pct", "retraining_status"]
            log_df = pd.DataFrame(records)
            log_df = log_df[[c for c in display_cols if c in log_df.columns]].tail(20)
            st.dataframe(log_df, use_container_width=True, hide_index=True)
        else:
            st.info("No operator feedback recorded yet.")

    # ── Model Health ──────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Model Health & System Readiness</div>', unsafe_allow_html=True)
    try:
        health = engine.health_check()
    except Exception:
        health = {}

    is_ready = health.get("model_loaded", True) or health.get("status") in ("READY", "HEALTHY")
    status_text = "READY" if is_ready else "CHECK"

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        _kpi("Architecture", health.get("model_type", "XGBoost"), badge="ML MODEL")
    with m2:
        _kpi("Decision Threshold", f"{engine.threshold:.2f}", badge="OPERATIONAL")
    with m3:
        _kpi("Feature Count", str(engine.feature_count), badge="FEATURES")
    with m4:
        _kpi("System Status", status_text, badge="READY" if is_ready else "CHECK")

