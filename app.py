import os
import sys
import subprocess
import tempfile

import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# Add src to path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from generate_features import build_unified_table
from llm_insights import generate_causal_summary

st.set_page_config(
    page_title="NetElixir Revenue Forecaster",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Probabilistic Revenue & ROAS Forecaster")
st.caption("NetElixir AIgnition 2026 — AI-Assisted E-commerce Forecasting Utility")

# ── Sidebar controls ──────────────────────────────────────────────────────────
st.sidebar.header("Forecast Settings")

horizon = st.sidebar.selectbox(
    "Forecast Horizon",
    options=[30, 60, 90],
    index=0,
    help="Number of days to forecast into the future"
)

st.sidebar.subheader("Future Daily Budget (optional)")
st.sidebar.caption("Leave at 0 to use historical average spend per channel.")
google_budget = st.sidebar.number_input("Google Ads daily budget ($)", min_value=0.0, value=0.0, step=50.0)
meta_budget = st.sidebar.number_input("Meta Ads daily budget ($)", min_value=0.0, value=0.0, step=50.0)
bing_budget = st.sidebar.number_input("Bing Ads daily budget ($)", min_value=0.0, value=0.0, step=10.0)

run_button = st.sidebar.button("🚀 Generate Forecast", type="primary")

# ── Main area ─────────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "pickle", "model.pkl")
FEATURES_PATH = os.path.join(os.path.dirname(__file__), "features.parquet")
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "output", "predictions.csv")

if not run_button:
    st.info("👈 Set your forecast horizon and optional budget overrides in the sidebar, then click **Generate Forecast**.")
    st.markdown("""
    ### What this tool does
    - Ingests your Google Ads, Meta Ads, and Bing Ads campaign data
    - Fits a **Prophet time-series model** per channel with yearly + weekly seasonality
    - Generates **probabilistic revenue and ROAS ranges** (P10/P50/P90) using bootstrapped residuals
    - Simulates the impact of **different media budgets** on expected revenue
    - Produces **AI-generated causal insights** explaining what's driving the forecast
    """)
else:
    with st.spinner("Running feature generation..."):
        try:
            unified = build_unified_table(DATA_DIR)
            unified.to_parquet(FEATURES_PATH, index=False)
        except Exception as e:
            st.error(f"Feature generation failed: {e}")
            st.stop()

    # Build budget string for predict.py
    budget_parts = []
    if google_budget > 0:
        budget_parts.append(f"google={google_budget}")
    if meta_budget > 0:
        budget_parts.append(f"meta={meta_budget}")
    if bing_budget > 0:
        budget_parts.append(f"bing={bing_budget}")
    budget_arg = ",".join(budget_parts) if budget_parts else None

    with st.spinner("Generating probabilistic forecast..."):
        cmd = [
            sys.executable, "src/predict.py",
            "--features", FEATURES_PATH,
            "--model", MODEL_PATH,
            "--output", OUTPUT_PATH,
            "--horizon", str(horizon),
        ]
        if budget_arg:
            cmd += ["--budget", budget_arg]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            st.error(f"Prediction failed:\n{result.stderr}")
            st.stop()

    predictions = pd.read_csv(OUTPUT_PATH)

    # ── Summary metrics ───────────────────────────────────────────────────────
    total = predictions[predictions["level"] == "total"].iloc[0]
    st.success(f"✅ Forecast generated for next **{horizon} days**")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Revenue (Likely)", f"${total['revenue_p50']:,.0f}")
    col2.metric("Revenue Range", f"${total['revenue_p10']:,.0f} – ${total['revenue_p90']:,.0f}")
    col3.metric("Blended ROAS (Likely)", f"{total['roas_p50']:.2f}x")
    col4.metric("Total Spend", f"${total['spend_total']:,.0f}")

    st.divider()

    # ── Channel breakdown chart ───────────────────────────────────────────────
    st.subheader("Channel Revenue Forecast (P10 / P50 / P90)")
    channels = predictions[predictions["level"] == "channel"].copy()

    fig = go.Figure()
    colors = {"google": "#4285F4", "meta": "#1877F2", "bing": "#00809D"}

    for _, row in channels.iterrows():
        ch = row["channel"]
        color = colors.get(ch, "#888888")
        fig.add_trace(go.Bar(
            name=f"{ch.title()} P50",
            x=[ch.title()],
            y=[row["revenue_p50"]],
            marker_color=color,
            error_y=dict(
                type="data",
                symmetric=False,
                array=[row["revenue_p90"] - row["revenue_p50"]],
                arrayminus=[row["revenue_p50"] - row["revenue_p10"]],
            ),
        ))

    fig.update_layout(
        yaxis_title="Revenue ($)",
        showlegend=False,
        height=400,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Campaign type table ───────────────────────────────────────────────────
    st.subheader("Campaign Type Breakdown")
    type_rows = predictions[predictions["level"] == "campaign_type"].copy()
    type_rows = type_rows.sort_values("revenue_p50", ascending=False)
    type_display = type_rows[[
        "channel", "campaign_type",
        "revenue_p10", "revenue_p50", "revenue_p90",
        "spend_total", "roas_p50"
    ]].copy()
    type_display.columns = [
        "Channel", "Campaign Type",
        "Revenue P10 ($)", "Revenue P50 ($)", "Revenue P90 ($)",
        "Spend ($)", "ROAS (likely)"
    ]
    for col in ["Revenue P10 ($)", "Revenue P50 ($)", "Revenue P90 ($)", "Spend ($)"]:
        type_display[col] = type_display[col].apply(lambda x: f"${x:,.0f}")
    type_display["ROAS (likely)"] = type_display["ROAS (likely)"].apply(lambda x: f"{x:.2f}x")
    st.dataframe(type_display, use_container_width=True, hide_index=True)

    # ── AI Insights ───────────────────────────────────────────────────────────
    st.subheader("🤖 AI-Generated Business Insights")
    with st.spinner("Generating AI insights..."):
        insight = generate_causal_summary(predictions)
    st.info(insight)

    # ── Download ──────────────────────────────────────────────────────────────
    st.divider()
    csv_data = predictions.to_csv(index=False)
    st.download_button(
        label="⬇️ Download Full Forecast CSV",
        data=csv_data,
        file_name=f"forecast_{horizon}d.csv",
        mime="text/csv",
    )
