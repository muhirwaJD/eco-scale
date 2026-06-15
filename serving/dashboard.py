"""
dashboard.py — Streamlit dashboard for the Eco-Scale autoscaler.

Two panels:
  1. Live decision — set the cluster state with sliders and see the agent's
     scaling action in real time.
  2. Results — the headline figures and the PPO-vs-HPA comparison table.

Run:
    streamlit run serving/dashboard.py
"""

import os
import sys
import pandas as pd
import streamlit as st

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from serving.inference_engine import InferenceEngine

OUTPUTS = os.path.join(ROOT, "outputs")

st.set_page_config(page_title="Eco-Scale Autoscaler", page_icon="📈", layout="wide")


@st.cache_resource
def get_engine():
    return InferenceEngine()


engine = get_engine()

st.title("📈 Eco-Scale — RL Kubernetes Autoscaler")
st.caption(f"Deployed agent: **{engine.algorithm}** (run {engine.metadata.get('run')}) "
           f"· trained on real Alibaba cluster traces")

tab_live, tab_results = st.tabs(["Live decision", "Results"])

# ── Panel 1: live scaling decision ──────────────────────────────────
with tab_live:
    st.subheader("Current cluster state")
    col1, col2 = st.columns(2)
    with col1:
        cpu_util = st.slider("CPU utilization", 0.0, 1.0, 0.5, 0.01)
        pods = st.slider("Current pods", 1, 20, 5)
    with col2:
        queue_depth = st.slider("Request queue depth", 0, 1000, 300, 10)
        day_progress = st.slider("Time of day (fraction)", 0.0, 1.0, 0.5, 0.01)

    decision = engine.decide(cpu_util, pods, queue_depth, day_progress)
    emoji = {"scale_up": "⬆️", "maintain": "⏸️", "scale_down": "⬇️"}[decision["action_name"]]

    st.subheader("Agent decision")
    st.metric("Recommended action", f"{emoji} {decision['action_name'].replace('_', ' ').title()}")
    st.caption("The agent reacts to current load, pod count, queue, and time of day.")

# ── Panel 2: results ────────────────────────────────────────────────
with tab_results:
    st.subheader("RL vs Kubernetes HPA (held-out test traces)")
    results_csv = os.path.join(OUTPUTS, "hpa_comparison", "dqn_vs_hpa_results.csv")
    if os.path.exists(results_csv):
        st.dataframe(pd.read_csv(results_csv), width="stretch")

    for caption, fname in [
        ("PPO vs HPA — reward / latency / energy", "hpa_comparison/dqn_vs_hpa_comparison.png"),
        ("Energy vs reliability frontier", "hpa_comparison/energy_frontier.png"),
        ("Champion selection (per-run reward)", "training/champion_mean_reward.png"),
    ]:
        path = os.path.join(OUTPUTS, fname)
        if os.path.exists(path):
            st.markdown(f"**{caption}**")
            st.image(path, width="stretch")
