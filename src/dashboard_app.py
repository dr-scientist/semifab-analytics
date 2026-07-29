"""Streamlit dashboard prototype for SemiFab Analytics."""
from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from semifab_db import query_wafer_data
from semifab_analytics import calculate_kpis, calculate_spc_limits, generate_risk_alerts, pareto_failure_patterns

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "semifab_unit6.sqlite"

st.set_page_config(page_title="SemiFab Analytics", layout="wide")
st.title("SemiFab Analytics: Process, Defect, Yield, SPC, and Risk Dashboard")

df = query_wafer_data(DB_PATH)
spc = calculate_spc_limits(df, "defect_density")
alerts = generate_risk_alerts(df, spc)
kpis = calculate_kpis(alerts)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Wafers", kpis["total_wafers"])
c2.metric("Average Yield", f"{kpis['avg_yield_percent']}%")
c3.metric("Pass Rate", f"{kpis['pass_rate_percent']}%")
c4.metric("Requires Review", int((alerts["risk_status"] == "requires review").sum()))

left, right = st.columns(2)
with left:
    st.subheader("Yield Loss by Failure Pattern")
    st.plotly_chart(px.bar(pareto_failure_patterns(alerts), x="failure_pattern", y="yield_loss"), use_container_width=True)
with right:
    st.subheader("Defect Density by Tool")
    st.plotly_chart(px.box(alerts, x="tool_id", y="defect_density", color="tool_id"), use_container_width=True)

st.subheader("Lot-Level Risk Table")
st.dataframe(alerts.loc[alerts["risk_status"] == "requires review", ["wafer_id", "lot_id", "tool_id", "failure_pattern", "contamination_score", "defect_density", "yield_percent", "risk_status"]].head(25))
