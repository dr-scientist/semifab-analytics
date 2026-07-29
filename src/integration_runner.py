"""End-to-end integration runner connecting data, database, analytics, and dashboard-ready outputs."""
from __future__ import annotations
import json
import time
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

from semifab_data import generate_wafer_data
from semifab_db import load_dataframe, query_wafer_data
from semifab_analytics import calculate_kpis, calculate_spc_limits, train_quality_model, generate_risk_alerts, pareto_failure_patterns, qualitative_usability_score

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "database" / "semifab_unit6.sqlite"
OUT = ROOT / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)

def run_pipeline() -> dict:
    t0 = time.perf_counter()
    df_in = generate_wafer_data(n=360, seed=42)
    load_dataframe(df_in, DB_PATH)
    df = query_wafer_data(DB_PATH)
    db_rows = len(df)
    spc = calculate_spc_limits(df, "defect_density")
    kpis = calculate_kpis(df)
    model_result = train_quality_model(df)
    alerts = generate_risk_alerts(df, spc)
    pareto = pareto_failure_patterns(alerts)
    usability = qualitative_usability_score()
    integration_latency_ms = round((time.perf_counter() - t0) * 1000, 2)

    # Save dashboard-ready data
    alerts.to_csv(ROOT / "data" / "dashboard_ready_wafer_alerts.csv", index=False)
    pareto.to_csv(ROOT / "data" / "failure_pattern_pareto.csv", index=False)

    # Chart: failure pattern Pareto
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.bar(pareto["failure_pattern"], pareto["yield_loss"])
    ax.set_title("Yield Loss by Wafer Failure Pattern")
    ax.set_ylabel("Total Yield Loss")
    ax.set_xlabel("Failure Pattern")
    ax.tick_params(axis='x', rotation=30)
    fig.tight_layout()
    fig.savefig(OUT / "unit6_failure_pattern_pareto.png", dpi=180)
    plt.close(fig)

    # Chart: monthly yield trend
    monthly = alerts.assign(month=alerts["process_date"].dt.to_period("M").astype(str)).groupby("month").agg(total_wafers=("wafer_id","count"), avg_yield=("yield_percent","mean")).reset_index()
    fig, ax = plt.subplots(figsize=(8,4.5))
    ax.plot(monthly["month"], monthly["avg_yield"], marker="o")
    ax.set_title("Average Yield Over Time")
    ax.set_ylabel("Average Yield (%)")
    ax.set_xlabel("Process Month")
    ax.tick_params(axis='x', rotation=30)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "unit6_yield_trend.png", dpi=180)
    plt.close(fig)

    results = {
        "input_rows": len(df_in),
        "database_rows_after_load": db_rows,
        "kpis": kpis,
        "spc": spc,
        "model_accuracy": model_result["accuracy"],
        "model_latency_ms": model_result["latency_ms"],
        "integration_latency_ms": integration_latency_ms,
        "requires_review_count": int((alerts["risk_status"] == "requires review").sum()),
        "usability": usability,
        "top_failure_pattern": pareto.iloc[0].to_dict() if len(pareto) else {},
        "top_features": model_result["feature_importance"].head(5).to_dict("records")
    }
    with open(OUT / "unit6_integration_output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    return results

if __name__ == "__main__":
    results = run_pipeline()
    print("SemiFab Analytics Unit 6 Integrated System Output")
    print(json.dumps(results, indent=2))
