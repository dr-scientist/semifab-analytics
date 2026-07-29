"""Analytics, SPC, model evaluation, and dashboard-preparation functions."""
from __future__ import annotations
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["exposure_dose","focus_offset","overlay_error","cd_mean","cd_uniformity","resist_thickness","particle_count","contamination_score","defect_density","die_size"]

def calculate_kpis(df: pd.DataFrame) -> dict:
    return {
        "total_wafers": int(len(df)),
        "avg_yield_percent": round(float(df["yield_percent"].mean()), 2),
        "pass_rate_percent": round(float((df["pass_fail"] == "pass").mean() * 100), 2),
        "total_defects_est": int(round(df["defect_density"].sum() * 10)),
        "avg_defects_per_wafer": round(float(df["defect_density"].mean()), 3),
        "high_contamination_wafers": int((df["contamination_score"] > 3.5).sum())
    }

def calculate_spc_limits(df: pd.DataFrame, metric: str = "defect_density") -> dict:
    mean = float(df[metric].mean())
    sigma = float(df[metric].std(ddof=1))
    ucl = mean + 3 * sigma
    lcl = max(0.0, mean - 3 * sigma)
    out = int(((df[metric] > ucl) | (df[metric] < lcl)).sum())
    return {"metric": metric, "mean": round(mean, 3), "ucl": round(ucl, 3), "lcl": round(lcl, 3), "out_of_control_count": out}

def train_quality_model(df: pd.DataFrame, seed: int = 42) -> dict:
    X = df[FEATURES]
    y = (df["pass_fail"] == "pass").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    start = time.perf_counter()
    model = Pipeline([("scale", StandardScaler()), ("rf", RandomForestClassifier(n_estimators=120, random_state=seed, class_weight="balanced"))])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    latency_ms = (time.perf_counter() - start) * 1000
    accuracy = accuracy_score(y_test, predictions)
    cm = confusion_matrix(y_test, predictions)
    rf = model.named_steps["rf"]
    importance = pd.DataFrame({"feature": FEATURES, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
    return {"model": model, "accuracy": round(float(accuracy), 3), "confusion_matrix": cm, "feature_importance": importance, "latency_ms": round(latency_ms, 2)}

def generate_risk_alerts(df: pd.DataFrame, spc: dict) -> pd.DataFrame:
    alerts = df.copy()
    alerts["risk_status"] = np.where(
        (alerts["defect_density"] > spc["ucl"]) | (alerts["yield_percent"] < 82) | (alerts["contamination_score"] > 4.0),
        "requires review", "normal"
    )
    return alerts

def pareto_failure_patterns(df: pd.DataFrame) -> pd.DataFrame:
    p = df[df["failure_pattern"] != "None"].groupby("failure_pattern", as_index=False)["yield_loss"].sum()
    p = p.sort_values("yield_loss", ascending=False)
    p["cum_percent"] = (p["yield_loss"].cumsum() / p["yield_loss"].sum() * 100).round(2)
    return p

def qualitative_usability_score() -> dict:
    return {"navigation_clarity": 4, "chart_readability": 4, "alert_explainability": 5, "overall_usability": 4.33, "scale": "1=poor, 5=excellent"}
