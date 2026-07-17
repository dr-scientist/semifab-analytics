"""
SemiFab Analytics core logic for MSIT 5910 Unit 5.
This module simulates lithography/wafer process data, calculates KPI and SPC metrics,
and trains a small pass/fail classifier for process-quality prediction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split


FEATURE_COLUMNS = [
    "exposure_dose",
    "focus_offset",
    "overlay_error",
    "cd_uniformity",
    "resist_thickness",
    "contamination_score",
    "particle_count",
    "defect_density",
]


@dataclass
class ModelResult:
    accuracy: float
    confusion_matrix: np.ndarray
    feature_importance: pd.DataFrame
    predictions: np.ndarray


def generate_wafer_data(n: int = 240, seed: int = 42) -> pd.DataFrame:
    """Generate realistic simulated lithography and wafer-quality data."""
    if n <= 0:
        raise ValueError("n must be positive")

    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "wafer_id": [f"W{idx:04d}" for idx in range(1, n + 1)],
        "lot_id": rng.choice(["LOT-A", "LOT-B", "LOT-C", "LOT-D"], size=n),
        "tool_id": rng.choice(["LITHO-01", "LITHO-02", "LITHO-03"], size=n),
        "exposure_dose": rng.normal(25.0, 1.2, n),
        "focus_offset": rng.normal(0.0, 0.07, n),
        "overlay_error": rng.normal(4.5, 1.1, n),
        "cd_uniformity": rng.normal(1.6, 0.35, n),
        "resist_thickness": rng.normal(110.0, 4.0, n),
        "contamination_score": rng.gamma(2.0, 0.7, n),
        "particle_count": rng.poisson(12, n),
    })

    # Defect density increases when focus, overlay, contamination, and particle count worsen.
    df["defect_density"] = (
        0.15
        + 0.10 * np.abs(df["focus_offset"]) * 10
        + 0.07 * df["overlay_error"]
        + 0.12 * df["contamination_score"]
        + 0.018 * df["particle_count"]
        + rng.normal(0, 0.12, n)
    ).clip(lower=0.02)

    df["yield_percent"] = (
        98
        - 8.5 * df["defect_density"]
        - 1.1 * df["contamination_score"]
        - 0.12 * df["particle_count"]
        + rng.normal(0, 1.2, n)
    ).clip(lower=55, upper=99.8)

    df["pass_fail"] = np.where(
        (df["yield_percent"] >= 88.0) & (df["defect_density"] <= 1.35),
        "PASS",
        "FAIL",
    )
    return df.round(3)


def calculate_kpis(df: pd.DataFrame) -> Dict[str, float]:
    """Calculate dashboard-level process KPIs from wafer data."""
    _validate_columns(df, ["yield_percent", "defect_density", "contamination_score", "pass_fail"])
    return {
        "total_wafers": int(len(df)),
        "pass_rate_percent": round(float((df["pass_fail"] == "PASS").mean() * 100), 2),
        "average_yield_percent": round(float(df["yield_percent"].mean()), 2),
        "average_defect_density": round(float(df["defect_density"].mean()), 3),
        "high_contamination_wafers": int((df["contamination_score"] > 2.5).sum()),
    }


def calculate_spc_limits(df: pd.DataFrame, column: str) -> Dict[str, float]:
    """Calculate mean, UCL, and LCL for a process variable using three-sigma limits."""
    _validate_columns(df, [column])
    mean = float(df[column].mean())
    sigma = float(df[column].std(ddof=1))
    return {
        "metric": column,
        "mean": round(mean, 3),
        "ucl": round(mean + 3 * sigma, 3),
        "lcl": round(mean - 3 * sigma, 3),
        "out_of_control_count": int(((df[column] > mean + 3 * sigma) | (df[column] < mean - 3 * sigma)).sum()),
    }


def train_quality_model(df: pd.DataFrame, seed: int = 42) -> ModelResult:
    """Train and evaluate a Random Forest pass/fail model."""
    _validate_columns(df, FEATURE_COLUMNS + ["pass_fail"])
    X = df[FEATURE_COLUMNS]
    y = (df["pass_fail"] == "PASS").astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=seed)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    importance = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    return ModelResult(
        accuracy=round(float(accuracy_score(y_test, predictions)), 3),
        confusion_matrix=confusion_matrix(y_test, predictions),
        feature_importance=importance.round(3),
        predictions=predictions,
    )


def contamination_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Flag wafers needing human review based on contamination and defect risk."""
    _validate_columns(df, ["wafer_id", "lot_id", "tool_id", "contamination_score", "defect_density", "yield_percent"])
    alerts = df[(df["contamination_score"] > 2.5) | (df["defect_density"] > 1.35)].copy()
    alerts["risk_status"] = "requires review"
    return alerts[["wafer_id", "lot_id", "tool_id", "contamination_score", "defect_density", "yield_percent", "risk_status"]]


def run_pipeline(n: int = 240, seed: int = 42) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float], ModelResult, pd.DataFrame]:
    """Run the complete core pipeline used by the Unit 5 report."""
    df = generate_wafer_data(n=n, seed=seed)
    kpis = calculate_kpis(df)
    spc = calculate_spc_limits(df, "defect_density")
    model_result = train_quality_model(df, seed=seed)
    alerts = contamination_alerts(df)
    return df, kpis, spc, model_result, alerts


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


if __name__ == "__main__":
    data, kpi_result, spc_result, result, alert_table = run_pipeline()
    print("SemiFab Analytics - Core Logic Output")
    print("KPI summary:", kpi_result)
    print("SPC limits for defect_density:", spc_result)
    print("Model accuracy:", result.accuracy)
    print("Confusion matrix:\n", result.confusion_matrix)
    print("Top feature importance:\n", result.feature_importance.head())
    print("First 5 risk alerts:\n", alert_table.head())
