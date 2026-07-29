"""Synthetic data generation for SemiFab Analytics Unit 6 integration."""
from __future__ import annotations
import numpy as np
import pandas as pd

FAILURE_PATTERNS = ["None", "Center", "Edge-Loc", "Edge-Ring", "Loc", "Near-full", "Random", "Scratch", "Donut"]
DEFECT_TYPES = ["Particle", "Misalignment", "Etch Residue", "Scratch", "Pattern Collapse"]
TOOLS = ["LITHO-01", "LITHO-02", "LITHO-03", "LITHO-04"]

def generate_wafer_data(n: int = 360, seed: int = 42) -> pd.DataFrame:
    """Generate realistic simulated wafer, lithography, defect, and yield data."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-01-01", periods=180, freq="D")
    df = pd.DataFrame({
        "wafer_id": [f"W{i:04d}" for i in range(1, n + 1)],
        "lot_id": rng.choice([f"LOT-{i:02d}" for i in range(1, 31)], n),
        "tool_id": rng.choice(TOOLS, n, p=[0.28, 0.25, 0.27, 0.20]),
        "process_date": rng.choice(dates, n),
        "exposure_dose": rng.normal(21.0, 0.55, n),
        "focus_offset": rng.normal(0.0, 0.18, n),
        "overlay_error": np.abs(rng.normal(2.2, 0.70, n)),
        "cd_mean": rng.normal(45.0, 1.2, n),
        "cd_uniformity": np.abs(rng.normal(1.5, 0.35, n)),
        "resist_thickness": rng.normal(102.0, 3.8, n),
        "particle_count": rng.poisson(7, n),
        "contamination_score": np.clip(rng.gamma(2.2, 0.9, n), 0, 8),
        "die_size": rng.choice([515, 530, 560, 600, 710, 740], n, p=[0.28,0.18,0.16,0.13,0.12,0.13]),
        "defect_type": rng.choice(DEFECT_TYPES, n, p=[0.32,0.26,0.21,0.12,0.09]),
        "failure_pattern": rng.choice(FAILURE_PATTERNS, n, p=[0.45,0.14,0.11,0.08,0.08,0.04,0.06,0.03,0.01])
    })
    pattern_penalty = df["failure_pattern"].map({"None":0,"Center":5.5,"Edge-Loc":4.0,"Edge-Ring":3.2,"Loc":3.8,"Near-full":22.0,"Random":7.0,"Scratch":4.8,"Donut":3.5}).astype(float)
    defect_density = (
        0.35 + 0.10 * df["particle_count"] + 0.18 * df["contamination_score"] +
        0.20 * df["overlay_error"] + 0.30 * np.abs(df["focus_offset"]) + pattern_penalty / 18.0 +
        rng.normal(0, 0.17, n)
    )
    df["defect_density"] = np.clip(defect_density, 0.05, None).round(3)
    yield_percent = 97.5 - 5.4 * df["defect_density"] - 0.75 * df["contamination_score"] - 0.35 * df["overlay_error"] - pattern_penalty + rng.normal(0, 2.3, n)
    df["yield_percent"] = np.clip(yield_percent, 0, 99.5).round(2)
    df["pass_fail"] = np.where(df["yield_percent"] >= 85, "pass", "fail")
    df["yield_loss"] = (100 - df["yield_percent"]).round(2)
    return df
