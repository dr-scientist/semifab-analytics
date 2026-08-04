import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from semifab_data import generate_sample_data
from semifab_analytics import calculate_kpis, calculate_spc_limits, train_model


def test_data_generation_has_expected_columns():
    df = generate_sample_data(120)

    expected_columns = {
        "wafer_id",
        "lot_id",
        "tool_id",
        "exposure_dose",
        "focus_offset",
        "overlay_error",
        "cd_mean",
        "cd_uniformity",
        "resist_thickness",
        "contamination_score",
        "particle_count",
        "defect_density",
        "yield_percent",
        "pass_fail",
        "risk_status",
        "failure_pattern",
    }

    assert len(df) == 120
    assert expected_columns.issubset(set(df.columns))


def test_kpi_calculation_returns_valid_values():
    df = generate_sample_data(120)
    kpis = calculate_kpis(df)

    assert kpis["total_wafers"] == 120
    assert 0 <= kpis["avg_yield_percent"] <= 100
    assert 0 <= kpis["pass_rate_percent"] <= 100
    assert kpis["avg_defects_per_wafer"] >= 0


def test_spc_limits_are_valid():
    df = generate_sample_data(120)
    spc = calculate_spc_limits(df, "defect_density")

    assert spc["metric"] == "defect_density"
    assert spc["ucl"] > spc["mean"]
    assert spc["lcl"] < spc["mean"]
    assert spc["out_of_control_count"] >= 0


def test_model_training_returns_accuracy_and_features():
    df = generate_sample_data(200)
    results = train_model(df)

    assert 0 <= results["accuracy"] <= 1
    assert "feature_importance" in results
    assert len(results["feature_importance"]) > 0


def test_requires_review_alerts_exist():
    df = generate_sample_data(200)

    assert "risk_status" in df.columns
    assert df["risk_status"].isin(["normal", "requires review"]).all()
