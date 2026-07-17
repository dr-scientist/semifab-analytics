import pandas as pd
import pytest

from src.semifab_core import (
    FEATURE_COLUMNS,
    calculate_kpis,
    calculate_spc_limits,
    contamination_alerts,
    generate_wafer_data,
    train_quality_model,
)


def test_generate_wafer_data_has_required_shape_and_columns():
    df = generate_wafer_data(n=50, seed=7)
    assert len(df) == 50
    for column in FEATURE_COLUMNS + ["wafer_id", "lot_id", "tool_id", "yield_percent", "pass_fail"]:
        assert column in df.columns
    assert set(df["pass_fail"].unique()).issubset({"PASS", "FAIL"})


def test_kpi_calculation_returns_expected_business_metrics():
    df = generate_wafer_data(n=80, seed=9)
    kpis = calculate_kpis(df)
    assert kpis["total_wafers"] == 80
    assert 0 <= kpis["pass_rate_percent"] <= 100
    assert 55 <= kpis["average_yield_percent"] <= 100
    assert kpis["average_defect_density"] > 0


def test_spc_limits_contain_control_boundaries():
    df = generate_wafer_data(n=100, seed=11)
    limits = calculate_spc_limits(df, "defect_density")
    assert limits["ucl"] > limits["mean"] > limits["lcl"]
    assert limits["out_of_control_count"] >= 0


def test_model_training_returns_accuracy_and_confusion_matrix():
    df = generate_wafer_data(n=220, seed=12)
    result = train_quality_model(df)
    assert result.accuracy >= 0.70
    assert result.confusion_matrix.shape == (2, 2)
    assert not result.feature_importance.empty


def test_contamination_alerts_are_review_based_not_automatic_failure():
    df = generate_wafer_data(n=120, seed=14)
    alerts = contamination_alerts(df)
    assert "risk_status" in alerts.columns
    if not alerts.empty:
        assert set(alerts["risk_status"]) == {"requires review"}


def test_missing_column_raises_error_for_white_box_validation():
    df = pd.DataFrame({"yield_percent": [90.0], "pass_fail": ["PASS"]})
    with pytest.raises(ValueError):
        calculate_spc_limits(df, "defect_density")
