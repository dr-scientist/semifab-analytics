import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_PATH))

from semifab_data import generate_wafer_data
from semifab_analytics import calculate_kpis, calculate_spc_limits


def test_data_generation_creates_rows():
    df = generate_wafer_data(120)

    assert df is not None
    assert len(df) == 120
    assert len(df.columns) > 0


def test_data_generation_has_important_columns():
    df = generate_wafer_data(120)

    important_columns = [
        "wafer_id",
        "lot_id",
        "tool_id",
        "exposure_dose",
        "focus_offset",
        "overlay_error",
        "cd_mean",
        "cd_uniformity",
        "resist_thickness",
        "particle_count",
        "contamination_score",
        "defect_type",
        "failure_pattern",
        "defect_density",
        "yield_percent",
        "pass_fail",
        "yield_loss",
    ]

    for column in important_columns:
        assert column in df.columns


def test_kpi_calculation_returns_expected_keys():
    df = generate_wafer_data(120)
    kpis = calculate_kpis(df)

    assert isinstance(kpis, dict)

    expected_keys = [
        "total_wafers",
        "avg_yield_percent",
        "pass_rate_percent",
        "total_defects_est",
        "avg_defects_per_wafer",
        "high_contamination_wafers",
    ]

    for key in expected_keys:
        assert key in kpis


def test_kpi_values_are_valid():
    df = generate_wafer_data(120)
    kpis = calculate_kpis(df)

    assert kpis["total_wafers"] == 120
    assert 0 <= kpis["avg_yield_percent"] <= 100
    assert 0 <= kpis["pass_rate_percent"] <= 100
    assert kpis["total_defects_est"] >= 0
    assert kpis["avg_defects_per_wafer"] >= 0
    assert kpis["high_contamination_wafers"] >= 0


def test_spc_limits_are_valid():
    df = generate_wafer_data(120)
    spc = calculate_spc_limits(df, "defect_density")

    assert isinstance(spc, dict)
    assert spc["metric"] == "defect_density"
    assert spc["ucl"] > spc["mean"]
    assert spc["lcl"] < spc["mean"]
    assert spc["out_of_control_count"] >= 0


def test_pass_fail_values_are_valid():
    df = generate_wafer_data(120)

    assert "pass_fail" in df.columns
    assert df["pass_fail"].notna().all()

    allowed_values = {"pass", "fail"}
    actual_values = set(df["pass_fail"].unique())

    assert actual_values.issubset(allowed_values)


def test_failure_pattern_values_exist():
    df = generate_wafer_data(120)

    assert "failure_pattern" in df.columns
    assert df["failure_pattern"].notna().all()
    assert len(df["failure_pattern"].unique()) > 0