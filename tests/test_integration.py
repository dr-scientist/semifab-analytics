from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from integration_runner import run_pipeline

def test_unit6_pipeline_runs_end_to_end():
    results = run_pipeline()
    assert results["input_rows"] == 360
    assert results["database_rows_after_load"] == 360
    assert results["model_accuracy"] >= 0.70
    assert results["requires_review_count"] > 0
