"""SQLite persistence layer for SemiFab Analytics."""
from __future__ import annotations
import sqlite3
import pandas as pd
from pathlib import Path

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS wafer_process (
    wafer_id TEXT PRIMARY KEY,
    lot_id TEXT,
    tool_id TEXT,
    process_date TEXT,
    exposure_dose REAL,
    focus_offset REAL,
    overlay_error REAL,
    cd_mean REAL,
    cd_uniformity REAL,
    resist_thickness REAL,
    particle_count INTEGER,
    contamination_score REAL,
    die_size INTEGER,
    defect_type TEXT,
    failure_pattern TEXT,
    defect_density REAL,
    yield_percent REAL,
    pass_fail TEXT,
    yield_loss REAL
);
"""

def init_database(db_path: str | Path) -> None:
    """Create database schema if it does not exist."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(SCHEMA_SQL)

def load_dataframe(df: pd.DataFrame, db_path: str | Path) -> None:
    """Load a wafer dataframe into SQLite."""
    init_database(db_path)
    clean = df.copy()
    clean["process_date"] = pd.to_datetime(clean["process_date"]).astype(str)
    with sqlite3.connect(db_path) as conn:
        clean.to_sql("wafer_process", conn, if_exists="replace", index=False)

def query_wafer_data(db_path: str | Path) -> pd.DataFrame:
    """Read wafer process data from SQLite for processing and UI layers."""
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql_query("SELECT * FROM wafer_process", conn, parse_dates=["process_date"])
