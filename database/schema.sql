-- SemiFab Analytics database schema placeholder

CREATE TABLE IF NOT EXISTS wafer_process_data (
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
    contamination_score REAL,
    particle_count INTEGER,
    defect_density REAL,
    yield_percent REAL,
    pass_fail TEXT
);
