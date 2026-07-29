# semifab-analytics
SemiFab Analytics: Semiconductor process analytics dashboard for lithography monitoring, contamination analysis, defect trends, and quality prediction.

## Version Control Plan

This repository uses main for stable reviewed work and development for active project work. Feature branches will be used for specific updates such as data simulation, dashboard KPI cards, database schema, machine learning model development, testing, and documentation improvements.

Changes will be committed with clear messages and merged through pull requests to maintain traceability and consistency between code, documentation, and design files.
## Unit 6 System Integration Milestone

The Unit 6 milestone integrates the SemiFab Analytics modules into a unified workflow. The integrated system connects simulated wafer data generation, database loading, KPI calculation, SPC-style monitoring, machine-learning prediction, Streamlit dashboard visualization, PyTest integration testing, and deployment planning.

### Integrated Features

- Simulated wafer and lithography data generation
- SQLite/database loading and row-count validation
- KPI calculation: total wafers, average yield, pass rate, total defects, and high-contamination wafers
- SPC-style calculation for defect density with UCL, LCL, and out-of-control count
- Machine-learning model accuracy and latency reporting
- Failure-pattern Pareto analysis
- Streamlit dashboard with KPI cards, charts, and lot-level risk table
- PyTest integration test
- Deployment planning using Streamlit Community Cloud or local execution

### Unit 6 Release

- v0.3-system-integration: System integration, evaluation, dashboard evidence, and deployment planning milestone.
