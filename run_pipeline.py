#!/usr/bin/env python3
"""Run the full consumer-survey-tracking-toolkit pipeline end to end.

Generates synthetic wave data, cleans and weights it, then produces the
crosstab, trend, dashboard, and narrative-report deliverables -- exactly
the sequence a market research analyst would run each week.

Usage:
    python3 run_pipeline.py
"""

from pathlib import Path

from survey_toolkit.crosstab_analysis import generate_banner_table
from survey_toolkit.dashboard_export import export_all
from survey_toolkit.data_cleaning import apply_demographic_weights, clean_survey_data
from survey_toolkit.narrative_report import generate_narrative_report, save_narrative_report, word_count
from survey_toolkit.survey_data_generator import generate_all_waves, save_waves

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"


def main() -> None:
    print("1/6 Generating synthetic survey waves...")
    raw = generate_all_waves()
    save_waves(raw, DATA_DIR)
    print(f"    {len(raw):,} responses across {raw['wave'].nunique()} waves -> {DATA_DIR}")

    print("2/6 Cleaning, flagging, and weighting...")
    cleaned = clean_survey_data(raw)
    weighted = apply_demographic_weights(cleaned)
    weighted.to_csv(DATA_DIR / "survey_cleaned_weighted.csv", index=False)
    n_ready = int(weighted["is_analysis_ready"].sum())
    print(f"    {n_ready:,} / {len(weighted):,} interviews analysis-ready after quality screening")

    ready = weighted[weighted["is_analysis_ready"]].copy()

    print("3/6 Running an example crosstab banner table...")
    latest_wave = ready["wave"].max()
    latest = ready[ready["wave"] == latest_wave].copy()
    latest["_aware_nimbus"] = latest.get("aware_ai_Nimbus AI")
    banner = generate_banner_table(latest, "_aware_nimbus")
    banner.to_csv(OUTPUT_DIR / "example_banner_table.csv", index=False)
    print(f"    -> {OUTPUT_DIR / 'example_banner_table.csv'}")

    print("4/6 Exporting dashboard-ready dataset and summary tables...")
    dashboard_results = export_all(ready, OUTPUT_DIR)
    print(f"    dashboard_metrics.csv/.json: {len(dashboard_results['dashboard']):,} rows -> {OUTPUT_DIR}")

    print("5/6 Generating narrative report...")
    report_text = generate_narrative_report(weighted)
    save_narrative_report(report_text, OUTPUT_DIR / "narrative_report.md")
    print(f"    {word_count(report_text)} words -> {OUTPUT_DIR / 'narrative_report.md'}")

    print("6/6 Done. See ./data for raw/cleaned survey files and ./output for all deliverables.")


if __name__ == "__main__":
    main()
