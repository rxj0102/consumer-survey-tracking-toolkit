"""Export a dashboard-ready dataset for a BI tool (Tableau / Power BI).

Takes the long-format tracking dataset produced by trend_tracking.py and
writes it to CSV and JSON in the shape a BI tool expects: one row per
(wave, category, brand, metric, segment_var, segment_value), each with a
single numeric value column -- no pivoting required on the BI side.

Also produces a handful of pre-aggregated summary tables for a quick
executive review without opening a BI tool at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from survey_toolkit.trend_tracking import (
    add_rolling_average,
    build_tracking_dataset,
    compute_wow_shifts,
    find_headline_shift,
    find_notable_non_obvious_shift,
)

DASHBOARD_COLUMNS = [
    "wave",
    "wave_date",
    "category",
    "brand",
    "metric",
    "segment_var",
    "segment_value",
    "n",
    "weighted_n",
    "value",
    "rolling_avg",
]


def build_dashboard_dataset(df: pd.DataFrame, weight_col: Optional[str] = "survey_weight") -> pd.DataFrame:
    """Build the long-format, BI-tool-ready dataset."""
    tracking = build_tracking_dataset(df, weight_col=weight_col)
    tracking = add_rolling_average(tracking)
    return tracking[DASHBOARD_COLUMNS].copy()


def export_dashboard_csv(dashboard_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dashboard_df.to_csv(path, index=False)


def export_dashboard_json(dashboard_df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = dashboard_df.to_dict(orient="records")
    with open(path, "w") as f:
        json.dump(records, f, indent=2, default=str)


def build_summary_tables(df: pd.DataFrame, weight_col: Optional[str] = "survey_weight") -> dict:
    """Pre-aggregated tables for an executive who won't open the BI tool:
    latest-wave brand scorecard, topline trend-over-time, and the flagged
    headline/notable shifts.
    """
    tracking = add_rolling_average(build_tracking_dataset(df, weight_col=weight_col))
    shifts = compute_wow_shifts(tracking)

    latest_wave = tracking["wave"].max()
    scorecard = tracking[
        (tracking["wave"] == latest_wave) & (tracking["segment_var"] == "Total")
    ][["category", "brand", "metric", "value", "rolling_avg", "n"]].sort_values(
        ["category", "brand", "metric"]
    ).reset_index(drop=True)

    topline_trend = tracking[tracking["segment_var"] == "Total"][
        ["wave", "wave_date", "category", "brand", "metric", "value", "rolling_avg"]
    ].sort_values(["category", "brand", "metric", "wave"]).reset_index(drop=True)

    headline = find_headline_shift(shifts)
    notable = find_notable_non_obvious_shift(shifts)
    key_shifts = pd.DataFrame([r for r in (headline, notable) if r is not None])

    quality_cols = [c for c in ["flag_straightliner", "flag_speeder", "flag_inconsistent", "is_analysis_ready"] if c in df.columns]
    if quality_cols:
        data_quality_by_wave = (
            df.groupby("wave")[quality_cols]
            .mean()
            .reset_index()
            .rename(columns={c: f"{c}_rate" for c in quality_cols if c != "is_analysis_ready"})
        )
    else:
        data_quality_by_wave = pd.DataFrame()

    return {
        "latest_wave_scorecard": scorecard,
        "topline_trend": topline_trend,
        "key_shifts": key_shifts,
        "data_quality_by_wave": data_quality_by_wave,
    }


def export_all(
    df: pd.DataFrame,
    output_dir: Path,
    weight_col: Optional[str] = "survey_weight",
) -> dict:
    """Write the dashboard CSV/JSON plus every summary table CSV to
    ``output_dir``. Returns the dict of DataFrames written, for reuse
    (e.g. by narrative_report.py) without recomputation."""
    output_dir.mkdir(parents=True, exist_ok=True)

    dashboard_df = build_dashboard_dataset(df, weight_col=weight_col)
    export_dashboard_csv(dashboard_df, output_dir / "dashboard_metrics.csv")
    export_dashboard_json(dashboard_df, output_dir / "dashboard_metrics.json")

    summaries = build_summary_tables(df, weight_col=weight_col)
    for name, table in summaries.items():
        table.to_csv(output_dir / f"summary_{name}.csv", index=False)

    return {"dashboard": dashboard_df, **summaries}


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    cleaned_path = project_root / "data" / "survey_cleaned_weighted.csv"
    output_dir = project_root / "output"

    raw = pd.read_csv(cleaned_path)
    ready = raw[raw["is_analysis_ready"]].copy()

    results = export_all(ready, output_dir)
    print(f"Dashboard dataset: {len(results['dashboard'])} rows -> {output_dir / 'dashboard_metrics.csv'}")
    for name in ("latest_wave_scorecard", "topline_trend", "key_shifts", "data_quality_by_wave"):
        print(f"  summary_{name}.csv: {len(results[name])} rows")
