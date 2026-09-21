import json

import pandas as pd

from survey_toolkit.dashboard_export import (
    DASHBOARD_COLUMNS,
    build_dashboard_dataset,
    build_summary_tables,
    export_dashboard_csv,
    export_dashboard_json,
)


def test_build_dashboard_dataset_and_export_round_trip(small_ready_df, tmp_path):
    dashboard_df = build_dashboard_dataset(small_ready_df)
    assert list(dashboard_df.columns) == DASHBOARD_COLUMNS
    assert len(dashboard_df) > 0

    small_df = pd.DataFrame({"wave": [1, 2], "value": [10.0, 20.0]})
    csv_path, json_path = tmp_path / "dashboard.csv", tmp_path / "dashboard.json"
    export_dashboard_csv(small_df, csv_path)
    export_dashboard_json(small_df, json_path)

    reloaded_csv = pd.read_csv(csv_path)
    assert list(reloaded_csv["wave"]) == [1, 2]
    with open(json_path) as f:
        records = json.load(f)
    assert records == [{"wave": 1, "value": 10.0}, {"wave": 2, "value": 20.0}]


def test_build_summary_tables_structure(small_ready_df):
    summaries = build_summary_tables(small_ready_df)
    assert set(summaries.keys()) == {
        "latest_wave_scorecard",
        "topline_trend",
        "key_shifts",
        "data_quality_by_wave",
    }
    assert len(summaries["latest_wave_scorecard"]) > 0
    assert len(summaries["topline_trend"]) > 0

    scorecard = summaries["latest_wave_scorecard"]
    assert not scorecard.duplicated(subset=["category", "brand", "metric"]).any()
