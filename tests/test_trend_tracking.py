import pandas as pd
import pytest

from survey_toolkit.trend_tracking import (
    add_rolling_average,
    build_tracking_dataset,
    compute_wow_shifts,
    find_headline_shift,
    find_notable_non_obvious_shift,
)


def _tracking_row(wave, category, brand, metric, segment_var, segment_value, n, success_n, value):
    return {
        "wave": wave,
        "wave_date": f"2026-{wave:02d}-01",
        "category": category,
        "brand": brand,
        "metric": metric,
        "segment_var": segment_var,
        "segment_value": segment_value,
        "n": n,
        "weighted_n": float(n),
        "value": value,
        "success_n": success_n,
    }


@pytest.fixture
def synthetic_tracking_df():
    rows = [
        # Topline "BrandA" nps: significant rise wave1->wave2, flat wave2->wave3
        _tracking_row(1, "telecom", "BrandA", "nps", "Total", "All", 100, 50, 50.0),
        _tracking_row(2, "telecom", "BrandA", "nps", "Total", "All", 100, 70, 70.0),
        _tracking_row(3, "telecom", "BrandA", "nps", "Total", "All", 100, 72, 72.0),
        # Segment-level: much sharper rise in the same metric/brand for age_band 18-24
        _tracking_row(1, "telecom", "BrandA", "nps", "age_band", "18-24", 50, 10, 20.0),
        _tracking_row(2, "telecom", "BrandA", "nps", "age_band", "18-24", 50, 40, 80.0),
        # A tiny, noisy cell that should be excluded entirely by the min_n floor
        _tracking_row(1, "airline", "BrandX", "awareness", "Total", "All", 10, 2, 20.0),
        _tracking_row(2, "airline", "BrandX", "awareness", "Total", "All", 10, 8, 80.0),
    ]
    return pd.DataFrame(rows)


def test_add_rolling_average_computes_expected_window():
    df = pd.DataFrame(
        [
            _tracking_row(w, "telecom", "BrandA", "nps", "Total", "All", 100, 10, val)
            for w, val in zip(range(1, 6), [10, 20, 30, 40, 50])
        ]
    )
    out = add_rolling_average(df, window=3)
    assert out["rolling_avg"].tolist() == pytest.approx([10.0, 15.0, 20.0, 30.0, 40.0])


def test_compute_wow_shifts_excludes_small_cells(synthetic_tracking_df):
    shifts = compute_wow_shifts(synthetic_tracking_df, min_n=30)
    assert shifts[shifts["brand"] == "BrandX"].empty


def test_compute_wow_shifts_significance(synthetic_tracking_df):
    shifts = compute_wow_shifts(synthetic_tracking_df, min_n=30)
    topline = shifts[
        (shifts["brand"] == "BrandA") & (shifts["segment_var"] == "Total") & (shifts["wave"] == 2)
    ].iloc[0]
    assert bool(topline["significant"]) is True
    assert topline["change"] == pytest.approx(20.0)

    flat = shifts[
        (shifts["brand"] == "BrandA") & (shifts["segment_var"] == "Total") & (shifts["wave"] == 3)
    ].iloc[0]
    assert bool(flat["significant"]) is False


def test_find_headline_shift_picks_largest_total_change(synthetic_tracking_df):
    shifts = compute_wow_shifts(synthetic_tracking_df, min_n=30)
    headline = find_headline_shift(shifts)
    assert headline["segment_var"] == "Total"
    assert headline["brand"] == "BrandA"
    assert headline["wave"] == 2


def test_find_headline_shift_returns_none_when_no_significant_total_rows():
    rows = [
        _tracking_row(1, "telecom", "BrandA", "nps", "age_band", "18-24", 50, 25, 50.0),
        _tracking_row(2, "telecom", "BrandA", "nps", "age_band", "18-24", 50, 26, 52.0),
    ]
    shifts = compute_wow_shifts(pd.DataFrame(rows), min_n=30)
    assert find_headline_shift(shifts) is None


def test_find_notable_non_obvious_shift_prefers_bigger_segment_surprise(synthetic_tracking_df):
    shifts = compute_wow_shifts(synthetic_tracking_df, min_n=30)
    notable = find_notable_non_obvious_shift(shifts)
    assert notable["segment_var"] == "age_band"
    assert notable["segment_value"] == "18-24"
    assert notable["change"] == pytest.approx(60.0)


def test_find_notable_non_obvious_shift_returns_none_without_segment_rows():
    rows = [
        _tracking_row(1, "telecom", "BrandA", "nps", "Total", "All", 100, 50, 50.0),
        _tracking_row(2, "telecom", "BrandA", "nps", "Total", "All", 100, 70, 70.0),
    ]
    shifts = compute_wow_shifts(pd.DataFrame(rows), min_n=30)
    assert find_notable_non_obvious_shift(shifts) is None


def test_build_tracking_dataset_structure_and_bounds(small_ready_df):
    tracking = build_tracking_dataset(small_ready_df)
    expected_cols = {
        "wave", "wave_date", "category", "brand", "metric",
        "segment_var", "segment_value", "n", "weighted_n", "value", "success_n",
    }
    assert expected_cols.issubset(tracking.columns)
    assert tracking["wave"].nunique() == small_ready_df["wave"].nunique()
    assert set(tracking["segment_var"].unique()) >= {"Total", "age_band", "region"}

    proportion_rows = tracking[~tracking["metric"].str.contains("nps")]
    assert proportion_rows["value"].between(0, 100).all()
