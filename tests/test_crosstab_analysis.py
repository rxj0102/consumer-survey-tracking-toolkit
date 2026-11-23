import math

import pandas as pd
import pytest

from survey_toolkit.crosstab_analysis import (
    generate_banner_table,
    generate_mean_banner_table,
    ztest_proportions,
)


def test_ztest_proportions_known_significant_case():
    result = ztest_proportions(50, 100, 30, 100)
    assert result.z == pytest.approx(2.887, abs=0.01)
    assert result.p_value < 0.05
    assert result.significant is True


def test_ztest_proportions_identical_proportions_not_significant():
    result = ztest_proportions(50, 100, 50, 100)
    assert result.z == pytest.approx(0.0, abs=1e-9)
    assert result.p_value == pytest.approx(1.0, abs=1e-9)
    assert result.significant is False


def test_ztest_proportions_handles_empty_sample():
    result = ztest_proportions(0, 0, 10, 20)
    assert math.isnan(result.z)
    assert result.significant is False


def test_ztest_proportions_respects_alpha():
    # p_value here is ~0.0047: significant at alpha=0.10 but not at alpha=0.001
    result_loose = ztest_proportions(60, 100, 40, 100, alpha=0.10)
    result_strict = ztest_proportions(60, 100, 40, 100, alpha=0.001)
    assert result_loose.significant is True
    assert result_strict.significant is False


def _make_banner_df():
    # group A: 40/50 True (80%); group B: 10/50 True (20%) -- a large, obvious gap
    return pd.DataFrame(
        {
            "metric": [True] * 40 + [False] * 10 + [True] * 10 + [False] * 40,
            "group": ["A"] * 50 + ["B"] * 50,
        }
    )


def test_generate_banner_table_flags_known_difference_and_total_row():
    df = _make_banner_df()
    table = generate_banner_table(df, "metric", banner_vars=["group"], weight_col=None)

    seg_a = table[(table["banner_var"] == "group") & (table["segment"] == "A")].iloc[0]
    seg_b = table[(table["banner_var"] == "group") & (table["segment"] == "B")].iloc[0]
    assert seg_a["proportion"] == pytest.approx(0.8)
    assert seg_b["proportion"] == pytest.approx(0.2)
    assert bool(seg_a["significant"]) is True
    assert bool(seg_b["significant"]) is True

    total_row = table[table["segment"] == "Total"].iloc[0]
    assert total_row["proportion"] == pytest.approx(0.5)
    assert total_row["n"] == 100


def test_generate_banner_table_no_difference_not_significant():
    df = pd.DataFrame(
        {
            "metric": [True] * 25 + [False] * 25 + [True] * 25 + [False] * 25,
            "group": ["A"] * 50 + ["B"] * 50,
        }
    )
    table = generate_banner_table(df, "metric", banner_vars=["group"], weight_col=None)
    assert not table[table["banner_var"] == "group"]["significant"].any()


def test_generate_banner_table_weighting_and_mean_banner():
    weighted_df = _make_banner_df()
    weighted_df["survey_weight"] = 2.0
    table = generate_banner_table(weighted_df, "metric", banner_vars=["group"], weight_col="survey_weight")
    seg_a = table[(table["banner_var"] == "group") & (table["segment"] == "A")].iloc[0]
    assert seg_a["proportion"] == pytest.approx(0.8)  # weighting shouldn't change the proportion
    assert seg_a["weighted_n"] == pytest.approx(100.0)  # but should scale the weighted base

    mean_df = pd.DataFrame(
        {
            "score": [1, 2, 3, 4],
            "group": ["A", "A", "B", "B"],
            "survey_weight": [1.0, 3.0, 1.0, 1.0],
        }
    )
    mean_table = generate_mean_banner_table(mean_df, "score", banner_vars=["group"], weight_col="survey_weight")
    seg_a_mean = mean_table[(mean_table["banner_var"] == "group") & (mean_table["segment"] == "A")].iloc[0]
    # weighted mean of [1,2] with weights [1,3] = (1*1 + 2*3)/4 = 1.75
    assert seg_a_mean["mean"] == pytest.approx(1.75)


def test_kish_effective_n_and_weighted_test():
    import numpy as np
    import pandas as pd

    from survey_toolkit.crosstab_analysis import kish_effective_n, ztest_proportions, ztest_weighted

    assert kish_effective_n(None, 50) == 50
    assert kish_effective_n(pd.Series([1.0] * 40), 40) == 40
    w = pd.Series([1.0] * 50 + [3.0] * 50)
    assert kish_effective_n(w, 100) == (200.0**2) / (50 + 450)  # 80
    # equal weights reduce to the ordinary test
    a = ztest_weighted(0.6, 100, 0.5, 120)
    b = ztest_proportions(60, 100, 60, 120)
    assert np.isclose(a.z, b.z)
    # a smaller effective n gives a weaker test for the same proportions
    assert abs(ztest_weighted(0.6, 50, 0.5, 60).z) < abs(a.z)
