import numpy as np
import pandas as pd
import pytest

from survey_toolkit import questionnaire_design as qd
from survey_toolkit.data_cleaning import (
    apply_demographic_weights,
    clean_survey_data,
    compute_raking_weights,
    flag_inconsistent,
    flag_speeders,
    flag_straightliners,
)


def test_flag_straightliners_detects_constant_ratings():
    df = pd.DataFrame(
        {
            "telecom_satisfaction": [3, 4, 5],
            "switch_likelihood": [3, 1, 5],
            "ai_trust": [3, 2, 5],
        }
    )
    assert flag_straightliners(df).tolist() == [True, False, True]


def test_flag_straightliners_requires_all_present_and_checks_ai_satisfaction():
    df = pd.DataFrame(
        {
            "telecom_satisfaction": [3, np.nan, 3],
            "switch_likelihood": [3, 3, 3],
            "ai_trust": [3, 3, 3],
            "ai_satisfaction": [3, 3, 1],  # row 0: matches; row 1: missing core value; row 2: mismatches
        }
    )
    assert flag_straightliners(df).tolist() == [True, False, False]


def test_flag_speeders_default_and_custom_thresholds():
    default_seconds = [420, 430, 45, 440, 400, 60, 410, 420] * 5  # 40 rows, 10 implausibly fast
    df_default = pd.DataFrame({"completion_seconds": default_seconds})
    flags_default = flag_speeders(df_default)
    assert flags_default.sum() == 10
    assert set(df_default.loc[flags_default, "completion_seconds"].unique()) == {45, 60}

    df_custom = pd.DataFrame({"completion_seconds": [100, 200, 300, 400, 500] * 10})
    flags_custom = flag_speeders(df_custom, threshold_seconds=150)
    assert flags_custom.sum() > 0
    assert (df_custom.loc[flags_custom, "completion_seconds"] <= 150).all()


def test_flag_inconsistent_skip_logic_violation():
    df = pd.DataFrame(
        {
            "flew_last_12mo": ["No", "Yes", "No"],
            "airline_nps": [7, 8, np.nan],
            "ai_product_used": [qd.AI_NONE] * 3,
            "ai_satisfaction": [np.nan] * 3,
        }
    )
    assert flag_inconsistent(df).tolist() == [True, False, False]


def test_flag_inconsistent_ai_usage_mismatch():
    df = pd.DataFrame(
        {
            "flew_last_12mo": ["No", "No"],
            "airline_nps": [np.nan, np.nan],
            "ai_product_used": [qd.AI_NONE, "Nimbus AI"],
            "ai_satisfaction": [4, 4],
        }
    )
    assert flag_inconsistent(df).tolist() == [True, False]


def test_flag_inconsistent_funnel_violation():
    brand = qd.TELECOM_BRANDS[0]
    df = pd.DataFrame(
        {
            "flew_last_12mo": ["No", "No"],
            "airline_nps": [np.nan, np.nan],
            "ai_product_used": [qd.AI_NONE, qd.AI_NONE],
            "ai_satisfaction": [np.nan, np.nan],
            f"aware_telecom_{brand}": [False, True],
            f"consider_telecom_{brand}": [True, True],
        }
    )
    assert flag_inconsistent(df).tolist() == [True, False]


def test_clean_survey_data_recall_on_generator_noise(small_raw_df):
    cleaned = clean_survey_data(small_raw_df)
    for kind in ("straightliner", "speeder", "inconsistent"):
        true_positives = (cleaned[f"flag_{kind}"] & cleaned[f"_true_{kind}"]).sum()
        recall = true_positives / cleaned[f"_true_{kind}"].sum()
        assert recall == 1.0, f"{kind} recall was {recall}"
    assert (cleaned["is_analysis_ready"] == ~cleaned["any_quality_flag"]).all()


def test_compute_raking_weights_correctness():
    df = pd.DataFrame({"age_band": ["18-24"] * 80 + ["25-34"] * 20})
    weights = compute_raking_weights(df, {"age_band": {"18-24": 0.5, "25-34": 0.5}})
    weighted_share = weights[df["age_band"] == "18-24"].sum() / weights.sum()
    assert weighted_share == pytest.approx(0.5, abs=1e-3)

    bad_df = pd.DataFrame({"age_band": ["18-24", "99+"]})
    with pytest.raises(ValueError):
        compute_raking_weights(bad_df, {"age_band": {"18-24": 1.0}})


def test_apply_demographic_weights_correctness(small_cleaned_df):
    assert small_cleaned_df["survey_weight"].mean() == pytest.approx(1.0, abs=1e-6)

    for wave, wave_df in small_cleaned_df.groupby("wave"):
        for band, target in qd.AGE_BAND_QUOTAS.items():
            mask = wave_df["age_band"] == band
            if mask.sum() == 0:
                continue
            weighted_share = wave_df.loc[mask, "survey_weight"].sum() / wave_df["survey_weight"].sum()
            assert weighted_share == pytest.approx(target, abs=1e-3)
