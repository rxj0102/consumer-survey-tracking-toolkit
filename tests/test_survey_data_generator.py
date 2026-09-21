import datetime as dt

from survey_toolkit import questionnaire_design as qd
from survey_toolkit.survey_data_generator import (
    NUM_WAVES,
    RESPONDENTS_PER_WAVE_RANGE,
    SPEEDER_MAX_SECONDS,
    generate_all_waves,
    generate_wave,
)


def test_generate_wave_and_reproducibility():
    df = generate_wave(1, dt.date(2026, 1, 5), n_respondents=200, seed=1)
    assert len(df) == 200

    df_same_seed = generate_wave(1, dt.date(2026, 1, 5), n_respondents=100, seed=42)
    df_same_seed_2 = generate_wave(1, dt.date(2026, 1, 5), n_respondents=100, seed=42)
    assert df_same_seed.equals(df_same_seed_2)

    df_other_seed = generate_wave(1, dt.date(2026, 1, 5), n_respondents=100, seed=43)
    assert not df_same_seed["telecom_satisfaction"].equals(df_other_seed["telecom_satisfaction"])


def test_generate_all_waves_respects_documented_ranges():
    assert 12 <= NUM_WAVES <= 16
    df = generate_all_waves(num_waves=3, seed=7)
    counts = df.groupby("wave").size()
    assert len(counts) == 3
    for count in counts:
        assert RESPONDENTS_PER_WAVE_RANGE[0] <= count <= RESPONDENTS_PER_WAVE_RANGE[1]


def test_noise_injection_matches_declared_patterns(small_raw_df):
    likert_cols = ["telecom_satisfaction", "switch_likelihood", "ai_trust"]

    straightliners = small_raw_df[small_raw_df["_true_straightliner"]]
    assert len(straightliners) > 0
    assert (straightliners[likert_cols].nunique(axis=1) == 1).all()

    speeders = small_raw_df[small_raw_df["_true_speeder"]]
    assert len(speeders) > 0
    assert (speeders["completion_seconds"] < SPEEDER_MAX_SECONDS).all()

    inconsistent = small_raw_df[small_raw_df["_true_inconsistent"]]
    assert len(inconsistent) > 0


def test_skip_logic_produces_nulls_for_clean_respondents(small_raw_df):
    no_flight = small_raw_df[small_raw_df["flew_last_12mo"] == "No"]
    clean_no_flight = no_flight[~no_flight["_true_inconsistent"]]
    assert clean_no_flight["airline_nps"].isna().all()
    for brand in qd.AIRLINE_BRANDS:
        assert clean_no_flight[f"aware_airline_{brand}"].isna().all()

    no_ai = small_raw_df[small_raw_df["ai_product_used"] == qd.AI_NONE]
    clean_no_ai = no_ai[~no_ai["_true_inconsistent"]]
    assert clean_no_ai["ai_satisfaction"].isna().all()
