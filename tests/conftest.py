import datetime as dt
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from survey_toolkit.survey_data_generator import generate_all_waves, generate_wave
from survey_toolkit.data_cleaning import apply_demographic_weights, clean_survey_data


@pytest.fixture(scope="session")
def small_raw_df():
    """A fast, realistic-schema dataset: 3 waves of 150 respondents each,
    built directly with generate_wave so tests don't pay for the full
    500-800-respondent default wave size."""
    waves = []
    start = dt.date(2026, 1, 5)
    for wave in range(1, 4):
        wave_date = start + dt.timedelta(weeks=wave - 1)
        waves.append(generate_wave(wave, wave_date, n_respondents=150, seed=1000 + wave))
    import pandas as pd

    return pd.concat(waves, ignore_index=True)


@pytest.fixture(scope="session")
def small_cleaned_df(small_raw_df):
    cleaned = clean_survey_data(small_raw_df)
    return apply_demographic_weights(cleaned)


@pytest.fixture(scope="session")
def small_ready_df(small_cleaned_df):
    return small_cleaned_df[small_cleaned_df["is_analysis_ready"]].copy()


@pytest.fixture(scope="session")
def full_pipeline_df():
    """The full default-size pipeline (14 waves, 500-800 respondents each),
    computed once per test session. Used by tests that need realistic
    statistical power (narrative word count, dashboard summaries)."""
    raw = generate_all_waves()
    cleaned = clean_survey_data(raw)
    weighted = apply_demographic_weights(cleaned)
    return weighted
