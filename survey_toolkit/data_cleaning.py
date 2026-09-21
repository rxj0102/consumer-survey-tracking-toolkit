"""Clean and validate raw synthetic survey data.

Implements the three data-quality checks a market research analyst runs on
every wave of tracker data before anything gets reported:

  1. Straight-lining: a respondent gives the same rating to every Likert
     item regardless of content.
  2. Speeding: a respondent completes the survey implausibly fast given
     its length.
  3. Internal inconsistency: a respondent's answers contradict each other
     (e.g. an answer exists to a question that skip logic should have
     hidden, or a brand is considered without ever being marked as aware).

It also applies simple demographic (rim/raking) weighting so the sample
matches known population quota targets on age band and region.
"""

from __future__ import annotations

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd

from survey_toolkit import questionnaire_design as qd

LIKERT_COLS = ["telecom_satisfaction", "switch_likelihood", "ai_trust"]
LIKERT_COLS_WITH_OPTIONAL = LIKERT_COLS + ["ai_satisfaction"]

# A respondent is flagged a speeder if completion time falls at or below
# this many seconds -- deliberately matched to the generator's
# SPEEDER_MAX_SECONDS so injected speeders are always caught, but also
# usable standalone against any completion-time column.
SPEEDER_THRESHOLD_SECONDS = 90


def flag_straightliners(df: pd.DataFrame, likert_cols: Iterable[str] = LIKERT_COLS) -> pd.Series:
    """Flag respondents who gave an identical rating across all of
    ``likert_cols`` that they answered (ai_satisfaction is excluded from
    the required set by default since it's legitimately skipped for
    respondents with no AI product, but if present it must also match).
    """
    likert_cols = list(likert_cols)
    sub = df[likert_cols]
    all_same = sub.nunique(axis=1, dropna=False) == 1
    all_present = sub.notna().all(axis=1)
    flag = all_same & all_present

    if "ai_satisfaction" in df.columns:
        base_value = df[likert_cols[0]]
        ai_sat = df["ai_satisfaction"]
        ai_mismatch = ai_sat.notna() & (ai_sat != base_value)
        flag = flag & ~ai_mismatch

    return flag.rename("flag_straightliner")


def flag_speeders(
    df: pd.DataFrame,
    time_col: str = "completion_seconds",
    threshold_seconds: float = SPEEDER_THRESHOLD_SECONDS,
) -> pd.Series:
    """Flag respondents whose completion time is at/below an absolute
    floor. Also flags anyone below the 2.5th percentile of the observed
    distribution, whichever is more permissive (i.e. catches both
    absolute implausibility and wave-relative outliers)."""
    percentile_floor = df[time_col].quantile(0.025)
    effective_threshold = max(threshold_seconds, percentile_floor)
    flag = df[time_col] <= effective_threshold
    return flag.rename("flag_speeder")


def flag_inconsistent(df: pd.DataFrame) -> pd.Series:
    """Flag internally inconsistent respondents across three checks:

    (a) skip-logic violation: no flight taken but an airline NPS score present
    (b) funnel violation: a telecom brand is "considered" without being
        marked as "aware" of it
    (c) usage/attitude mismatch: claims no AI product used, but an AI
        satisfaction score is present
    """
    a = (df["flew_last_12mo"] == "No") & df["airline_nps"].notna()

    b = pd.Series(False, index=df.index)
    for brand in qd.TELECOM_BRANDS:
        aware_col = f"aware_telecom_{brand}"
        consider_col = f"consider_telecom_{brand}"
        if aware_col in df.columns and consider_col in df.columns:
            b = b | (df[consider_col].fillna(False) & ~df[aware_col].fillna(False))

    c = (df["ai_product_used"] == qd.AI_NONE) & df["ai_satisfaction"].notna()

    flag = a | b | c
    return flag.rename("flag_inconsistent")


def clean_survey_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of ``df`` with flag_* columns added, plus a composite
    ``is_analysis_ready`` column. Rows are never dropped here -- callers
    decide whether to filter or down-weight flagged rows."""
    out = df.copy()
    out["flag_straightliner"] = flag_straightliners(out)
    out["flag_speeder"] = flag_speeders(out)
    out["flag_inconsistent"] = flag_inconsistent(out)
    out["any_quality_flag"] = out["flag_straightliner"] | out["flag_speeder"] | out["flag_inconsistent"]
    out["is_analysis_ready"] = ~out["any_quality_flag"]
    return out


def analysis_ready_subset(df: pd.DataFrame) -> pd.DataFrame:
    """Convenience: clean then return only rows passing all quality checks."""
    cleaned = clean_survey_data(df) if "is_analysis_ready" not in df.columns else df
    return cleaned[cleaned["is_analysis_ready"]].copy()


# ---------------------------------------------------------------------------
# Demographic weighting (iterative proportional fitting / "raking")
# ---------------------------------------------------------------------------

def compute_raking_weights(
    df: pd.DataFrame,
    targets: Dict[str, Dict[str, float]],
    max_iterations: int = 25,
    tolerance: float = 1e-6,
) -> pd.Series:
    """Compute rim (raking) weights so the weighted sample matches marginal
    population targets on each dimension in ``targets``.

    ``targets`` maps column name -> {category: target_proportion}, e.g.
    ``{"age_band": AGE_BAND_QUOTAS, "region": REGION_QUOTAS}``.

    Uses the standard iterative proportional fitting algorithm: for each
    dimension in turn, rescale each respondent's weight by
    (target proportion for their category) / (current weighted proportion
    for their category), then repeat until weights stop moving or
    ``max_iterations`` is reached. Returns a Series of weights indexed like
    ``df``, with mean weight normalized to 1.0.
    """
    n = len(df)
    weights = pd.Series(1.0, index=df.index)

    for dim, target_props in targets.items():
        missing = set(df[dim].dropna().unique()) - set(target_props)
        if missing:
            raise ValueError(f"No raking target provided for {dim} categories: {missing}")

    for _ in range(max_iterations):
        prev_weights = weights.copy()
        for dim, target_props in targets.items():
            col = df[dim]
            total_weight = weights.sum()
            for category, target_prop in target_props.items():
                mask = col == category
                current_weight = weights[mask].sum()
                if current_weight <= 0:
                    continue
                current_prop = current_weight / total_weight
                if current_prop <= 0:
                    continue
                adjustment = target_prop / current_prop
                weights.loc[mask] = weights.loc[mask] * adjustment

        max_change = (weights - prev_weights).abs().max()
        if max_change < tolerance:
            break

    weights = weights / weights.mean()
    return weights.rename("survey_weight")


DEFAULT_WEIGHTING_TARGETS: Dict[str, Dict[str, float]] = {
    "age_band": qd.AGE_BAND_QUOTAS,
    "region": qd.REGION_QUOTAS,
}


def apply_demographic_weights(
    df: pd.DataFrame,
    targets: Dict[str, Dict[str, float]] = None,
) -> pd.DataFrame:
    """Return a copy of df with a `survey_weight` column added, computed
    per wave (each wave is weighted independently, as is standard for a
    tracker) so wave sample composition drift doesn't bias trend lines."""
    targets = targets or DEFAULT_WEIGHTING_TARGETS
    out = df.copy()
    weights = pd.Series(index=out.index, dtype=float)
    for wave, wave_df in out.groupby("wave"):
        wave_weights = compute_raking_weights(wave_df, targets)
        weights.loc[wave_df.index] = wave_weights.values
    out["survey_weight"] = weights
    return out


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    raw_path = project_root / "data" / "survey_raw_all_waves.csv"
    raw = pd.read_csv(raw_path)
    cleaned = clean_survey_data(raw)
    weighted = apply_demographic_weights(cleaned)
    out_path = project_root / "data" / "survey_cleaned_weighted.csv"
    weighted.to_csv(out_path, index=False)
    print(f"Straightliners flagged: {weighted['flag_straightliner'].sum()}")
    print(f"Speeders flagged: {weighted['flag_speeder'].sum()}")
    print(f"Inconsistent flagged: {weighted['flag_inconsistent'].sum()}")
    print(f"Analysis-ready: {weighted['is_analysis_ready'].sum()} / {len(weighted)}")
    print(f"Saved to {out_path}")
