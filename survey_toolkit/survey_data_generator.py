"""Synthetic weekly survey response generator.

Generates 12-16 weekly waves of ~500-800 respondents each, consistent with
the instrument defined in ``questionnaire_design.py``. The generator is
deliberately noisy: a small share of respondents straight-line, a small
share speed through the survey implausibly fast, and a small share give
internally inconsistent answers -- exactly the kind of data quality issues
a real analyst has to catch before reporting results. ``data_cleaning.py``
is responsible for catching them; this module's job is only to create them.

ALL DATA IS SYNTHETIC. Brand names, respondents, and events are invented
for this toolkit and do not describe any real company, product, or survey.

Wave-over-wave story baked into the data (used to exercise trend_tracking
and narrative_report):
  * NorthLink Mobile (the largest telecom brand by share) suffers a
    service outage centered on waves 9-11: its satisfaction/NPS and
    consideration drop sharply, which drags down the *topline* telecom NPS
    number because NorthLink is the largest brand -- this is the obvious,
    headline-grabbing trend.
  * At the same time, 18-24 year-olds show a much sharper rise in
    consideration of Vantar Wireless than the overall population does --
    a leading indicator of share shift that is easy to miss if an analyst
    only looks at topline numbers. This is the "non-obvious" finding
    narrative_report.py is designed to surface.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from survey_toolkit import questionnaire_design as qd

# ---------------------------------------------------------------------------
# Study configuration
# ---------------------------------------------------------------------------

NUM_WAVES = 14
RESPONDENTS_PER_WAVE_RANGE = (500, 800)
START_DATE = dt.date(2026, 1, 5)  # a Monday
DEFAULT_SEED = 42

# Data-quality noise rates (documented so cleaning/tests can reason about them)
STRAIGHTLINER_RATE = 0.05
SPEEDER_RATE = 0.04
INCONSISTENT_RATE = 0.05
MEDIAN_COMPLETION_SECONDS = 420  # ~7 minutes for a 20-item survey
SPEEDER_MAX_SECONDS = 65  # implausibly fast for this instrument

# Event windows for the injected story
OUTAGE_WAVES = {9, 10, 11}
OUTAGE_BRAND = "NorthLink Mobile"
YOUTH_SHIFT_WAVES = {9, 10, 11, 12}
YOUTH_SHIFT_BRAND = "Vantar Wireless"
YOUTH_SHIFT_AGE_BAND = "18-24"

# ---------------------------------------------------------------------------
# Brand base rates (awareness / consideration / share), wave-1 baseline
# ---------------------------------------------------------------------------

TELECOM_BASE: Dict[str, Dict[str, float]] = {
    "NorthLink Mobile": {"awareness": 0.86, "consideration": 0.42, "share": 0.30, "nps_mean": 32.0},
    "Vantar Wireless": {"awareness": 0.78, "consideration": 0.38, "share": 0.24, "nps_mean": 28.0},
    "CoastalTel": {"awareness": 0.62, "consideration": 0.27, "share": 0.18, "nps_mean": 18.0},
    "Ridgeway Communications": {"awareness": 0.55, "consideration": 0.22, "share": 0.14, "nps_mean": 12.0},
    "PrimeAir Cellular": {"awareness": 0.47, "consideration": 0.18, "share": 0.09, "nps_mean": 8.0},
}
TELECOM_OTHER_SHARE = 1.0 - sum(v["share"] for v in TELECOM_BASE.values())

AIRLINE_BASE: Dict[str, Dict[str, float]] = {
    "Meridian Airways": {"awareness": 0.83, "consideration": 0.40, "share": 0.28, "nps_mean": 24.0},
    "Skyline Air": {"awareness": 0.75, "consideration": 0.35, "share": 0.23, "nps_mean": 20.0},
    "Comet Airlines": {"awareness": 0.60, "consideration": 0.26, "share": 0.20, "nps_mean": 10.0},
    "Harbor Jet": {"awareness": 0.52, "consideration": 0.21, "share": 0.17, "nps_mean": 6.0},
    "Altitude Express": {"awareness": 0.44, "consideration": 0.17, "share": 0.12, "nps_mean": 14.0},
}

AI_BASE: Dict[str, Dict[str, float]] = {
    "Nimbus AI": {"awareness": 0.80, "consideration": 0.36, "share": 0.24, "sat_mean": 3.9},
    "Cortexa": {"awareness": 0.70, "consideration": 0.32, "share": 0.19, "sat_mean": 3.7},
    "Palette Assistant": {"awareness": 0.58, "consideration": 0.24, "share": 0.13, "sat_mean": 3.6},
    "Ledger AI": {"awareness": 0.49, "consideration": 0.19, "share": 0.09, "sat_mean": 3.5},
    "Solace AI": {"awareness": 0.40, "consideration": 0.15, "share": 0.07, "sat_mean": 3.8},
}
AI_NONE_SHARE = 1.0 - sum(v["share"] for v in AI_BASE.values())


def _renormalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    return {k: v / total for k, v in weights.items()}


def _weighted_choice(rng: np.random.Generator, options: List[str], weights: List[float]) -> str:
    weights = np.array(weights, dtype=float)
    weights = np.clip(weights, 1e-6, None)
    weights = weights / weights.sum()
    return rng.choice(options, p=weights)


def _outage_factor(wave: int) -> float:
    """Severity multiplier for the NorthLink outage story, peaking wave 10."""
    if wave not in OUTAGE_WAVES:
        return 0.0
    severity = {9: 0.6, 10: 1.0, 11: 0.5}
    return severity.get(wave, 0.0)


def _youth_shift_factor(wave: int) -> float:
    if wave not in YOUTH_SHIFT_WAVES:
        return 0.0
    severity = {9: 0.3, 10: 0.7, 11: 1.0, 12: 0.6}
    return severity.get(wave, 0.0)


def _wave_carrier_shares(wave: int) -> Dict[str, float]:
    shares = {k: v["share"] for k, v in TELECOM_BASE.items()}
    shares[qd.TELECOM_OTHER] = TELECOM_OTHER_SHARE
    outage = _outage_factor(wave)
    if outage:
        shift = 0.05 * outage
        shares[OUTAGE_BRAND] -= shift
        shares["Vantar Wireless"] += shift * 0.6
        shares["CoastalTel"] += shift * 0.4
    return _renormalize(shares)


def _wave_telecom_metrics(wave: int, age_band: Optional[str] = None) -> Dict[str, Dict[str, float]]:
    """Awareness/consideration/nps for telecom brands at a given wave,
    optionally conditioned on an age band for the youth-shift story."""
    out = {}
    outage = _outage_factor(wave)
    youth = _youth_shift_factor(wave) if age_band == YOUTH_SHIFT_AGE_BAND else 0.0
    for brand, base in TELECOM_BASE.items():
        awareness = base["awareness"]
        consideration = base["consideration"]
        nps_mean = base["nps_mean"]
        if brand == OUTAGE_BRAND and outage:
            awareness -= 0.02 * outage
            consideration -= 0.14 * outage
            nps_mean -= 28.0 * outage
        if brand == YOUTH_SHIFT_BRAND and youth:
            # sharper rise in consideration among 18-24s than overall population sees
            consideration += 0.28 * youth
        out[brand] = {
            "awareness": float(np.clip(awareness, 0.02, 0.99)),
            "consideration": float(np.clip(consideration, 0.01, 0.95)),
            "nps_mean": float(np.clip(nps_mean, -80.0, 80.0)),
        }
    return out


def _wave_airline_metrics(wave: int) -> Dict[str, Dict[str, float]]:
    # Airlines drift with small independent weekly noise only (no scripted event).
    out = {}
    rng = np.random.default_rng(1000 + wave)
    for brand, base in AIRLINE_BASE.items():
        jitter = rng.normal(0, 0.01)
        out[brand] = {
            "awareness": float(np.clip(base["awareness"] + jitter, 0.02, 0.99)),
            "consideration": float(np.clip(base["consideration"] + jitter, 0.01, 0.95)),
            "nps_mean": float(np.clip(base["nps_mean"] + jitter * 50, -80.0, 80.0)),
        }
    return out


def _wave_ai_metrics(wave: int) -> Dict[str, Dict[str, float]]:
    # Slow, steady upward drift in AI awareness/consideration over the study period.
    progress = wave / NUM_WAVES
    out = {}
    for brand, base in AI_BASE.items():
        out[brand] = {
            "awareness": float(np.clip(base["awareness"] + 0.06 * progress, 0.02, 0.99)),
            "consideration": float(np.clip(base["consideration"] + 0.05 * progress, 0.01, 0.95)),
            "sat_mean": base["sat_mean"],
        }
    return out


def _sample_likert(rng: np.random.Generator, mean: float, sd: float = 0.9, low: int = 1, high: int = 5) -> int:
    val = rng.normal(mean, sd)
    return int(np.clip(round(val), low, high))


def _sample_nps(rng: np.random.Generator, mean: float, sd: float = 22.0) -> int:
    """Sample a 0-10 NPS-item score whose *population* NPS (promoters -
    detractors, as percentages) centers near `mean`. We approximate by
    drawing a latent score around a linear map of mean into 0-10 space."""
    center = np.clip(5.0 + mean / 20.0, 0.5, 9.5)
    val = rng.normal(center, 1.8)
    return int(np.clip(round(val), 0, 10))


def _sample_age(rng: np.random.Generator, age_band: str) -> int:
    bounds = {
        "18-24": (18, 24), "25-34": (25, 34), "35-44": (35, 44),
        "45-54": (45, 54), "55-64": (55, 64), "65+": (65, 85),
    }
    lo, hi = bounds[age_band]
    return int(rng.integers(lo, hi + 1))


def generate_respondent(rng: np.random.Generator, wave: int, wave_date: dt.date, response_id: str) -> dict:
    # --- demographics (quota-controlled dimensions get small per-wave jitter) ---
    age_band = _weighted_choice(rng, list(qd.AGE_BAND_QUOTAS), list(qd.AGE_BAND_QUOTAS.values()))
    region = _weighted_choice(rng, list(qd.REGION_QUOTAS), list(qd.REGION_QUOTAS.values()))
    gender = _weighted_choice(rng, list(qd.GENDER_QUOTAS), list(qd.GENDER_QUOTAS.values()))
    income_band = _weighted_choice(rng, list(qd.INCOME_QUOTAS), list(qd.INCOME_QUOTAS.values()))
    age = _sample_age(rng, age_band)

    carrier_shares = _wave_carrier_shares(wave)
    telecom_carrier = _weighted_choice(rng, list(carrier_shares), list(carrier_shares.values()))

    flew_12mo = "Yes" if rng.random() < qd.FLEW_LAST_12MO_RATE else "No"

    airline_used = None
    if flew_12mo == "Yes":
        airline_shares = {k: v["share"] for k, v in AIRLINE_BASE.items()}
        airline_used = _weighted_choice(rng, list(airline_shares), list(airline_shares.values()))

    ai_shares = {k: v["share"] for k, v in AI_BASE.items()}
    ai_shares[qd.AI_NONE] = AI_NONE_SHARE
    ai_product = _weighted_choice(rng, list(ai_shares), list(ai_shares.values()))

    # --- funnel metrics (awareness / consideration) ---
    telecom_metrics = _wave_telecom_metrics(wave, age_band=age_band)
    airline_metrics = _wave_airline_metrics(wave)
    ai_metrics = _wave_ai_metrics(wave)

    telecom_awareness = {b: bool(rng.random() < m["awareness"]) for b, m in telecom_metrics.items()}
    telecom_awareness[telecom_carrier] = True if telecom_carrier in telecom_awareness else telecom_awareness.get(telecom_carrier, False)

    telecom_consideration = {
        b: bool(telecom_awareness[b] and rng.random() < (m["consideration"] / max(m["awareness"], 1e-6)))
        for b, m in telecom_metrics.items()
    }

    airline_awareness = {b: bool(rng.random() < m["awareness"]) for b, m in airline_metrics.items()}
    if airline_used is not None:
        airline_awareness[airline_used] = True
    airline_consideration = {
        b: bool(airline_awareness[b] and rng.random() < (m["consideration"] / max(m["awareness"], 1e-6)))
        for b, m in airline_metrics.items()
    }

    ai_awareness = {b: bool(rng.random() < m["awareness"]) for b, m in ai_metrics.items()}
    if ai_product != qd.AI_NONE:
        ai_awareness[ai_product] = True

    # --- attitude items tied to the respondent's own current brand ---
    if telecom_carrier in TELECOM_BASE:
        t_nps_mean = telecom_metrics[telecom_carrier]["nps_mean"]
    else:
        t_nps_mean = 5.0  # Other/Prepaid baseline
    telecom_nps = _sample_nps(rng, t_nps_mean)
    telecom_satisfaction = _sample_likert(rng, 2.6 + t_nps_mean / 60.0)
    switch_likelihood = _sample_likert(rng, 3.4 - t_nps_mean / 40.0)

    ai_trust = _sample_likert(rng, 3.2)

    airline_nps = None
    if flew_12mo == "Yes":
        a_nps_mean = airline_metrics[airline_used]["nps_mean"]
        airline_nps = _sample_nps(rng, a_nps_mean)

    ai_satisfaction = None
    if ai_product != qd.AI_NONE:
        ai_satisfaction = _sample_likert(rng, ai_metrics[ai_product]["sat_mean"])

    verbatim_improve = rng.choice(qd.VERBATIM_IMPROVE_BANK)
    verbatim_switch_reason = None
    if switch_likelihood >= 4:
        verbatim_switch_reason = rng.choice(qd.VERBATIM_SWITCH_REASON_BANK)

    completion_seconds = max(45, int(rng.normal(MEDIAN_COMPLETION_SECONDS, 110)))

    record = {
        "response_id": response_id,
        "wave": wave,
        "wave_date": wave_date.isoformat(),
        "age": age,
        "age_band": age_band,
        "region": region,
        "gender": gender,
        "income_band": income_band,
        "telecom_carrier_used": telecom_carrier,
        "flew_last_12mo": flew_12mo,
        "airline_used": airline_used,
        "ai_product_used": ai_product,
        "completion_seconds": completion_seconds,
    }
    for b in qd.TELECOM_BRANDS:
        record[f"aware_telecom_{b}"] = telecom_awareness[b]
        record[f"consider_telecom_{b}"] = telecom_consideration[b]
    for b in qd.AIRLINE_BRANDS:
        record[f"aware_airline_{b}"] = airline_awareness[b] if not (flew_12mo == "No") else airline_awareness[b]
        record[f"consider_airline_{b}"] = airline_consideration[b]
    for b in qd.AI_BRANDS:
        record[f"aware_ai_{b}"] = ai_awareness[b]

    record["telecom_satisfaction"] = telecom_satisfaction
    record["switch_likelihood"] = switch_likelihood
    record["ai_trust"] = ai_trust
    record["airline_nps"] = airline_nps
    record["telecom_nps"] = telecom_nps
    record["ai_satisfaction"] = ai_satisfaction
    record["verbatim_improve"] = verbatim_improve
    record["verbatim_switch_reason"] = verbatim_switch_reason

    # Airline funnel questions are skipped entirely (None) if respondent hasn't flown
    if flew_12mo == "No":
        for b in qd.AIRLINE_BRANDS:
            record[f"aware_airline_{b}"] = None
            record[f"consider_airline_{b}"] = None

    return record


def _apply_straightlining(rng: np.random.Generator, record: dict) -> dict:
    """Overwrite the four Likert items with a single repeated value,
    the classic straight-lining pattern."""
    value = int(rng.integers(1, 6))
    record["telecom_satisfaction"] = value
    record["switch_likelihood"] = value
    record["ai_trust"] = value
    if record["ai_satisfaction"] is not None:
        record["ai_satisfaction"] = value
    return record


def _apply_speeding(rng: np.random.Generator, record: dict) -> dict:
    record["completion_seconds"] = int(rng.integers(20, SPEEDER_MAX_SECONDS))
    return record


def _apply_inconsistency(rng: np.random.Generator, record: dict) -> dict:
    """Inject one of three internally-inconsistent patterns a real analyst
    must catch: (1) an answer given to a question that skip logic should
    have hidden, (2) consideration of a brand never marked as aware,
    (3) a brand-usage/attitude mismatch."""
    kind = rng.integers(0, 3)
    if kind == 0:
        # Skip-logic violation: no flight taken, but an airline NPS score present anyway.
        record["flew_last_12mo"] = "No"
        record["airline_nps"] = int(rng.integers(0, 11))
        for b in qd.AIRLINE_BRANDS:
            record[f"aware_airline_{b}"] = None
            record[f"consider_airline_{b}"] = None
    elif kind == 1:
        # Considers a brand never marked as aware of.
        candidate_brands = qd.TELECOM_BRANDS
        b = rng.choice(candidate_brands)
        record[f"aware_telecom_{b}"] = False
        record[f"consider_telecom_{b}"] = True
    else:
        # Claims no AI product usage, but an AI satisfaction rating is present anyway.
        record["ai_product_used"] = qd.AI_NONE
        record["ai_satisfaction"] = int(rng.integers(1, 6))
    return record


def generate_wave(wave: int, wave_date: dt.date, n_respondents: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    records = []
    n_straightliners = int(round(n_respondents * STRAIGHTLINER_RATE))
    n_speeders = int(round(n_respondents * SPEEDER_RATE))
    n_inconsistent = int(round(n_respondents * INCONSISTENT_RATE))

    flag_idx = rng.permutation(n_respondents)
    straightline_idx = set(flag_idx[:n_straightliners].tolist())
    speeder_idx = set(flag_idx[n_straightliners:n_straightliners + n_speeders].tolist())
    inconsistent_idx = set(
        flag_idx[n_straightliners + n_speeders: n_straightliners + n_speeders + n_inconsistent].tolist()
    )

    for i in range(n_respondents):
        response_id = f"W{wave:02d}-{i:05d}"
        record = generate_respondent(rng, wave, wave_date, response_id)

        record["_true_straightliner"] = i in straightline_idx
        record["_true_speeder"] = i in speeder_idx
        record["_true_inconsistent"] = i in inconsistent_idx

        if i in straightline_idx:
            record = _apply_straightlining(rng, record)
        if i in speeder_idx:
            record = _apply_speeding(rng, record)
        if i in inconsistent_idx:
            record = _apply_inconsistency(rng, record)

        records.append(record)

    return pd.DataFrame.from_records(records)


def generate_all_waves(
    num_waves: int = NUM_WAVES,
    start_date: dt.date = START_DATE,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    frames = []
    for wave in range(1, num_waves + 1):
        wave_date = start_date + dt.timedelta(weeks=wave - 1)
        n_respondents = int(rng.integers(RESPONDENTS_PER_WAVE_RANGE[0], RESPONDENTS_PER_WAVE_RANGE[1] + 1))
        wave_seed = seed * 1000 + wave
        frames.append(generate_wave(wave, wave_date, n_respondents, wave_seed))
    return pd.concat(frames, ignore_index=True)


def save_waves(df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for wave, wave_df in df.groupby("wave"):
        wave_df.to_csv(output_dir / f"wave_{wave:02d}_raw.csv", index=False)
    df.to_csv(output_dir / "survey_raw_all_waves.csv", index=False)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / "data"
    all_waves = generate_all_waves()
    save_waves(all_waves, data_dir)
    print(f"Generated {len(all_waves)} responses across {all_waves['wave'].nunique()} waves.")
    print(f"Saved to {data_dir}")
