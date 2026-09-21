"""Wave-over-wave trend tracking.

Builds a long-format "tracking dataset" of key metrics (brand awareness,
consideration, usage share, NPS-style scores) by wave and by segment,
computes rolling averages, tests each wave-over-wave move for statistical
significance, and picks out the single most notable shift in the data --
explicitly distinguishing the obvious topline ("headline") movement from a
more interesting segment-level ("non-obvious") one.

All significance testing reduces to a two-proportion z-test (see
crosstab_analysis.ztest_proportions): proportion metrics test directly,
and NPS-style metrics are tested via their promoter rate (share of
scores >= 9), which is documented as a standard, tractable proxy for
movement in the full NPS figure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd

from survey_toolkit import questionnaire_design as qd
from survey_toolkit.crosstab_analysis import ztest_proportions

SEGMENT_VARS = ["age_band", "region"]
ROLLING_WINDOW = 3


@dataclass
class MetricSpec:
    category: str  # telecom | airline | ai | general
    brand: str  # brand name, or "ALL" for a topline/blended metric
    metric: str  # awareness | consideration | share | nps | top2box_<name>
    test_kind: str  # "proportion" or "nps"
    base_mask: Callable[[pd.DataFrame], pd.Series]
    success_col: str  # boolean column (proportion) or raw 0-10 score column (nps)


def _build_metric_specs(df: pd.DataFrame) -> List[MetricSpec]:
    specs: List[MetricSpec] = []

    for brand in qd.TELECOM_BRANDS:
        specs.append(MetricSpec("telecom", brand, "awareness", "proportion",
                                 lambda d: pd.Series(True, index=d.index), f"aware_telecom_{brand}"))
        specs.append(MetricSpec("telecom", brand, "consideration", "proportion",
                                 lambda d: pd.Series(True, index=d.index), f"consider_telecom_{brand}"))

    flew_mask = lambda d: d["flew_last_12mo"] == "Yes"
    for brand in qd.AIRLINE_BRANDS:
        specs.append(MetricSpec("airline", brand, "awareness", "proportion", flew_mask, f"aware_airline_{brand}"))
        specs.append(MetricSpec("airline", brand, "consideration", "proportion", flew_mask, f"consider_airline_{brand}"))

    for brand in qd.AI_BRANDS:
        specs.append(MetricSpec("ai", brand, "awareness", "proportion",
                                 lambda d: pd.Series(True, index=d.index), f"aware_ai_{brand}"))

    # usage / market share metrics
    for brand in qd.TELECOM_BRANDS + [qd.TELECOM_OTHER]:
        col = f"_is_telecom_user_{brand}"
        specs.append(MetricSpec("telecom", brand, "usage_share", "proportion",
                                 lambda d: pd.Series(True, index=d.index), col))
    for brand in qd.AIRLINE_BRANDS:
        col = f"_is_airline_user_{brand}"
        specs.append(MetricSpec("airline", brand, "usage_share", "proportion", flew_mask, col))
    for brand in qd.AI_BRANDS + [qd.AI_NONE]:
        col = f"_is_ai_user_{brand}"
        specs.append(MetricSpec("ai", brand, "usage_share", "proportion",
                                 lambda d: pd.Series(True, index=d.index), col))

    # NPS-style metrics (brand-level and topline)
    for brand in qd.TELECOM_BRANDS + [qd.TELECOM_OTHER]:
        mask = lambda d, b=brand: d["telecom_carrier_used"] == b
        specs.append(MetricSpec("telecom", brand, "nps", "nps", mask, "telecom_nps"))
    specs.append(MetricSpec("telecom", "ALL", "nps", "nps", lambda d: d["telecom_nps"].notna(), "telecom_nps"))

    for brand in qd.AIRLINE_BRANDS:
        mask = lambda d, b=brand: d["airline_used"] == b
        specs.append(MetricSpec("airline", brand, "nps", "nps", mask, "airline_nps"))
    specs.append(MetricSpec("airline", "ALL", "nps", "nps", flew_mask, "airline_nps"))

    # AI satisfaction (1-5 Likert) reported as top-2-box proportion
    for brand in qd.AI_BRANDS:
        col = "_ai_satisfaction_top2box"
        mask = lambda d, b=brand: d["ai_product_used"] == b
        specs.append(MetricSpec("ai", brand, "top2box_satisfaction", "proportion", mask, col))

    # general attitude topline metrics
    specs.append(MetricSpec("general", "ALL", "top2box_telecom_satisfaction", "proportion",
                             lambda d: pd.Series(True, index=d.index), "_top2box_telecom_satisfaction"))
    specs.append(MetricSpec("general", "ALL", "top2box_switch_likelihood", "proportion",
                             lambda d: pd.Series(True, index=d.index), "_top2box_switch_likelihood"))
    specs.append(MetricSpec("general", "ALL", "top2box_ai_trust", "proportion",
                             lambda d: pd.Series(True, index=d.index), "_top2box_ai_trust"))

    return specs


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for brand in qd.TELECOM_BRANDS + [qd.TELECOM_OTHER]:
        out[f"_is_telecom_user_{brand}"] = out["telecom_carrier_used"] == brand
    for brand in qd.AIRLINE_BRANDS:
        out[f"_is_airline_user_{brand}"] = out["airline_used"] == brand
    for brand in qd.AI_BRANDS + [qd.AI_NONE]:
        out[f"_is_ai_user_{brand}"] = out["ai_product_used"] == brand
    out["_ai_satisfaction_top2box"] = out["ai_satisfaction"] >= 4
    out["_top2box_telecom_satisfaction"] = out["telecom_satisfaction"] >= 4
    out["_top2box_switch_likelihood"] = out["switch_likelihood"] >= 4
    out["_top2box_ai_trust"] = out["ai_trust"] >= 4
    return out


def _success_series(sub: pd.DataFrame, spec: MetricSpec) -> pd.Series:
    if spec.test_kind == "proportion":
        return sub[spec.success_col].astype(bool)
    # nps: "success" for the z-test proxy is being a promoter (score >= 9)
    return sub[spec.success_col] >= 9


def _display_value(sub: pd.DataFrame, spec: MetricSpec, weight_col: Optional[str]) -> float:
    weights = sub[weight_col] if (weight_col and weight_col in sub.columns) else pd.Series(1.0, index=sub.index)
    total_w = weights.sum()
    if total_w == 0:
        return np.nan
    if spec.test_kind == "proportion":
        successes = sub[spec.success_col].astype(bool)
        return float((successes * weights).sum() / total_w * 100)
    # nps: classic promoters% - detractors%, weighted
    scores = sub[spec.success_col]
    promoters = scores >= 9
    detractors = scores <= 6
    prom_pct = (promoters * weights).sum() / total_w
    det_pct = (detractors * weights).sum() / total_w
    return float((prom_pct - det_pct) * 100)


def build_tracking_dataset(
    df: pd.DataFrame,
    weight_col: Optional[str] = "survey_weight",
    segment_vars: List[str] = None,
) -> pd.DataFrame:
    """Build the long-format wave-by-segment tracking dataset.

    Returns columns: wave, wave_date, category, brand, metric, segment_var,
    segment_value, n, weighted_n, value, success_n (unweighted successes,
    used later for significance testing).
    """
    segment_vars = segment_vars if segment_vars is not None else SEGMENT_VARS
    working = _add_derived_columns(df)
    specs = _build_metric_specs(working)

    wave_dates = working.groupby("wave")["wave_date"].first()

    rows = []
    for wave, wdf in working.groupby("wave"):
        for spec in specs:
            base = wdf[spec.base_mask(wdf)]
            base = base[base[spec.success_col].notna()]
            if base.empty:
                continue

            segment_defs = [("Total", "All", base)]
            for seg_var in segment_vars:
                for seg_val, seg_df in base.groupby(seg_var):
                    segment_defs.append((seg_var, seg_val, seg_df))

            for seg_var, seg_val, seg_df in segment_defs:
                if seg_df.empty:
                    continue
                value = _display_value(seg_df, spec, weight_col)
                success_uw = int(_success_series(seg_df, spec).sum())
                rows.append(
                    {
                        "wave": wave,
                        "wave_date": wave_dates.loc[wave],
                        "category": spec.category,
                        "brand": spec.brand,
                        "metric": spec.metric,
                        "segment_var": seg_var,
                        "segment_value": seg_val,
                        "n": len(seg_df),
                        "weighted_n": float(seg_df[weight_col].sum()) if weight_col and weight_col in seg_df else float(len(seg_df)),
                        "value": value,
                        "success_n": success_uw,
                    }
                )

    result = pd.DataFrame(rows)
    result = result.sort_values(["category", "brand", "metric", "segment_var", "segment_value", "wave"]).reset_index(drop=True)
    return result


def add_rolling_average(tracking_df: pd.DataFrame, window: int = ROLLING_WINDOW) -> pd.DataFrame:
    out = tracking_df.copy()
    group_keys = ["category", "brand", "metric", "segment_var", "segment_value"]
    out["rolling_avg"] = (
        out.groupby(group_keys)["value"]
        .transform(lambda s: s.rolling(window=window, min_periods=1).mean())
    )
    return out


def compute_wow_shifts(tracking_df: pd.DataFrame, alpha: float = 0.05, min_n: int = 40) -> pd.DataFrame:
    """For each (category, brand, metric, segment_var, segment_value) time
    series, test every consecutive wave pair for a statistically
    significant shift via a two-proportion z-test on success_n/n.

    ``min_n`` guards against spurious "significant" results in tiny cells
    (e.g. a niche brand cut by a small age segment): wave-pairs where
    either wave's unweighted base is below this floor are skipped
    entirely, since a z-test on n<30 is not reliable and, run across
    thousands of cells, would otherwise surface noise instead of signal.
    """
    group_keys = ["category", "brand", "metric", "segment_var", "segment_value"]
    rows = []
    for _, group in tracking_df.groupby(group_keys):
        group = group.sort_values("wave")
        prev = None
        for _, curr in group.iterrows():
            if prev is not None and curr["n"] >= min_n and prev["n"] >= min_n:
                test = ztest_proportions(
                    int(curr["success_n"]), int(curr["n"]),
                    int(prev["success_n"]), int(prev["n"]),
                    alpha=alpha,
                )
                rows.append(
                    {
                        "category": curr["category"],
                        "brand": curr["brand"],
                        "metric": curr["metric"],
                        "segment_var": curr["segment_var"],
                        "segment_value": curr["segment_value"],
                        "wave": curr["wave"],
                        "prev_wave": prev["wave"],
                        "value": curr["value"],
                        "prev_value": prev["value"],
                        "change": curr["value"] - prev["value"],
                        "n": curr["n"],
                        "prev_n": prev["n"],
                        "z": test.z,
                        "p_value": test.p_value,
                        "significant": test.significant,
                    }
                )
            prev = curr
    return pd.DataFrame(rows)


def find_headline_shift(shifts_df: pd.DataFrame) -> Optional[pd.Series]:
    """The largest-magnitude significant wave-over-wave move in the
    topline ("Total" segment) numbers -- what an analyst would flag first."""
    sig_total = shifts_df[(shifts_df["significant"]) & (shifts_df["segment_var"] == "Total")]
    if sig_total.empty:
        return None
    idx = sig_total["change"].abs().idxmax()
    return sig_total.loc[idx]


def find_notable_non_obvious_shift(shifts_df: pd.DataFrame) -> Optional[pd.Series]:
    """The segment-level significant move whose magnitude most exceeds
    what the matching topline number moved -- i.e. the story that's easy
    to miss if you only read Total-row numbers."""
    sig = shifts_df[shifts_df["significant"]]
    sig_seg = sig[sig["segment_var"] != "Total"].copy()
    if sig_seg.empty:
        return None

    total_lookup = sig[sig["segment_var"] == "Total"].set_index(["category", "brand", "metric", "wave"])["change"]

    def surprise(row):
        key = (row["category"], row["brand"], row["metric"], row["wave"])
        total_change = total_lookup.get(key, 0.0)
        return abs(row["change"]) - abs(total_change)

    sig_seg["surprise"] = sig_seg.apply(surprise, axis=1)
    idx = sig_seg["surprise"].idxmax()
    return sig_seg.loc[idx]


def run_full_trend_analysis(df: pd.DataFrame, weight_col: Optional[str] = "survey_weight") -> dict:
    tracking = build_tracking_dataset(df, weight_col=weight_col)
    tracking = add_rolling_average(tracking)
    shifts = compute_wow_shifts(tracking)
    return {
        "tracking": tracking,
        "shifts": shifts,
        "headline_shift": find_headline_shift(shifts),
        "notable_shift": find_notable_non_obvious_shift(shifts),
    }


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    cleaned_path = project_root / "data" / "survey_cleaned_weighted.csv"
    df = pd.read_csv(cleaned_path)
    ready = df[df["is_analysis_ready"]].copy()

    result = run_full_trend_analysis(ready)
    print(f"Tracking dataset: {len(result['tracking'])} rows")
    print(f"Wave-over-wave comparisons: {len(result['shifts'])}")
    print(f"Significant shifts: {result['shifts']['significant'].sum()}")

    headline = result["headline_shift"]
    notable = result["notable_shift"]
    if headline is not None:
        print("\nHeadline (topline) shift:")
        print(headline)
    if notable is not None:
        print("\nMost notable non-obvious shift:")
        print(notable)
