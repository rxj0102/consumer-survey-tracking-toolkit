"""Cross-tabulated banner tables and significance testing.

Produces the classic "banner table" a market research analyst hands to a
client: a metric (e.g. brand awareness, top-2-box satisfaction) broken out
by a set of demographic/behavioral columns ("the banner"), with two-sample
z-tests for proportions flagging which segment cells differ significantly
from the total.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

DEFAULT_BANNER_VARS: List[str] = ["age_band", "region", "gender", "telecom_carrier_used"]
DEFAULT_ALPHA = 0.05


@dataclass
class ZTestResult:
    count1: int
    n1: int
    count2: int
    n2: int
    p1: float
    p2: float
    z: float
    p_value: float
    significant: bool


def ztest_proportions(count1: int, n1: int, count2: int, n2: int, alpha: float = DEFAULT_ALPHA) -> ZTestResult:
    """Two-sample z-test for a difference in proportions (pooled variance).

    Returns a ZTestResult; z and p_value are NaN when either sample is
    empty or the pooled proportion is degenerate (0 or 1), in which case
    ``significant`` is False.
    """
    if n1 == 0 or n2 == 0:
        return ZTestResult(count1, n1, count2, n2, np.nan, np.nan, np.nan, np.nan, False)

    p1 = count1 / n1
    p2 = count2 / n2
    pooled_p = (count1 + count2) / (n1 + n2)

    if pooled_p in (0.0, 1.0):
        return ZTestResult(count1, n1, count2, n2, p1, p2, 0.0, 1.0, False)

    se = np.sqrt(pooled_p * (1 - pooled_p) * (1 / n1 + 1 / n2))
    if se == 0:
        return ZTestResult(count1, n1, count2, n2, p1, p2, 0.0, 1.0, False)

    z = (p1 - p2) / se
    p_value = 2 * (1 - norm.cdf(abs(z)))
    significant = bool(p_value < alpha)

    return ZTestResult(count1, n1, count2, n2, p1, p2, z, p_value, significant)


def weighted_proportion(mask: pd.Series, weights: Optional[pd.Series] = None) -> tuple:
    """Return (weighted_count, weighted_n) for a boolean mask. Falls back
    to unweighted counts when weights is None."""
    if weights is None:
        return float(mask.sum()), float(len(mask))
    return float((mask * weights).sum()), float(weights.sum())


def generate_banner_table(
    df: pd.DataFrame,
    metric_col: str,
    banner_vars: Iterable[str] = DEFAULT_BANNER_VARS,
    weight_col: Optional[str] = "survey_weight",
    alpha: float = DEFAULT_ALPHA,
) -> pd.DataFrame:
    """Build a banner table for a boolean/binary metric column.

    ``metric_col`` must be interpretable as boolean (True/False, or 1/0);
    NaNs are excluded from denominators (skip-logic-appropriate base).

    Returns a long-format DataFrame with one row per (banner_var, segment)
    plus a "Total" segment for each banner_var, with columns:
    banner_var, segment, n, weighted_n, proportion, z_vs_total, p_value,
    significant (segment differs from the *rest of the total* at alpha).
    """
    weights = df[weight_col] if (weight_col and weight_col in df.columns) else None
    valid = df[metric_col].notna()
    metric = df[metric_col].astype(bool)

    total_count, total_n = weighted_proportion(metric[valid], weights[valid] if weights is not None else None)
    # unweighted counts, used for the significance test (z-test assumes
    # simple random sample counts; weighted N is reported separately)
    total_count_uw = int(metric[valid].sum())
    total_n_uw = int(valid.sum())

    rows = []
    for banner_var in banner_vars:
        for segment in sorted(df[banner_var].dropna().unique(), key=str):
            seg_mask = valid & (df[banner_var] == segment)
            seg_count_uw = int(metric[seg_mask].sum())
            seg_n_uw = int(seg_mask.sum())
            seg_count_w, seg_n_w = weighted_proportion(
                metric[seg_mask], weights[seg_mask] if weights is not None else None
            )

            # compare this segment against the rest of the sample (excludes itself)
            rest_count_uw = total_count_uw - seg_count_uw
            rest_n_uw = total_n_uw - seg_n_uw
            test = ztest_proportions(seg_count_uw, seg_n_uw, rest_count_uw, rest_n_uw, alpha=alpha)

            rows.append(
                {
                    "banner_var": banner_var,
                    "segment": segment,
                    "n": seg_n_uw,
                    "weighted_n": seg_n_w,
                    "proportion": (seg_count_w / seg_n_w) if seg_n_w else np.nan,
                    "z_vs_rest": test.z,
                    "p_value": test.p_value,
                    "significant": test.significant,
                }
            )

    rows.append(
        {
            "banner_var": "ALL",
            "segment": "Total",
            "n": total_n_uw,
            "weighted_n": total_n,
            "proportion": (total_count / total_n) if total_n else np.nan,
            "z_vs_rest": np.nan,
            "p_value": np.nan,
            "significant": False,
        }
    )

    result = pd.DataFrame(rows)
    result.insert(0, "metric", metric_col)
    return result


def generate_mean_banner_table(
    df: pd.DataFrame,
    metric_col: str,
    banner_vars: Iterable[str] = DEFAULT_BANNER_VARS,
    weight_col: Optional[str] = "survey_weight",
) -> pd.DataFrame:
    """Same idea as generate_banner_table but for a continuous/ordinal
    metric (e.g. a Likert or NPS-style item): reports weighted mean by
    segment instead of a proportion. No significance test is attached
    (use trend_tracking's approach or a t-test for that; kept simple here)."""
    weights = df[weight_col] if (weight_col and weight_col in df.columns) else None
    valid = df[metric_col].notna()

    rows = []
    for banner_var in banner_vars:
        for segment in sorted(df[banner_var].dropna().unique(), key=str):
            seg_mask = valid & (df[banner_var] == segment)
            seg_values = df.loc[seg_mask, metric_col]
            if weights is not None and seg_mask.sum() > 0:
                seg_weights = weights[seg_mask]
                mean_val = float(np.average(seg_values, weights=seg_weights))
            else:
                mean_val = float(seg_values.mean()) if len(seg_values) else np.nan
            rows.append(
                {
                    "banner_var": banner_var,
                    "segment": segment,
                    "n": int(seg_mask.sum()),
                    "mean": mean_val,
                }
            )

    total_values = df.loc[valid, metric_col]
    if weights is not None:
        total_mean = float(np.average(total_values, weights=weights[valid])) if valid.sum() else np.nan
    else:
        total_mean = float(total_values.mean()) if len(total_values) else np.nan
    rows.append({"banner_var": "ALL", "segment": "Total", "n": int(valid.sum()), "mean": total_mean})

    result = pd.DataFrame(rows)
    result.insert(0, "metric", metric_col)
    return result


if __name__ == "__main__":
    from pathlib import Path

    project_root = Path(__file__).resolve().parent.parent
    cleaned_path = project_root / "data" / "survey_cleaned_weighted.csv"
    df = pd.read_csv(cleaned_path)
    ready = df[df["is_analysis_ready"]].copy()

    latest_wave = ready["wave"].max()
    latest = ready[ready["wave"] == latest_wave].copy()
    latest["aware_northlink"] = latest["aware_telecom_NorthLink Mobile"]

    banner = generate_banner_table(latest, "aware_northlink")
    print(banner.to_string(index=False))
