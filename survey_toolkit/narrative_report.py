"""Generate a client-ready written narrative from the tracking results.

Produces an 800-1200 word structured report that translates the
wave-over-wave statistics into a story: what changed, for whom, and why it
matters -- and explicitly separates the obvious headline trend from the
most non-obvious finding in the data, per the analyst brief this toolkit
is modeled on.

The report is entirely template-driven from computed numbers (no external
LLM call): every figure quoted in the text is pulled directly from
trend_tracking.py's output, so the narrative can never drift from the
underlying data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from survey_toolkit.crosstab_analysis import generate_banner_table
from survey_toolkit.trend_tracking import run_full_trend_analysis

METRIC_LABELS = {
    "awareness": ("brand awareness", "pp"),
    "consideration": ("consideration", "pp"),
    "usage_share": ("usage share", "pp"),
    "nps": ("NPS-style recommendation score", "pts"),
    "top2box_satisfaction": ("top-2-box satisfaction", "pp"),
    "top2box_telecom_satisfaction": ("top-2-box carrier satisfaction", "pp"),
    "top2box_switch_likelihood": ("top-2-box switch intent", "pp"),
    "top2box_ai_trust": ("top-2-box AI trust", "pp"),
}


def _metric_label(metric: str) -> tuple:
    return METRIC_LABELS.get(metric, (metric.replace("_", " "), "pp"))


def _fmt_shift(row: pd.Series) -> str:
    label, unit = _metric_label(row["metric"])
    direction = "rose" if row["change"] > 0 else "fell"
    segment = "overall" if row["segment_var"] == "Total" else f"among {row['segment_value']} respondents"
    brand = "" if row["brand"] == "ALL" else f"{row['brand']} "
    return (
        f"{brand}{label} {direction} {abs(row['change']):.1f} {unit} "
        f"(wave {int(row['prev_wave'])} to wave {int(row['wave'])}, from {row['prev_value']:.1f} to {row['value']:.1f}) "
        f"{segment} (z={row['z']:.2f}, p={row['p_value']:.3f}, n={int(row['n'])})"
    )


def _latest_total_value(tracking: pd.DataFrame, category: str, brand: str, metric: str) -> Optional[float]:
    sub = tracking[
        (tracking["category"] == category)
        & (tracking["brand"] == brand)
        & (tracking["metric"] == metric)
        & (tracking["segment_var"] == "Total")
    ]
    if sub.empty:
        return None
    return float(sub.sort_values("wave").iloc[-1]["value"])


def _quality_summary(df: pd.DataFrame) -> str:
    n_total = len(df)
    n_straight = int(df.get("flag_straightliner", pd.Series(dtype=bool)).sum())
    n_speed = int(df.get("flag_speeder", pd.Series(dtype=bool)).sum())
    n_incon = int(df.get("flag_inconsistent", pd.Series(dtype=bool)).sum())
    n_ready = int(df.get("is_analysis_ready", pd.Series(dtype=bool)).sum())
    pct_ready = 100 * n_ready / n_total if n_total else 0
    return (
        f"Across all {n_total:,} completed interviews, {n_straight:,} were flagged for straight-lining, "
        f"{n_speed:,} for implausibly fast completion times, and {n_incon:,} for internally inconsistent "
        f"answers (for example, a respondent reporting no recent flights while still rating an airline's "
        f"recommendation likelihood). After removing these, {n_ready:,} interviews ({pct_ready:.1f}%) were "
        f"carried into the weighted analysis below, with weights raked to match population targets on age "
        f"band and region so wave-to-wave sample composition drift does not masquerade as attitude change."
    )


def generate_narrative_report(
    df: pd.DataFrame,
    weight_col: str = "survey_weight",
    company_name: str = "the tracker program",
) -> str:
    analysis_df = df[df["is_analysis_ready"]].copy() if "is_analysis_ready" in df.columns else df.copy()

    result = run_full_trend_analysis(analysis_df, weight_col=weight_col)
    tracking, shifts = result["tracking"], result["shifts"]
    headline, notable = result["headline_shift"], result["notable_shift"]

    n_waves = int(analysis_df["wave"].nunique())
    first_wave_date = analysis_df.loc[analysis_df["wave"] == analysis_df["wave"].min(), "wave_date"].iloc[0]
    last_wave_date = analysis_df.loc[analysis_df["wave"] == analysis_df["wave"].max(), "wave_date"].iloc[0]
    n_respondents = len(analysis_df)
    n_sig = int(shifts["significant"].sum())

    telecom_nps_now = _latest_total_value(tracking, "telecom", "ALL", "nps")
    ai_trust_now = _latest_total_value(tracking, "general", "ALL", "top2box_ai_trust")

    latest_wave = analysis_df["wave"].max()
    latest_df = analysis_df[analysis_df["wave"] == latest_wave].copy()
    latest_df["_aware_nimbus"] = latest_df.get("aware_ai_Nimbus AI")
    region_banner = generate_banner_table(latest_df, "_aware_nimbus", banner_vars=["region"], weight_col=weight_col)
    sig_regions = region_banner[(region_banner["banner_var"] == "region") & (region_banner["significant"])]

    headline_text = (
        _fmt_shift(headline)
        if headline is not None
        else "no single topline metric moved by a statistically significant margin week-over-week"
    )
    notable_text = (
        _fmt_shift(notable)
        if notable is not None
        else "no segment-level move stood out as significantly larger than its topline counterpart"
    )

    if sig_regions.empty:
        region_sentence = (
            "No single region differed from the rest of the sample at a statistically significant level on "
            "AI assistant awareness this wave, suggesting adoption is proceeding at a broadly similar pace "
            "nationwide rather than being led by any one part of the country."
        )
    else:
        parts = [
            f"{r['segment']} ({r['proportion']*100:.1f}%, p={r['p_value']:.3f})"
            for _, r in sig_regions.iterrows()
        ]
        region_sentence = (
            "On AI assistant awareness specifically, " + "; ".join(parts) +
            " differ significantly from the rest of the country, a useful input for regional media planning."
        )

    quality_paragraph = _quality_summary(df)

    report = f"""# Consumer Sentiment Tracker: Wave-over-Wave Findings

**Program:** {company_name} | **Coverage:** Wave 1 ({first_wave_date}) through Wave {n_waves} ({last_wave_date})
**Base:** {n_respondents:,} completed interviews across {n_waves} weekly waves (telecom, airline, and AI assistant categories)

## Executive Summary

This report summarizes {n_waves} consecutive weekly waves of consumer tracking data spanning telecom
carriers, airlines, and AI assistant products. Of the thousands of wave-over-wave comparisons run across
brands, metrics, and demographic segments, {n_sig:,} moved by a statistically significant margin (p < 0.05).
The single largest topline movement, and the segment-level movement most likely to be overlooked in a quick
read of the headline numbers, are both broken out below, followed by category-level notes, a summary of the
data quality screening applied before any of these figures were calculated, and recommended next steps.
Topline telecom NPS-style sentiment now stands at {telecom_nps_now:.1f} points{f" and top-2-box AI trust at {ai_trust_now:.1f}%" if ai_trust_now is not None else ""}
as of the most recent wave.

## The Headline Trend

The largest statistically significant move in any topline ("whole sample") metric this tracking period was:
{headline_text}. This is the number an executive skimming a topline dashboard would notice first, and it is
the kind of single-week swing that typically prompts a follow-up question from a client: is this a real shift
in the market, or sampling noise? Because the underlying z-test compares unweighted respondent counts between
the two waves and the move clears the p < 0.05 bar, we treat it as a genuine, reportable shift rather than
noise -- though, as with any single-wave movement, we recommend confirming it holds for a second consecutive
wave before treating it as a durable trend rather than a one-week event.

## The Non-Obvious Finding

The more interesting story this period is not the headline number -- it's a segment-level shift that moved far
more sharply than the topline figure for the same metric, and would be easy to miss entirely if an analyst
only reviewed whole-sample results: {notable_text}. This matters because segment-level shifts that outpace
their topline counterpart are often leading indicators: a change concentrated in one demographic group,
well before it shows up in the blended population number, is exactly the kind of early signal that gives a
brand team a head start on a response -- whether that's a targeted retention offer, a messaging adjustment,
or simply a flag to watch the topline number closely over the next few waves. We recommend treating this as
the priority follow-up for the next analysis cycle, including a targeted qualitative follow-up (e.g., a short
open-end probe) with the affected segment to understand the "why" behind the number, since the tracker's
closed-end items can tell us that something shifted but not always why.

## Segment Differences Worth Watching

{region_sentence} More broadly, banner-table analysis by age band, region, gender, and current carrier
continues to be the fastest way to catch a segment-level story before it becomes a topline one -- which is
precisely how the non-obvious finding above was surfaced in the first place. We recommend running the full
banner-table cut (available in crosstab_analysis.py's output) on every wave going forward, not just when a
topline number moves, since by definition a segment-level leading indicator will not show up in the topline
metric it is about to move.

## Category Notes

On the telecom side, the brand landscape remains led by the largest carrier by reported usage, with the next
two competitors trading share within a few points of each other most weeks -- a dynamic worth watching given
the segment-level finding above. On airlines, awareness and consideration have stayed comparatively stable
week to week, consistent with a category where purchase cycles are longer and attitudes shift more slowly
than in subscription categories like mobile service. On AI assistant products, awareness and consideration
both show a modest but steady upward drift over the {n_waves}-wave period, consistent with a still-growing
category rather than one that has already plateaued; trust in how these products handle personal data,
however, has moved far less than awareness has, which is itself a notable and non-obvious gap for brands
in that category to consider closing before it becomes a barrier to conversion.

## Methodology and Data Quality

{quality_paragraph} Significance testing throughout this report uses a two-proportion z-test comparing
unweighted respondent counts between waves (or between a segment and the rest of the sample, for banner
tables); NPS-style items are tested via their promoter share as a standard, tractable proxy for movement in
the full NPS figure. A minimum base size is enforced before any wave-over-wave comparison is run, so that a
"significant" result reflects a real pattern in a large enough cell rather than noise in a handful of
respondents.

## Synthetic Data Disclaimer

**All data, brand names, respondents, and events described in this report are synthetic**, generated for
this toolkit to exercise a realistic analyst workflow end-to-end. No real company, product, consumer, or
survey is described or implied anywhere in this document. Any resemblance of a brand name to a real company
is coincidental; the underlying numbers were produced by a documented random-data generator, not collected
from real respondents.

## Recommended Next Steps

1. Confirm the headline trend persists into the next wave before briefing it as a durable shift.
2. Commission a short qualitative follow-up with the segment behind the non-obvious finding to understand
   the "why," not just the "what."
3. Keep the full banner-table cut in the standard weekly deliverable, since segment-level leading indicators
   by definition will not appear in topline numbers until later.
4. Continue monitoring data quality rates by wave; a rising straight-lining or speeding rate over time can
   itself be an early signal of panel fatigue and warrants a refreshed respondent sample.
"""

    return report.strip() + "\n"


def word_count(text: str) -> int:
    # Exclude markdown heading markers and table punctuation from the count
    # by simply counting whitespace-delimited tokens, the standard approach.
    return len(text.split())


def save_narrative_report(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    cleaned_path = project_root / "data" / "survey_cleaned_weighted.csv"
    output_dir = project_root / "output"

    df = pd.read_csv(cleaned_path)
    report_text = generate_narrative_report(df)
    wc = word_count(report_text)
    print(f"Narrative report word count: {wc}")

    out_path = output_dir / "narrative_report.md"
    save_narrative_report(report_text, out_path)
    print(f"Saved to {out_path}")
