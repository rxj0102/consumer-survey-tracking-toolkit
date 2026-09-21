# Consumer Survey Tracking Toolkit

A Python toolkit that models the end-to-end workflow of a market research
analyst running a **recurring weekly consumer-sentiment tracker**, in the
spirit of the telecom/airline/AI trackers that firms like YouGov, Morning
Consult, or an internal insights team run continuously. It goes from
questionnaire design through synthetic fieldwork, data cleaning, banner
tables, wave-over-wave trend detection, a BI-ready export, and a written
narrative report -- the full pipeline, with realistic mess baked in.

> **Synthetic data disclaimer.** Every respondent, brand, and data point in
> this repository is synthetically generated. `NorthLink Mobile`, `Meridian
> Airways`, `Nimbus AI`, and every other brand name here are invented for
> this toolkit and do not refer to any real company or product. No real
> survey, panel, or respondent was involved in producing any file in this
> repository. This project exists to demonstrate an analytical workflow,
> not to report on any real market.

## The scenario

A fictional insights team runs a weekly tracker across three categories --
**mobile carriers, airlines, and AI assistant products** -- to monitor
brand awareness, consideration, usage, satisfaction, and NPS-style
sentiment. Each week (a "wave") a fresh sample of ~500-800 respondents
completes a 20-question survey. The program has been running for 14
consecutive weeks. Midway through, one carrier suffers a service issue
that dents its numbers -- exactly the kind of story a real tracker is
built to catch.

Real tracker data is never clean: some respondents rush through, some
straight-line every rating, some give internally contradictory answers.
This toolkit's synthetic data generator deliberately injects that noise,
and the rest of the pipeline is built to catch and correct for it -- the
same way a real analyst would.

## Pipeline / module guide

Each module is runnable standalone (`python3 -m survey_toolkit.<module>`)
and also usable as a library. `run_pipeline.py` runs the whole sequence.

| # | Module | What it does |
|---|--------|---------------|
| 1 | `questionnaire_design.py` | The 20-question instrument: screeners, Likert attitude items, brand awareness/consideration/usage funnel questions, NPS-style items, and open-end verbatim placeholders, with skip-pattern logic and quota targets (age, region, carrier). |
| 2 | `survey_data_generator.py` | Generates 14 weekly waves of synthetic respondents consistent with the questionnaire, including realistic demographics, quota-based sampling, and deliberately injected straight-lining, speeding, and internally inconsistent responses. |
| 3 | `data_cleaning.py` | Flags straight-liners, speeders, and internally inconsistent respondents; applies iterative-proportional-fitting ("raking") demographic weights so the sample matches population targets. |
| 4 | `crosstab_analysis.py` | Builds banner tables (a metric x demographic/behavioral segments) and runs two-proportion z-tests to flag statistically meaningful segment differences. |
| 5 | `trend_tracking.py` | Builds a long-format wave-by-segment tracking dataset, computes rolling averages, tests every wave-over-wave move for significance, and identifies the single most notable shift -- explicitly distinguishing the obvious topline move from a non-obvious segment-level one. |
| 6 | `dashboard_export.py` | Exports the tracking dataset as a BI-tool-ready long-format CSV/JSON, plus pre-aggregated summary tables for a quick executive read. |
| 7 | `narrative_report.py` | Assembles an 800-1,200 word, fully data-driven client narrative: what changed, for whom, why it matters, and the non-obvious finding worth flagging. |

## Quickstart

```bash
pip install -r requirements.txt
python3 run_pipeline.py
```

This writes raw and cleaned survey files to `./data/` and every
deliverable (banner table, dashboard dataset, summary tables, narrative
report) to `./output/`. Both directories are gitignored since everything
in them is reproducible from the generator's fixed random seed
(`DEFAULT_SEED = 42` in `survey_data_generator.py`).

To run an individual stage:

```bash
python3 -m survey_toolkit.survey_data_generator
python3 -m survey_toolkit.data_cleaning
python3 -m survey_toolkit.crosstab_analysis
python3 -m survey_toolkit.trend_tracking
python3 -m survey_toolkit.dashboard_export
python3 -m survey_toolkit.narrative_report
```

## Methodology notes

- **Quota sampling.** Each wave samples respondents against marginal
  quota targets for age band, region, gender, income, and current mobile
  carrier (`questionnaire_design.py`). Small per-wave sampling variance is
  intentional and realistic.
- **Data quality screening.** `data_cleaning.py` flags three classic
  survey data-quality problems: straight-lining (identical rating across
  every Likert item), speeding (implausibly fast completion time), and
  internal inconsistency (e.g. an answer present where skip logic should
  have hidden it, or a brand "considered" without ever being marked
  "aware of"). Flagged respondents are excluded from the analysis-ready
  subset rather than silently kept.
- **Weighting.** Demographic weights are computed per wave via iterative
  proportional fitting (raking) against age-band and region population
  targets, so shifts in sample composition wave-to-wave don't masquerade
  as real attitude change.
- **Significance testing.** All significance testing is a two-proportion
  z-test on unweighted respondent counts (`crosstab_analysis.ztest_proportions`).
  NPS-style items are tested via their promoter share, a standard,
  tractable proxy for movement in the full NPS figure. Wave-over-wave
  tests enforce a minimum base size (`min_n`) before testing a cell, since
  a z-test on a handful of respondents is not reliable and, run across
  thousands of brand/metric/segment cells, would otherwise surface noise
  instead of signal.
- **Trend detection.** `trend_tracking.py` separates the *headline* shift
  (the largest significant topline move) from the *non-obvious* shift (the
  segment-level move whose magnitude most exceeds what the matching
  topline number did) -- the latter is the kind of early, easy-to-miss
  signal a real tracker program exists to catch.

## Project layout

```
survey_toolkit/
    questionnaire_design.py
    survey_data_generator.py
    data_cleaning.py
    crosstab_analysis.py
    trend_tracking.py
    dashboard_export.py
    narrative_report.py
tests/
    test_questionnaire_design.py
    test_survey_data_generator.py
    test_data_cleaning.py
    test_crosstab_analysis.py
    test_trend_tracking.py
    test_dashboard_export.py
    test_narrative_report.py
run_pipeline.py
requirements.txt
```

## Testing

```bash
pip install -r requirements.txt
python3 -m pytest tests/ -v
```

The suite covers cleaning/flagging logic (straight-lining, speeding,
internal-inconsistency detection and recall against known-injected noise),
weighting math (raking convergence against population targets), the
significance-testing math (z-test correctness at known inputs), and
trend-detection logic (rolling averages, minimum-base-size filtering,
headline vs. non-obvious shift selection).
