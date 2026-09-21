from survey_toolkit.narrative_report import generate_narrative_report, save_narrative_report, word_count

REQUIRED_SECTIONS = [
    "## Executive Summary",
    "## The Headline Trend",
    "## The Non-Obvious Finding",
    "## Segment Differences Worth Watching",
    "## Category Notes",
    "## Methodology and Data Quality",
    "## Synthetic Data Disclaimer",
    "## Recommended Next Steps",
]


def test_word_count_and_required_sections(full_pipeline_df):
    report = generate_narrative_report(full_pipeline_df)
    assert 800 <= word_count(report) <= 1200
    for section in REQUIRED_SECTIONS:
        assert section in report
    assert "synthetic" in report.lower()
    assert "no real" in report.lower()

    n_flagged = int(full_pipeline_df["flag_straightliner"].sum())
    assert f"{n_flagged:,}" in report


def test_word_count_helper_and_save(tmp_path):
    assert word_count("one two three") == 3
    assert word_count("") == 0
    path = tmp_path / "nested" / "report.md"
    save_narrative_report("hello world", path)
    assert path.read_text() == "hello world"
