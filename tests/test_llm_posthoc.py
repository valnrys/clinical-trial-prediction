import pytest

from elman_exercise import llm_posthoc as ph

TIME_FRAME_PHRASES = [
    "Up to 30 months (through database cut-off date of 26 Oct 2017)",
    "From randomization until the primary analysis data cut-off date of 05 April 2019",
    "5 years; the median duration of follow-up was 21.8 months",
    "Up to early termination of the study",
    "Until 12 Mar 2019",
    "Maximum duration of follow-up was 22.14 months",
]
PLANNED = [
    "12 weeks",
    "Up to 30 months",
    "Baseline to Week 16",
    "Day 1 to Day 112, then every 28 days",
    "2000 mg per day for 12 weeks",
    "Up to 1000 days",
    "Change from baseline in FEV1 (RECIST 1.1)",
]


@pytest.mark.parametrize("text", TIME_FRAME_PHRASES)
def test_a_time_frame_that_says_what_happened_is_found(text):
    assert ph.has_phrase(text, time_frame=True)


@pytest.mark.parametrize("text", PLANNED)
def test_a_planned_time_frame_is_not_flagged(text):
    assert not ph.has_phrase(text, time_frame=True)


@pytest.mark.parametrize(
    "text",
    [
        "The study did not meet its primary endpoint",
        "Results showed benefit",
        "the study was terminated early",
        "median follow-up was 12 months",
    ],
)
def test_outcome_and_arm_text_with_a_result_phrase_is_found(text):
    assert ph.has_phrase(text)


def test_a_calendar_year_counts_only_in_a_time_frame():
    assert ph.has_phrase("Until 2019", time_frame=True) and not ph.has_phrase("Until 2019")


def test_the_clause_that_carries_the_phrase_is_cut_and_the_rest_kept():
    cut = ph.strip_clauses
    assert (
        cut("Up to 30 months (through database cut-off date of 26 Oct 2017)", True)
        == "Up to 30 months"
    )
    assert cut("5 years; the median duration of follow-up was 21.8 months", True) == "5 years"
    assert cut("Up to early termination of the study", True) is None
    assert (
        cut("Week 8, until the data cut-off date of 1 May 2018, then week 12.", True)
        == "Week 8, then week 12"
    )
    assert cut("Up to 12 weeks.", True) == "Up to 12 weeks."  # untouched text keeps its full stop
    assert cut(None) is None


def record(measure="OS", time_frame="5 years", label="Arm", description=None):
    return {
        "arms": [{"label": label, "description": description, "type": None, "interventions": []}],
        "primary_outcomes": [{"measure": measure, "time_frame": time_frame}],
    }


def test_a_trial_is_flagged_for_a_phrase_in_any_outcome_or_arm_text():
    assert not ph.flagged(record())
    assert ph.flagged(record(time_frame="until the data cut-off date of 1 May 2018"))
    assert ph.flagged(record(description="The study did not meet its endpoint"))
    assert ph.flagged(record(measure="Overall survival, median follow-up was 12 months"))


def test_stripping_a_record_changes_only_flagged_text():
    rec = record(time_frame="5 years; median follow-up was 3 years", description="Plain.")
    out = ph.strip_record(rec)
    assert out["primary_outcomes"][0]["time_frame"] == "5 years"
    assert out["arms"][0]["description"] == "Plain." and out["arms"][0]["label"] == "Arm"
