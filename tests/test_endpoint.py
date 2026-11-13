import polars as pl
import pytest

from trial_benchmark import endpoint as ep
from trial_benchmark.dating import PROTOCOL


def outcome(measure, description=None):
    return {"measure": measure, "description": description}


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("Overall Survival", ep.SURVIVAL),
        ("Time to first seizure", ep.SURVIVAL),
        ("Progression-free survival per RECIST 1.1", ep.SURVIVAL),
        ("Disease progression, relapse or death", ep.SURVIVAL),
        ("Annualized asthma exacerbation rate", ep.EVENT),
        ("Number of participants with hospitalization for heart failure", ep.EVENT),
        ("Incidence of influenza", ep.EVENT),
        ("Change from baseline in PANSS total score", ep.SYMPTOM),
        ("Pain intensity on a numeric rating scale", ep.SYMPTOM),
        ("Change in HbA1c at week 26", ep.BIOMARKER),
        ("Percent change in LDL-C", ep.BIOMARKER),
        ("Serum creatinine at day 7", ep.BIOMARKER),
        ("Cmax of the study drug", ep.PHARMACOKINETIC),
        ("AUC(0-24) of study drug in plasma", ep.PHARMACOKINETIC),
        ("Change in bone mineral density at the lumbar spine", ep.IMAGING),
        ("Central subfield thickness by OCT", ep.IMAGING),
        ("Number of participants with adverse events", ep.SAFETY),
        ("Dose-limiting toxicity in cycle 1", ep.SAFETY),
        ("Patient satisfaction", ep.OTHER),
    ],
)
def test_each_type_is_found(measure, expected):
    assert ep.classify_outcome(outcome(measure)) == expected


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        # Scouting mistakes: a symptom scale filed as a biomarker.
        ("Change in PANSS-EC", ep.SYMPTOM),
        ("Positive and Negative Syndrome Scale total score", ep.SYMPTOM),
        # A counted exacerbation filed as a biomarker.
        ("Asthma exacerbation rate", ep.EVENT),
        # ACR20 written out is a response on a scale, not an event.
        ("Proportion achieving American College of Rheumatology 20% response", ep.SYMPTOM),
        ("Percentage achieving ACR20 response", ep.SYMPTOM),
        ("PASI 75 response at week 16", ep.SYMPTOM),
        # Physiological tests with no lab keyword.
        ("Trough FEV1 at week 12", ep.BIOMARKER),
        ("Sleep efficiency from polysomnography", ep.BIOMARKER),
        # Glucose curves are not drug levels.
        ("Plasma glucose AUC from 0 to 3 hours", ep.BIOMARKER),
    ],
)
def test_the_known_misses_are_fixed(measure, expected):
    assert ep.classify_outcome(outcome(measure)) == expected


def test_a_blood_test_is_not_a_gene():
    assert ep.classify_outcome(outcome("Change in eGFR")) == ep.BIOMARKER
    assert ep.classify_outcome(outcome("Overall response in EGFR mutation patients")) == ep.EVENT


@pytest.mark.parametrize(
    ("measure", "expected"),
    [
        ("Response at week 12 in a 2-level scheme", ep.EVENT),  # "2-level" is not a lab "level"
        (
            "Phenylketonuria PKU registry visits",
            ep.OTHER,
        ),  # "PK" inside "PKU" is not pharmacokinetics
        ("Hospital stay length", ep.OTHER),  # "OS" inside a word is not overall survival
        ("Biopsy of the scalp", ep.BIOMARKER),
        ("Scarring outcomes", ep.OTHER),  # "scar" is not "score"
    ],
)
def test_words_must_be_whole_words(measure, expected):
    assert ep.classify_outcome(outcome(measure)) == expected


def test_the_description_is_read_only_when_the_title_says_nothing():
    assert ep.classify_outcome(outcome("Primary endpoint", "Change in HbA1c")) == ep.BIOMARKER
    assert (
        ep.classify_outcome(outcome("Change in HbA1c", "Reported with adverse events"))
        == ep.BIOMARKER
    )


def test_a_vague_response_takes_its_meaning_from_the_description():
    scale = outcome(
        "Percentage of responders", "Response on the Investigator Global Assessment scale"
    )
    assert ep.classify_outcome(scale) == ep.SYMPTOM
    assert ep.classify_outcome(outcome("Percentage of responders", "Nothing more")) == ep.EVENT


def test_an_outcome_with_no_text_is_unclear():
    assert ep.classify_outcome(outcome(None, None)) == ep.OTHER
    assert ep.classify_outcome({}) == ep.OTHER


@pytest.mark.parametrize(
    ("text", "weeks"),
    [
        ("12 weeks", 12),
        ("Up to 5 years", 5 * 365.25 / 7),
        ("6 months", 6 * 365.25 / 12 / 7),
        ("28 days", 4),
        ("Week 12", 12),
        ("Weeks 0, 4, 8 and 12", 12),
        ("Day 1 to Day 28", 4),
        ("Baseline to month 6", 6 * 365.25 / 12 / 7),
        ("0-24 hours", 24 / 168),
        ("12-week treatment period", 12),
        ("Week 4 and 24 weeks after the last dose", 24),
        ("Up to 2 years, plus 3 months of follow-up", 2 * 365.25 / 7),
        ("Year 1 to 18 months", 18 * 365.25 / 12 / 7),
        ("Week 4, 24 hours", 4),
        ("Day 1, 2, 4, 8, 12 and 24 hours post-dose", 12 / 7),
        ("Hour 24, 3 days", 3 / 7),
    ],
)
def test_time_frames_are_read_in_weeks(text, weeks):
    assert ep.time_frame_weeks(text) == pytest.approx(weeks)


@pytest.mark.parametrize("text", [None, "", "Through study completion", "Cycle 3", "At baseline"])
def test_a_time_frame_with_no_time_is_null(text):
    assert ep.time_frame_weeks(text) is None


def study(outcomes, first_submit="2015-01-10", start="2015-03"):
    return {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001"},
            "statusModule": {
                "studyFirstSubmitDate": first_submit,
                "startDateStruct": {"date": start} if start else None,
            },
            "outcomesModule": {"primaryOutcomes": outcomes} if outcomes is not None else {},
        }
    }


def test_the_row_describes_the_first_primary_outcome_and_counts_all_of_them():
    row = ep.make_row(
        study([{"measure": "Change in HbA1c", "timeFrame": "26 weeks"}, {"measure": "Weight"}])
    )
    assert row["endpoint_type"] == ep.BIOMARKER
    assert row["time_frame_weeks"] == 26
    assert row["n_primary_outcomes"] == 2


@pytest.mark.parametrize("outcomes", [None, []])
def test_a_trial_with_no_primary_outcome_has_nulls_and_a_count_of_zero(outcomes):
    row = ep.make_row(study(outcomes))
    assert row["endpoint_type"] is None
    assert row["time_frame_weeks"] is None
    assert row["n_primary_outcomes"] == 0


@pytest.mark.parametrize(
    ("first_submit", "start", "expected"),
    [
        ("2015-01-10", "2015-03", True),
        ("2015-03-20", "2015-03", False),  # same month: not clearly before
        ("2016-01-01", "2015-03-05", False),
        ("2015-03-05", "2015-03-05", False),
        ("2015-01-10", None, None),
    ],
)
def test_registered_before_start_needs_a_clear_gap(first_submit, start, expected):
    row = ep.make_row(study([{"measure": "OS"}], first_submit=first_submit, start=start))
    assert row["registered_before_start"] is expected


def test_build_keeps_the_trials_csv_order_and_split_and_demands_every_trial():
    studies = [study([{"measure": "OS"}])]
    table = ep.build(studies, {"NCT00000001": "late"})
    assert table["split"].to_list() == ["late"]
    assert table.schema == pl.Schema(ep.SCHEMA)
    with pytest.raises(ValueError, match="missing"):
        ep.build(studies, {"NCT00000001": "dev", "NCT00000002": "dev"})


def test_every_feature_column_is_declared_with_a_valid_dating():
    assert ep.TABLE == "endpoint"
    assert set(ep.FEATURES) <= set(ep.SCHEMA)
    assert set(ep.FEATURES.values()) == {PROTOCOL}


@pytest.mark.parametrize(
    ("measure", "description", "expected"),
    [
        ("Objective Response Rate", "Measured by CT scan and blood tests", ep.EVENT),
        ("Overall remission", "Judged on a serum marker", ep.EVENT),
        ("Serum concentration of interleukin-6", None, ep.BIOMARKER),
        ("Serum concentration of the study drug", None, ep.PHARMACOKINETIC),
        ("AUC of pain intensity", None, ep.SYMPTOM),
        ("Viral load AUC", None, ep.BIOMARKER),
        ("AUC0-24 of the drug in plasma", None, ep.PHARMACOKINETIC),
    ],
)
def test_explicit_responses_and_drug_levels(measure, description, expected):
    assert ep.classify_outcome(outcome(measure, description)) == expected
