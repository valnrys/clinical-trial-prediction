import polars as pl
import pytest

from trial_benchmark import text
from trial_benchmark.dating import AFTER_START, PROTOCOL, check


def record(nct="NCT1", **protocol):
    base = {
        "identificationModule": {"nctId": nct, "briefTitle": "Brief", "officialTitle": "Official"}
    }
    return {
        "protocolSection": {**base, **protocol},
        "resultsSection": {"outcomeMeasuresModule": "SECRET"},
    }


FULL = record(
    conditionsModule={"conditions": ["Asthma", " ", "COPD"]},
    armsInterventionsModule={"interventions": [{"name": "Drug A"}, {"name": "Placebo"}]},
    outcomesModule={
        "primaryOutcomes": [{"measure": "FEV1 at week 12"}, {"measure": "Exacerbations"}]
    },
    eligibilityModule={"eligibilityCriteria": "  Inclusion: adults  "},
    descriptionModule={"briefSummary": "A study of Drug A."},
)


def test_extract_reads_each_field_and_joins_lists():
    row = text.extract(FULL)
    assert row == {
        "nct_id": "NCT1",
        "title": "Official",
        "conditions": "Asthma; COPD",  # the blank condition is dropped
        "interventions": "Drug A; Placebo",
        "primary_outcomes": "FEV1 at week 12; Exacerbations",
        "eligibility": "Inclusion: adults",
        "summary": "A study of Drug A.",
    }


def test_extract_leaves_missing_fields_null_and_falls_back_to_brief_title():
    row = text.extract(record(identificationModule={"nctId": "NCT2", "briefTitle": "Brief"}))
    assert row["title"] == "Brief"
    assert all(row[c] is None for c in text.ALL_COLUMNS if c != "title")


def test_extract_ignores_the_results_section():
    assert "SECRET" not in str(text.extract(FULL))


def test_build_table_keeps_trial_order_and_refuses_missing_records():
    trials = pl.DataFrame({"nct_id": ["NCT2", "NCT1"], "split": ["dev", "test"]})
    studies = [FULL, record("NCT2"), record("NCT9")]  # NCT9 is not a benchmark trial
    table = text.build_table(trials, iter(studies))
    assert table["nct_id"].to_list() == ["NCT2", "NCT1"]
    assert table["split"].to_list() == ["dev", "test"]
    with pytest.raises(ValueError, match="records found"):
        text.build_table(trials, iter([FULL]))


def test_result_like_shares_counts_phrases_among_present_summaries():
    summaries = pl.Series(["The study met its primary endpoint.", "A plan.", None, "A plan."])
    shares = text.result_like_shares(summaries)
    assert shares["met its primary endpoint"] == pytest.approx(1 / 3)
    assert shares["any of these"] == pytest.approx(1 / 3)
    assert shares["did not show"] == 0


def test_default_columns_leave_out_the_summary_but_the_table_keeps_it():
    assert "summary" not in text.WORD_FIELDS
    assert "summary" in text.ALL_COLUMNS


def test_extract_survives_null_modules_and_lists():
    row = text.extract(
        record(
            conditionsModule={"conditions": None},
            armsInterventionsModule=None,
            outcomesModule={"primaryOutcomes": None},
            eligibilityModule=None,
            descriptionModule={"briefSummary": None},
        )
    )
    assert row["conditions"] is None
    assert row["interventions"] is None
    assert row["primary_outcomes"] is None
    assert row["eligibility"] is None
    assert row["summary"] is None


def test_dating_labels():
    check(text.FEATURES)
    assert text.FEATURES["summary"] == AFTER_START
    assert {c for c, d in text.FEATURES.items() if d == PROTOCOL} == set(text.WORD_FIELDS)
