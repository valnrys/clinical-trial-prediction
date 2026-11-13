"""Tests for the sponsor table: every trial and every split is covered."""

import polars as pl

from elman_exercise.dating import check
from elman_exercise.sponsor import FEATURES, build_table, sponsor_key


def study(nct_id, sponsor):
    module = {"leadSponsor": {"name": sponsor}} if sponsor else {}
    return {
        "protocolSection": {
            "identificationModule": {"nctId": nct_id},
            "sponsorCollaboratorsModule": module,
        }
    }


def test_key_drops_company_words_and_punctuation():
    assert sponsor_key("Pfizer Inc.") == sponsor_key("Pfizer") == "pfizer"
    assert sponsor_key("Inc") == "inc"
    assert sponsor_key(None) is None


def test_every_trial_and_split_is_covered():
    trials = pl.DataFrame(
        {"nct_id": ["N1", "N2", "N3", "N4"], "split": ["dev", "test", "late", "late"]}
    )
    studies = [study("N1", "A Ltd"), study("N2", "B"), study("N3", "C"), study("N9", "Z")]
    table = build_table(trials, studies)
    assert table["nct_id"].to_list() == ["N1", "N2", "N3", "N4"]
    assert set(table["split"]) == {"dev", "test", "late"}
    # a trial with no sponsor in the record stays, with nulls
    assert table.filter(pl.col("nct_id") == "N4")["sponsor_key"].to_list() == [None]


def test_features_are_labelled():
    check(FEATURES)
    assert set(FEATURES) == {"sponsor_name", "sponsor_key"}
