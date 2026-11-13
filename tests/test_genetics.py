import math
from datetime import date

import polars as pl
import pytest

from trial_benchmark import genetics as gn
from trial_benchmark import genetics_summary as gs
from trial_benchmark import opentargets as ot

DRUG_STRUCT = pl.Struct({"drugFromSource": pl.String, "drugId": pl.String})
DISEASE_STRUCT = pl.Struct({"diseaseFromSource": pl.String, "diseaseId": pl.String})
SERIES = pl.List(
    pl.Struct(
        {"year": pl.Int32, "associationScore": pl.Float64, "novelty": pl.Float64,
         "yearlyEvidenceCount": pl.Int64}
    )
)  # fmt: skip


def report(nct, drugs, diseases):
    return {
        "id": nct,
        "source": "ClinicalTrials.gov",
        "drugs": [{"drugFromSource": d, "drugId": d} for d in drugs],
        "diseases": [{"diseaseFromSource": d, "diseaseId": d} for d in diseases],
        "trialStartDate": date(2015, 1, 1),
    }


def reports(*rows):
    return pl.DataFrame(
        list(rows),
        schema={"id": pl.String, "source": pl.String, "drugs": pl.List(DRUG_STRUCT),
                "diseases": pl.List(DISEASE_STRUCT), "trialStartDate": pl.Date},
    )  # fmt: skip


def series(**by_year):
    """A history from keyword years: series(y2009=0.0, y2015=0.4). The undated entry is added."""
    entries = [
        {"year": int(k[1:]), "associationScore": v, "novelty": 0.0, "yearlyEvidenceCount": None}
        for k, v in by_year.items()
    ]
    latest = max(by_year.items(), key=lambda kv: kv[0])[1]
    entries.append(
        {"year": None, "associationScore": latest, "novelty": None, "yearlyEvidenceCount": None}
    )
    return entries


def associations(*rows):
    """Rows of (target, disease, history); today's score is the latest year's."""
    data = [
        {"targetId": t, "diseaseId": d, "associationScore": h[-1]["associationScore"],
         "timeseries": h}
        for t, d, h in rows
    ]  # fmt: skip
    return pl.DataFrame(
        data,
        schema={"targetId": pl.String, "diseaseId": pl.String, "associationScore": pl.Float64,
                "timeseries": SERIES},
    )  # fmt: skip


DRUG_TARGETS = pl.DataFrame({"drugId": ["D1", "D2", "D3"], "targetId": ["T1", "T2", "T3"]})
MOLECULES = pl.DataFrame(
    {"id": ["D1", "D2", "D3"], "name": ["a", "b", "c"],
     "drugType": ["Small molecule", "Antibody", "Unknown"]}
)  # fmt: skip


def trials(*rows):
    return pl.DataFrame(
        [{"nct_id": n, "split": "dev", "start_year": y} for n, y in rows],
        schema={"nct_id": pl.String, "split": pl.String, "start_year": pl.Int64},
    )


def build(trial_rows, report_rows, assoc_rows):
    return gn.build_table(
        trials(*trial_rows), reports(*report_rows), DRUG_TARGETS, associations(*assoc_rows),
        MOLECULES,
    ).to_dicts()  # fmt: skip


def by_id(rows):
    return {r["nct_id"]: r for r in rows}


# --- the join ---------------------------------------------------------------------------


def test_pairs_are_every_target_of_every_drug_against_every_disease():
    rep = reports(report("NCT1", ["D1", "D2"], ["E1", "E2"]))
    drugs = gn.trial_drugs(rep, ["NCT1"])
    pairs = gn.trial_pairs(drugs, gn.trial_diseases(rep, ["NCT1"]), DRUG_TARGETS)
    assert set(pairs.select("targetId", "diseaseId").rows()) == {
        ("T1", "E1"), ("T1", "E2"), ("T2", "E1"), ("T2", "E2"),
    }  # fmt: skip


def test_a_trial_not_in_open_targets_is_unmapped_with_null_scores():
    row = by_id(build([("NCT9", 2015)], [report("NCT1", ["D1"], ["E1"])], []))["NCT9"]
    assert row["mapped"] is False
    assert row["genetic_score_today"] is None
    assert row["supported_today"] is None and row["supported_at_start"] is None
    assert row["few_targets"] is None
    assert row["n_pairs"] == 0


def test_a_trial_whose_drug_has_no_target_is_unmapped():
    rows = by_id(build([("NCT1", 2015)], [report("NCT1", ["D9"], ["E1"])], []))
    assert rows["NCT1"]["mapped"] is False


def test_mapped_without_any_genetic_evidence_scores_zero_not_null():
    row = by_id(build([("NCT1", 2015)], [report("NCT1", ["D1"], ["E1"])], []))["NCT1"]
    assert row["mapped"] is True
    assert row["genetic_score_today"] == 0 and row["genetic_score_at_start"] == 0
    assert row["supported_today"] is False and row["supported_at_start"] is False


def test_the_trial_score_is_the_maximum_over_its_pairs():
    assoc = [
        ("T1", "E1", series(y2009=0.0, y2014=0.2)),
        ("T2", "E1", series(y2009=0.0, y2014=0.6)),
        ("T3", "E1", series(y2009=0.0, y2014=0.4)),
    ]
    rep = report("NCT1", ["D1", "D2"], ["E1"])
    row = by_id(build([("NCT1", 2016)], [rep], assoc))["NCT1"]
    assert row["genetic_score_today"] == pytest.approx(0.6)
    assert row["n_targets"] == 2 and row["n_pairs"] == 2


def test_the_start_score_uses_the_latest_year_strictly_before_our_start_year():
    history = series(y2009=0.0, y2012=0.3, y2014=0.5, y2016=0.9)
    assoc = [("T1", "E1", history)]
    rep = report("NCT1", ["D1"], ["E1"])
    # Open Targets' own start date (2015-01-01) is ignored; our start_year decides.
    assert by_id(build([("NCT1", 2014)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.3
    assert by_id(build([("NCT1", 2015)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.5
    assert by_id(build([("NCT1", 2017)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.9


def test_evidence_that_arrives_in_the_start_year_does_not_count_at_start():
    assoc = [("T1", "E1", series(y2009=0.0, y2015=0.7))]
    row = by_id(build([("NCT1", 2015)], [report("NCT1", ["D1"], ["E1"])], assoc))["NCT1"]
    assert row["genetic_score_at_start"] == 0
    assert row["genetic_score_today"] == pytest.approx(0.7)
    assert row["supported_today"] is True and row["supported_at_start"] is False


def test_a_start_year_before_the_whole_history_means_no_evidence_yet_so_zero():
    assoc = [("T1", "E1", series(y2009=0.0, y2014=0.5))]
    rep = report("NCT1", ["D1"], ["E1"])
    row = by_id(build([("NCT1", 2005)], [rep], assoc))["NCT1"]
    assert row["genetic_score_at_start"] == 0 and row["supported_at_start"] is False
    assert row["genetic_score_today"] == pytest.approx(0.5)


def test_a_missing_start_year_makes_the_start_score_unknown():
    assoc = [("T1", "E1", series(y2009=0.0, y2014=0.5))]
    row = by_id(build([("NCT1", None)], [report("NCT1", ["D1"], ["E1"])], assoc))["NCT1"]
    assert row["genetic_score_at_start"] is None and row["genetic_score_today"] > 0


def test_few_targets_is_three_or_fewer():
    drug_targets = pl.DataFrame({"drugId": [f"D{i}" for i in range(5)],
                                 "targetId": [f"T{i}" for i in range(5)]})  # fmt: skip
    rep = reports(report("NCT1", ["D0", "D1", "D2"], ["E1"]),
                  report("NCT2", ["D0", "D1", "D2", "D3"], ["E1"]))  # fmt: skip
    table = gn.build_table(
        trials(("NCT1", 2015), ("NCT2", 2015)), rep, drug_targets, associations(),
        pl.DataFrame({"id": [], "name": [], "drugType": []}, schema=pl.Schema(
            {"id": pl.String, "name": pl.String, "drugType": pl.String})),
    )  # fmt: skip
    flags = dict(zip(table["nct_id"], table["few_targets"], strict=True))
    assert flags == {"NCT1": True, "NCT2": False}


# --- drug type ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("types", "expected"),
    [
        (["Small molecule"], "small molecule"),
        (["Antibody", "Unknown"], "antibody"),
        (["Enzyme", "Protein"], "protein"),
        (["Small molecule", "Antibody"], "mixed"),
        (["Unknown", None], "unknown"),
        ([], "unknown"),
        (["Oligosaccharide"], "other"),
    ],
)
def test_drug_types_collapse_to_one_label(types, expected):
    assert gn.collapse_drug_types(types) == expected


def test_a_trial_with_one_small_molecule_and_one_antibody_is_mixed():
    rep = report("NCT1", ["D1", "D2"], ["E1"])
    row = by_id(build([("NCT1", 2015)], [rep], []))["NCT1"]
    assert row["drug_type"] == "mixed"


# --- grading -------------------------------------------------------------------------------


def test_bands_change_at_the_cut_points():
    cuts = (0.2, 0.5)
    got = [gn.band(s, cuts) for s in (0.0, 0.01, 0.2, 0.2001, 0.5, 0.5001, 1.0)]
    assert got == ["none", "low", "low", "middle", "middle", "high", "high"]


def test_an_unmapped_trial_has_no_band():
    assert gn.band(None, (0.2, 0.5)) is None


def test_terciles_cut_the_positive_scores_into_thirds_and_ignore_zeros():
    scores = [0.0, 0.0] + [i / 100 for i in range(1, 10)]
    low, high = gn.tercile_cuts(scores)
    assert (low, high) == (0.03, 0.06)
    bands = [gn.band(s, (low, high)) for s in scores if s > 0]
    assert [bands.count(b) for b in ("low", "middle", "high")] == [3, 3, 3]


def test_the_fixed_cut_points_are_in_order_and_positive():
    for low, high in (gn.TODAY_CUTS, gn.START_CUTS):
        assert 0 < low < high < 1


# --- the history column ---------------------------------------------------------------------


def test_the_timeseries_check_reports_agreement_on_a_clean_history():
    assoc = associations(("T1", "E1", series(y2009=0.0, y2014=0.5, y2016=0.9)))
    text = gn.check_timeseries(assoc)
    assert "equals today's score:      1/1" in text
    assert "never fall from one year to the next: 1/1" in text


# --- what is read from Open Targets ----------------------------------------------------------


def test_the_exact_columns_read_from_each_open_targets_table():
    # Any change to what is read must change this test: nothing about trial status, stop
    # reasons, clinical stage or approval may ever be added without it being seen.
    assert ot.WANTED_COLUMNS == {
        "clinical_report": ("id", "source", "drugs", "diseases", "trialStartDate"),
        "clinical_target": ("drugId", "targetId"),
        "drug_molecule": ("id", "name", "drugType"),
        "target": ("id", "approvedSymbol"),
    }
    assert ot.ASSOCIATION_COLUMNS == (
        "diseaseId", "targetId", "aggregationType", "aggregationValue", "associationScore",
        "timeseries",
    )  # fmt: skip


def test_a_small_table_is_written_with_only_its_wanted_columns(tmp_path, monkeypatch):
    full = pl.DataFrame(
        {"id": ["nct1", "x"], "source": ["ClinicalTrials.gov", "FDA"],
         "drugs": [[{"drugFromSource": "a", "drugId": "D1"}]] * 2,
         "diseases": [[{"diseaseFromSource": "b", "diseaseId": "E1"}]] * 2,
         "trialStartDate": [date(2015, 1, 1)] * 2,
         "trialOverallStatus": ["COMPLETED", "TERMINATED"],
         "trialWhyStopped": [None, "slow"], "clinicalStage": ["PHASE_3", "APPROVAL"]}
    )  # fmt: skip
    seen = {}

    def fake_download(url, destination):
        full.write_parquet(destination)

    real_read = pl.read_parquet

    def spy_read(path, columns=None, **kw):
        seen["columns"] = columns
        return real_read(path, columns=columns, **kw)

    monkeypatch.setattr(ot, "list_parquet_files", lambda table: ["00000000.parquet"])
    monkeypatch.setattr(ot, "download", fake_download)
    monkeypatch.setattr(ot.pl, "read_parquet", spy_read)
    path = ot.fetch_small_table("clinical_report", tmp_path)
    assert seen["columns"] == list(ot.WANTED_COLUMNS["clinical_report"])
    written = real_read(path)
    assert set(written.columns) == set(ot.WANTED_COLUMNS["clinical_report"])
    assert written["id"].to_list() == ["NCT1"]
    assert not (tmp_path / "_full_clinical_report.parquet").exists()


def test_the_genetic_filter_keeps_only_genetic_rows_for_our_targets():
    part = pl.DataFrame(
        {"diseaseId": ["E1"] * 4, "targetId": ["T1", "T1", "T9", "T2"],
         "aggregationType": ["datatypeId"] * 4,
         "aggregationValue": ["genetic_association", "literature", "genetic_association",
                              "genetic_association"],
         "associationScore": [0.5, 0.1, 0.9, 0.3], "timeseries": [[]] * 4},
        schema_overrides={"timeseries": SERIES},
    )  # fmt: skip
    kept = ot.genetic_rows_for(part.lazy(), ["T1", "T2"])
    assert sorted(kept["targetId"].to_list()) == ["T1", "T2"]
    assert kept.columns == [
        "diseaseId", "targetId", "aggregationValue", "associationScore", "timeseries",
    ]  # fmt: skip
    assert set(kept["aggregationValue"]) == {"genetic_association"}


def test_a_score_that_dips_in_the_history_is_read_as_the_latest_year_before_start():
    # Scores can be re-weighted downwards; the at-start score is the latest year's, not the max.
    assoc = [("T1", "E1", series(y2009=0.0, y2012=0.8, y2014=0.5, y2018=0.7))]
    rep = report("NCT1", ["D1"], ["E1"])
    assert by_id(build([("NCT1", 2013)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.8
    assert by_id(build([("NCT1", 2016)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.5
    assert by_id(build([("NCT1", 2020)], [rep], assoc))["NCT1"]["genetic_score_at_start"] == 0.7


def test_the_top_target_and_supported_targets_come_from_the_scored_pairs():
    assoc = [
        ("T1", "E1", series(y2009=0.0, y2014=0.2)),
        ("T2", "E1", series(y2009=0.0, y2014=0.6)),
    ]
    row = by_id(build([("NCT1", 2016)], [report("NCT1", ["D1", "D2", "D3"], ["E1"])], assoc))[
        "NCT1"
    ]
    assert row["top_target_today"] == "T2"
    assert row["supported_targets_today"] == ["T1", "T2"] and row["target_ids"] == [
        "T1",
        "T2",
        "T3",
    ]


# --- genetic rows only, and a crashed download -----------------------------------------------


def association_part(values):
    n = len(values)
    return pl.DataFrame(
        {"diseaseId": ["E1"] * n, "targetId": ["T1"] * n,
         "aggregationType": ["datatypeId"] * n, "aggregationValue": values,
         "associationScore": [0.5] * n, "timeseries": [[]] * n},
        schema_overrides={"timeseries": SERIES},
    )  # fmt: skip


def test_a_non_genetic_row_in_the_kept_file_is_rejected():
    gn.check_genetic_only(association_part(["genetic_association"]))
    with pytest.raises(ValueError, match="not genetic_association"):
        gn.check_genetic_only(association_part(["genetic_association", "clinical"]))
    with pytest.raises(ValueError, match="download it again"):
        gn.check_genetic_only(association_part(["genetic_association"]).drop("aggregationValue"))


def test_a_resumed_download_deletes_leftovers_and_ignores_half_written_files(tmp_path, monkeypatch):
    parts = tmp_path / "_association_parts"
    parts.mkdir()
    raw = tmp_path / "_raw_a.parquet"
    raw.write_bytes(b"outcome-bearing raw part left by a crash")
    (tmp_path / "_raw_a.parquet.part").write_bytes(b"x")
    (parts / "kept_a.parquet.tmp").write_bytes(b"half written")
    ot.genetic_rows_for(association_part(["genetic_association"]).lazy(), ["T1"]).write_parquet(
        parts / "kept_b.parquet"
    )
    fetched = []

    def fake_download(url, destination):
        fetched.append(destination.name)
        association_part(["genetic_association", "clinical"]).write_parquet(destination)

    monkeypatch.setattr(ot, "list_parquet_files", lambda table: ["a.parquet", "b.parquet"])
    monkeypatch.setattr(ot, "download", fake_download)
    final = ot.fetch_genetic_associations(["T1"], tmp_path)
    assert fetched == ["_raw_a.parquet"]  # part b was already finished
    assert sorted(p.name for p in tmp_path.iterdir()) == ["genetic_association.parquet"]
    result = pl.read_parquet(final)
    assert result.height == 2 and set(result["aggregationValue"]) == {"genetic_association"}
    gn.check_genetic_only(result)


def test_clean_leftovers_keeps_finished_parts(tmp_path):
    parts = tmp_path / "p"
    parts.mkdir()
    (tmp_path / "_full_x.parquet").write_bytes(b"1")
    (parts / "kept_a.parquet").write_bytes(b"1")
    (parts / "kept_b.parquet.tmp").write_bytes(b"1")
    ot.clean_leftovers(tmp_path, parts)
    assert [p.name for p in parts.iterdir()] == ["kept_a.parquet"]
    assert not (tmp_path / "_full_x.parquet").exists()


# --- the investigational-drug flag ------------------------------------------------------------


def named_report(nct, drugs, diseases):
    row = report(nct, [d for d, _ in drugs], diseases)
    row["drugs"] = [{"drugFromSource": name, "drugId": d} for d, name in drugs]
    return row


def test_names_match_whole_phrases_only():
    assert gn.names_match("metformin", "metformin 500 mg")
    assert gn.names_match("semaglutide injection", "semaglutide")
    assert not gn.names_match("met", "metformin")
    assert not gn.names_match("", "metformin")
    assert gn.normalise_name("Drug-A (50 mg)") == "drug a 50 mg"


def test_support_through_a_comparator_only_is_not_investigational_support():
    assoc = [
        ("T2", "E1", series(y2009=0.0, y2010=0.7))
    ]  # evidence only for the comparator's target
    rep = named_report("NCT1", [("D1", "Alphadrug"), ("D2", "Betadrug")], ["E1"])
    experimental = {"NCT1": ["alphadrug"]}
    table = gn.build_table(
        trials(("NCT1", 2015)), reports(rep), DRUG_TARGETS, associations(*assoc), MOLECULES,
        experimental,
    )  # fmt: skip
    row = table.to_dicts()[0]
    assert row["supported_today"] is True and row["supported_at_start"] is True
    assert row["investigational_mapped"] is True
    assert row["supported_today_investigational"] is False
    assert row["supported_at_start_investigational"] is False
    assert row["lead_drug"] == "D1"
    swapped = gn.build_table(
        trials(("NCT1", 2015)), reports(rep), DRUG_TARGETS, associations(*assoc), MOLECULES,
        {"NCT1": ["betadrug"]},
    ).to_dicts()[0]  # fmt: skip
    assert swapped["supported_today_investigational"] is True
    assert swapped["lead_drug"] == "D2"


def test_with_no_experimental_names_nothing_is_investigational_and_the_first_drug_leads():
    rep = named_report("NCT1", [("D3", "x"), ("D1", "y")], ["E1"])
    row = gn.build_table(
        trials(("NCT1", 2015)), reports(rep), DRUG_TARGETS, associations(), MOLECULES
    ).to_dicts()[0]
    assert row["investigational_mapped"] is False and row["lead_drug"] == "D3"


def test_the_experimental_names_come_from_the_experimental_arm_only():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "nct1"},
            "armsInterventionsModule": {
                "armGroups": [
                    {"label": "A", "type": "EXPERIMENTAL"},
                    {"label": "B", "type": "ACTIVE_COMPARATOR"},
                ],
                "interventions": [
                    {"name": "Alphadrug 5 mg", "type": "DRUG", "armGroupLabels": ["A"],
                     "otherNames": ["AB-123"]},
                    {"name": "Betadrug", "type": "DRUG", "armGroupLabels": ["B"]},
                ],
            },
        }
    }  # fmt: skip
    assert gn.experimental_names([study], {"NCT1"}) == {"NCT1": ["ab 123", "alphadrug 5 mg"]}
    assert gn.experimental_names([study], {"NCT2"}) == {}


def test_earlier_trials_of_the_lead_drug_count_strictly_earlier_starts():
    def row(nct, drug, start):
        r = report(nct, [drug], ["E1"])
        r["trialStartDate"] = start
        return r

    rep = reports(
        row("NCT1", "D1", date(2015, 1, 1)), row("NCT2", "D1", date(2010, 1, 1)),
        row("NCT3", "D1", date(2015, 1, 1)), row("NCT4", "D2", date(2001, 1, 1)),
        row("NCT5", "D1", None),
    )  # fmt: skip
    leads = pl.DataFrame({"nct_id": ["NCT1", "NCT2", "NCT5"], "lead_drug": ["D1", "D1", "D1"]})
    got = dict(gn.earlier_trial_counts(rep, leads).rows())
    assert got == {"NCT1": 1, "NCT2": 0, "NCT5": None}


# --- the development-set summary ----------------------------------------------------------------


def dev_frame(rows):
    """Development trials with the label, design columns and genetic features."""
    base = {
        "split": "dev", "primary_completion_date": date(2018, 1, 1), "phase": "phase3",
        "sponsor_class": "INDUSTRY", "therapeutic_area": "oncology", "start_year": 2014,
        "mapped": True, "few_targets": True, "any_arm": False,
        "supported_today": False, "supported_at_start": False,
        "genetic_score_today": 0.0, "genetic_score_at_start": 0.0,
        "investigational_mapped": True, "supported_today_investigational": False,
        "supported_at_start_investigational": False, "lead_drug": "D0",
        "lead_drug_earlier_trials": 0, "target_ids": ["T0"], "supported_targets_today": [],
        "supported_targets_at_start": [], "top_target_today": None, "top_target_at_start": None,
    }  # fmt: skip
    return pl.DataFrame(
        [{**base, "nct_id": f"NCT{i}", **row} for i, row in enumerate(rows)],
        schema_overrides={"top_target_today": pl.String, "top_target_at_start": pl.String,
                          "supported_targets_today": pl.List(pl.String),
                          "supported_targets_at_start": pl.List(pl.String)},
    )  # fmt: skip


def test_the_summary_refuses_a_table_with_a_trial_from_2021_or_later():
    frame = dev_frame([{}, {"primary_completion_date": date(2021, 1, 1)}])
    with pytest.raises(ValueError, match="2021 or later"):
        gs.report(frame, ("", 0.0, 0))


def test_the_summary_refuses_a_table_labelled_as_a_test_split():
    with pytest.raises(ValueError):
        gs.report(dev_frame([{"split": "test"}]), ("", 0.0, 0))


def test_only_the_development_file_can_be_loaded(tmp_path):
    other = tmp_path / "test_features.parquet"
    dev_frame([{}]).write_parquet(other)
    with pytest.raises(ValueError, match="reads only"):
        gs.load_dev(other)


def test_joining_genetics_fails_loudly_when_a_trial_has_no_row():
    dev = dev_frame([{}, {}]).drop("mapped")
    genetics = pl.DataFrame({"nct_id": ["NCT0"], "split": ["dev"], "mapped": [True]})
    with pytest.raises(ValueError, match="no row"):
        gs.with_genetics(dev, genetics.select("nct_id", "split", "mapped"))


def test_newcombe_difference_matches_a_hand_calculation():
    diff, low, high = gs.newcombe_difference(60, 100, 40, 100)
    assert diff == pytest.approx(0.2)
    assert (low, high) == pytest.approx((0.0614, 0.3281), abs=1e-3)


def test_newcombe_difference_is_nan_when_a_group_is_empty():
    assert all(math.isnan(x) for x in gs.newcombe_difference(0, 0, 3, 10))


def test_mantel_haenszel_with_one_stratum_is_the_plain_odds_ratio():
    odds, low, high = gs.mantel_haenszel([(30, 10, 20, 40)])
    assert odds == pytest.approx(30 * 40 / (10 * 20))
    assert low < odds < high


def test_mantel_haenszel_removes_a_confounder():
    # Within each stratum the odds ratio is 1; pooled crudely it would not be.
    strata = [(80, 20, 40, 10), (10, 40, 20, 80)]
    odds, _, _ = gs.mantel_haenszel(strata)
    assert odds == pytest.approx(1.0)


def test_graded_table_counts_each_band_once_and_skips_unmapped():
    cuts = gs.CUTS["today"]
    low, mid, high = cuts[0] / 2, (cuts[0] + cuts[1]) / 2, cuts[1] + 0.01
    frame = dev_frame([
        {"genetic_score_today": 0.0, "supported_today": False, "any_arm": True},
        {"genetic_score_today": low, "supported_today": True, "any_arm": True},
        {"genetic_score_today": mid, "supported_today": True},
        {"genetic_score_today": high, "supported_today": True},
        {"genetic_score_today": None, "supported_today": None, "mapped": False},
    ])  # fmt: skip
    rows = [line.split("|")[1:3] for line in gs.graded_table(frame, "today").splitlines()[2:]]
    assert [(r[0].strip(), r[1].strip()) for r in rows] == [
        ("none", "1"), ("low", "1"), ("middle", "1"), ("high", "1")]  # fmt: skip


def test_modality_agreement_counts_antibody_and_protein_as_biologic():
    frame = pl.DataFrame({
        "modality": ["biologic", "biologic", "small molecule", "small molecule", "biologic"],
        "drug_type": ["antibody", "protein", "small molecule", "protein", "unknown"],
    })  # fmt: skip
    _, rate, n = gs.modality_agreement(frame)
    assert n == 4 and rate == pytest.approx(0.75)


def test_adjusted_odds_ratio_runs_on_a_small_synthetic_table():
    import numpy as np

    rng = np.random.default_rng(1)
    rows = []
    for _ in range(400):
        supported = bool(rng.random() < 0.3)
        rows.append({
            "supported_today": supported,
            "any_arm": bool(rng.random() < (0.7 if supported else 0.5)),
            "phase": "phase3" if rng.random() < 0.5 else "phase2",
            "sponsor_class": "INDUSTRY" if rng.random() < 0.6 else "OTHER",
            "therapeutic_area": str(rng.choice(["oncology", "neurology"])),
            "start_year": int(rng.integers(2005, 2018)),
        })  # fmt: skip
    odds, low, high, method = gs.adjusted_odds_ratio(dev_frame(rows), "supported_today")
    assert low < odds < high and 1.0 < odds < 4.0
    assert method


def drug_clusters_frame():
    """Twenty lead drugs of six trials each; every drug's trials share one outcome."""
    rows = []
    for drug in range(20):
        supported = drug < 10
        success = (drug % 3 != 0) if supported else (drug % 2 == 0)
        rows += [{"lead_drug": f"D{drug}", "supported_today": supported, "any_arm": success}] * 6
    return dev_frame(rows)


def test_the_cluster_bootstrap_is_deterministic_and_seed_dependent():
    frame = drug_clusters_frame()
    a = gs.cluster_bootstrap_difference(frame, "supported_today", n_resamples=200, seed=1)
    b = gs.cluster_bootstrap_difference(frame, "supported_today", n_resamples=200, seed=1)
    c = gs.cluster_bootstrap_difference(frame, "supported_today", n_resamples=200, seed=2)
    assert a == b and a != c
    point, low, high, k = a
    assert k == 20 and low < point < high and point == pytest.approx(0.1)


def test_the_cluster_interval_is_wider_when_a_drug_repeats_with_the_same_result():
    # Each lead drug's trials all share one outcome, so the 60 trials hold about 20 independent
    # pieces of information; the cluster interval must be wider than Newcombe's.
    frame = drug_clusters_frame()
    _, low, high, _ = gs.cluster_bootstrap_difference(frame, "supported_today", n_resamples=500)
    yes, no = frame.filter(pl.col("supported_today")), frame.filter(~pl.col("supported_today"))
    _, n_low, n_high = gs.newcombe_difference(
        int(yes["any_arm"].sum()), yes.height, int(no["any_arm"].sum()), no.height
    )
    assert high - low > n_high - n_low


def test_mantel_haenszel_strata_include_therapeutic_area():
    frame = dev_frame([{"therapeutic_area": "oncology"}, {"therapeutic_area": None},
                       {"therapeutic_area": "oncology"}])  # fmt: skip
    strata = set(gs.strata_of(frame).to_list())
    assert len(strata) == 2 and all(x.startswith("phase3|ind|late|") for x in strata)


def test_a_full_report_runs_on_a_synthetic_development_table():
    import numpy as np

    rng = np.random.default_rng(3)
    rows = []
    for i in range(300):
        supported = bool(rng.random() < 0.4)
        rows.append({
            "supported_today": supported, "supported_at_start": supported and rng.random() < 0.5,
            "genetic_score_today": float(rng.random()) if supported else 0.0,
            "genetic_score_at_start": float(rng.random()) if supported else 0.0,
            "supported_today_investigational": supported and rng.random() < 0.8,
            "supported_at_start_investigational": supported and rng.random() < 0.4,
            "any_arm": bool(rng.random() < (0.7 if supported else 0.55)),
            "mapped": bool(rng.random() < 0.8) or supported,
            "phase": "phase3" if rng.random() < 0.5 else "phase2",
            "sponsor_class": "INDUSTRY" if rng.random() < 0.6 else "OTHER",
            "therapeutic_area": str(rng.choice(["oncology", "neurology"])),
            "start_year": int(rng.integers(2005, 2018)),
            "lead_drug": f"D{int(rng.integers(0, 40))}",
            "lead_drug_earlier_trials": int(rng.integers(0, 12)),
            "target_ids": ["T1", "T2"], "supported_targets_today": ["T1"] if supported else [],
            "supported_targets_at_start": [], "top_target_today": "T1" if supported else None,
        })  # fmt: skip
    text = gs.report(dev_frame(rows), ("", 0.8, 10), {"T1": "GENE1"})
    for heading in ("pre-specified", "added after the development result was seen",
                    "as curated today", "GENE1 (T1)", "cluster bootstrap"):  # fmt: skip
        assert heading in text
