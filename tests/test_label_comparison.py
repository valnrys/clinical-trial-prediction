import csv
from datetime import date

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise import label_comparison as lc


def development(ids):
    frame = synthetic_trials(len(ids), seed=1)
    return frame.with_columns(pl.Series("nct_id", ids))


IDS = ["NCT01", "NCT02", "NCT03"]


def write_csv(path, header, rows):
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def test_ids_that_did_not_come_from_a_checked_development_table_are_refused(tmp_path):
    path = write_csv(tmp_path / "top.csv", ["nctid", "label"], [["NCT01", "1"]])
    with pytest.raises(TypeError, match="Refusing"):
        lc.read_development_rows(path, "nctid", {"label": int}, frozenset(IDS))


def test_development_ids_cannot_be_built_from_arbitrary_ids():
    with pytest.raises(TypeError, match="from_frame"):
        lc.DevelopmentIds(IDS)
    with pytest.raises(TypeError, match="from_frame"):
        lc.DevelopmentIds()


def test_a_plain_set_of_ids_is_refused_by_the_registry_scan_too():
    with pytest.raises(TypeError, match="Refusing"):
        lc.one_sided_counts([], frozenset(IDS))


def test_finished_ids_come_only_from_a_pre_2021_population():
    late = pl.DataFrame(
        {"nct_id": ["NCT01"], "primary_completion_date": [date(2021, 1, 1)]},
        schema={"nct_id": pl.String, "primary_completion_date": pl.Date},
    )
    with pytest.raises(ValueError, match="2021 or later"):
        lc.FinishedIds.from_population(late)
    with pytest.raises(TypeError, match="from_frame"):
        lc.FinishedIds(["NCT01"])


def test_a_development_table_holding_a_2021_trial_cannot_make_the_id_list():
    frame = synthetic_trials(4, seed=2, split="test", first_day=date(2022, 1, 1))
    with pytest.raises(ValueError, match="2021 or later"):
        lc.DevelopmentIds.from_frame(frame)


def test_the_comparison_report_refuses_a_development_table_with_a_2021_trial():
    frame = synthetic_trials(4, seed=2, split="test", first_day=date(2022, 1, 1))
    empty = pl.DataFrame(schema={"nct_id": pl.String, "top_success": pl.Boolean})
    with pytest.raises(ValueError, match="2021 or later"):
        lc.report(frame, {}, (empty, 0), [])


def test_a_hand_read_trial_that_is_not_a_development_trial_is_refused():
    dev = development(IDS)
    rows = [{"nct_id": "NCT99", "category": "x"}]
    with pytest.raises(ValueError, match="NCT99"):
        lc.check_disagreement_ids(rows, dev)


def test_every_committed_hand_read_id_is_checked_against_the_report_dev_table():
    dev = development(["NCT01", "NCT02", "NCT03"])
    empty = pl.DataFrame(schema={"nct_id": pl.String, "top_success": pl.Boolean})
    with pytest.raises(ValueError, match="not development trials"):
        lc.report(dev, {}, (empty, 0), lc.load_disagreements())


def test_the_main_function_stops_before_reading_a_benchmark_if_the_development_file_is_bad(
    tmp_path, monkeypatch
):
    bad = synthetic_trials(4, seed=2, split="test", first_day=date(2022, 1, 1))
    monkeypatch.setattr(lc, "load_dev", lambda: lc.check_is_development(bad) or bad)
    with pytest.raises(ValueError, match="2021 or later"):
        lc.main(
            ["--cto-dir", str(tmp_path), "--top-file", str(tmp_path / "missing.csv")],
            tmp_path / "out.md",
        )


def test_label_cells_of_trials_outside_the_development_set_are_never_converted(tmp_path):
    seen = []

    def convert(text):
        seen.append(text)
        return int(text)

    path = write_csv(
        tmp_path / "top.csv",
        ["nctid", "label", "criteria"],
        [["NCT01", "1", "x"], ["NCT99", "NOT A NUMBER", "y"], ["nct02", "0", "z"]],
    )
    ids = lc.DevelopmentIds.from_frame(development(IDS))
    rows = lc.read_development_rows(path, "nctid", {"label": convert}, ids)
    assert rows == [{"nct_id": "NCT01", "label": 1}, {"nct_id": "NCT02", "label": 0}]
    assert seen == ["1", "0"]


def test_a_file_missing_a_wanted_column_is_refused(tmp_path):
    path = write_csv(tmp_path / "f.csv", ["nctid", "something"], [["NCT01", "1"]])
    ids = lc.DevelopmentIds.from_frame(development(IDS))
    with pytest.raises(ValueError, match="no column 'label'"):
        lc.read_development_rows(path, "nctid", {"label": int}, ids)


def test_a_very_long_multiline_cell_does_not_stop_the_reader(tmp_path):
    path = write_csv(
        tmp_path / "top.csv", ["nctid", "label", "criteria"], [["NCT01", "1", "line\n" * 100_000]]
    )
    ids = lc.DevelopmentIds.from_frame(development(IDS))
    assert len(lc.read_development_rows(path, "nctid", {"label": int}, ids)) == 1


def test_cto_duplicates_that_agree_count_once_and_probabilities_are_averaged(tmp_path):
    header = ["nct_id", "pred_proba", "pvalues", "other"]
    path = write_csv(
        tmp_path / "cto.csv",
        header,
        [["NCT01", "1.0", "1", "a"], ["NCT01", "1.0", "1", "b"], ["NCT02", "0.2", "0", "c"],
         ["NCT02", "0.4", "0", "d"], ["NCT99", "0.9", "1", "e"]],
    )  # fmt: skip
    read = lc.read_cto(path, lc.DevelopmentIds.from_frame(development(IDS)))
    rows = {r["nct_id"]: r for r in read.frame.iter_rows(named=True)}
    assert set(rows) == {"NCT01", "NCT02"}
    assert read.repeated == 2
    assert read.dropped == 0
    assert rows["NCT01"]["probability"] == 1.0
    assert rows["NCT02"]["probability"] == pytest.approx(0.3)


@pytest.mark.parametrize(("low", "high"), [(0.4, 0.6), (0.6, 0.8)])
def test_cto_copies_on_different_sides_of_a_cut_off_are_dropped_and_counted(tmp_path, low, high):
    path = write_csv(
        tmp_path / "cto.csv",
        ["nct_id", "pred_proba", "pvalues"],
        [["NCT01", str(low), "1"], ["NCT01", str(high), "1"], ["NCT02", "0.9", "1"]],
    )
    read = lc.read_cto(path, lc.DevelopmentIds.from_frame(development(IDS)))
    assert read.frame["nct_id"].to_list() == ["NCT02"]
    assert (read.repeated, read.dropped) == (1, 1)


def test_cto_copies_that_disagree_on_the_p_value_signal_give_unknown(tmp_path):
    path = write_csv(
        tmp_path / "cto.csv",
        ["nct_id", "pred_proba", "pvalues"],
        [["NCT01", "0.8", "1"], ["NCT01", "0.8", "0"]],
    )
    read = lc.read_cto(path, lc.DevelopmentIds.from_frame(development(IDS)))
    assert read.frame["p_value_signal"].to_list() == [-1]


def test_top_trials_whose_copies_disagree_are_dropped_and_counted(tmp_path):
    path = write_csv(
        tmp_path / "top.csv",
        ["nctid", "label"],
        [["NCT01", "1"], ["NCT01", "1"], ["NCT02", "1"], ["NCT02", "0"], ["NCT03", "0"]],
    )
    frame, dropped = lc.read_top(path, lc.DevelopmentIds.from_frame(development(IDS)))
    assert dict(frame.iter_rows()) == {"NCT01": True, "NCT03": False}
    assert dropped == 1


def test_kappa_matches_a_hand_calculation():
    # 4 both succeed, 1 only ours, 2 only theirs, 3 neither. Agreement 0.7. Chance 0.5 * 0.6 + 0.5 * 0.4 = 0.5.
    ours = np.array([1] * 4 + [1] + [0] * 2 + [0] * 3, dtype=bool)
    theirs = np.array([1] * 4 + [0] + [1] * 2 + [0] * 3, dtype=bool)
    assert lc.confusion(ours, theirs) == (4, 1, 2, 3)
    assert lc.kappa(ours, theirs) == pytest.approx(0.4)


def test_kappa_is_one_for_identical_labels_and_zero_for_independent_ones():
    a = np.array([True, True, False, False])
    assert lc.kappa(a, a) == pytest.approx(1.0)
    independent = np.array([True, False, True, False])
    assert lc.kappa(a, independent) == pytest.approx(0.0)


def test_the_kappa_interval_is_reproducible_and_holds_the_estimate():
    rng = np.random.default_rng(0)
    ours = rng.random(300) < 0.5
    theirs = ours ^ (rng.random(300) < 0.2)
    first = lc.bootstrap_interval(ours, theirs, lc.kappa)
    assert first == lc.bootstrap_interval(ours, theirs, lc.kappa)
    assert first[0] < lc.kappa(ours, theirs) < first[1]


def test_the_committed_hand_read_of_disagreements_is_complete():
    rows = lc.load_disagreements()
    assert len(rows) == len({r["nct_id"] for r in rows}) == 29
    assert set(rows[0]) == {"nct_id", "direction", "category", "note", "source"}
    counts = {}
    for row in rows:
        counts[row["category"]] = counts.get(row["category"], 0) + 1
    assert counts == {
        lc.CATEGORY_STOPPED: 17,
        lc.CATEGORY_WRONG_WAY: 5,
        lc.CATEGORY_ONE_SIDED: 1,
        lc.CATEGORY_THRESHOLD: 2,
        lc.CATEGORY_UNKNOWN: 4,
    }
    assert {r["source"] for r in rows} <= lc.SOURCES


def test_the_checked_rows_cite_registry_numbers_and_the_recalled_ones_say_so():
    by_id = {r["nct_id"]: r for r in lc.load_disagreements()}
    for nct in ("NCT01151137", "NCT02556203", "NCT00768300", "NCT03061812", "NCT00594568"):
        assert by_id[nct]["category"] == lc.CATEGORY_WRONG_WAY
        assert by_id[nct]["source"] == "registry record"
    assert by_id["NCT01191801"]["category"] == lc.CATEGORY_ONE_SIDED
    assert by_id["NCT00562965"]["category"] == lc.CATEGORY_STOPPED
    for nct in ("NCT01924533", "NCT00490139"):
        assert by_id[nct]["source"] == "model recall, unverified"


def analysis(p="0.03", **extra):
    return {"pValue": p, "groupIds": ["G1", "G2"]} | extra


def trial_with(analyses):
    return {
        "protocolSection": {"armsInterventionsModule": {"armGroups": []}},
        "resultsSection": {
            "outcomeMeasuresModule": {
                "outcomeMeasures": [
                    {
                        "type": "PRIMARY",
                        "groups": [
                            {"id": "G1", "title": "Drug X"},
                            {"id": "G2", "title": "Placebo"},
                        ],
                        "analyses": analyses,
                    }
                ]
            }
        },
    }


@pytest.mark.parametrize(
    ("extra", "expected"),
    [
        ({"pValueComment": "Unstratified one sided P-Value"}, True),
        ({"statisticalComment": "One-sided test at 2.5%"}, True),
        ({"pValueComment": "1-sided"}, True),
        ({"pValueComment": "two sided"}, False),
        ({"statisticalComment": "Someone sided with the sponsor"}, False),
        ({}, False),
    ],
)
def test_one_sided_comments_are_found(extra, expected):
    assert lc.mentions_one_sided(analysis(**extra)) is expected


def test_the_deciding_analyses_are_each_arms_first_comparison_on_the_first_primary_outcome():
    study = trial_with([analysis("0.03", pValueComment="one sided"), analysis("0.9")])
    chosen = lc.deciding_analyses(study)
    assert [a["pValue"] for a in chosen] == ["0.03"]


def test_one_sided_counts_look_only_at_development_trials():
    def study(nct, **extra):
        record = trial_with([analysis(**extra)])
        record["protocolSection"]["identificationModule"] = {"nctId": nct}
        return record

    studies = [
        study("NCT01", pValueComment="one sided"),
        study("NCT02"),
        study("NCT09", pValueComment="one sided"),
    ]
    ids = lc.DevelopmentIds.from_frame(development(["NCT01", "NCT02"]))
    assert lc.one_sided_counts(studies, ids) == (2, 1)


def population_of(rows):
    return pl.DataFrame(
        [
            {"nct_id": nct, "status": status, "primary_completion_date": date(2019, 1, 1)}
            for nct, status in rows
        ],
        schema={"nct_id": pl.String, "status": pl.String, "primary_completion_date": pl.Date},
    )


def test_registry_facts_count_statuses_and_how_many_of_each_are_in_our_set():
    population = population_of(
        [
            ("NCT01", "COMPLETED"),
            ("NCT02", "COMPLETED"),
            ("NCT03", "TERMINATED"),
            ("NCT04", "WITHDRAWN"),
        ]
    )
    ids = lc.DevelopmentIds.from_frame(development(["NCT01", "NCT03"]))
    facts = lc.registry_facts(population, [], ids)
    assert (facts.finished, facts.terminated, facts.withdrawn) == (4, 1, 1)
    assert facts.inclusion == {"COMPLETED": 0.5, "TERMINATED": 1.0, "WITHDRAWN": 0.0}


def test_the_wider_cto_shares_are_read_over_the_whole_finished_population(tmp_path):
    population = population_of(
        [
            ("NCT01", "COMPLETED"),
            ("NCT02", "TERMINATED"),
            ("NCT03", "COMPLETED"),
            ("NCT04", "WITHDRAWN"),
        ]
    )
    path = write_csv(
        tmp_path / "cto.csv",
        ["nct_id", "pred_proba", "pvalues"],
        [
            ["NCT01", "0.95", "1"],
            ["NCT02", "0.02", "0"],
            ["NCT03", "0.6", "0"],
            ["NCT09", "0.9", "1"],
        ],
    )
    wide = lc.wider_cto(path, population)
    assert wide.n == 3
    assert wide.success_half == pytest.approx(2 / 3)
    assert wide.stopped == pytest.approx(1 / 3)


def test_a_report_on_small_synthetic_files_has_every_section_and_no_always_success_framing(
    tmp_path,
):
    dev = development(["NCT00428597", "NCT00748189", "NCT03", "NCT04", "NCT05", "NCT06"])
    rows = [r for r in lc.load_disagreements() if r["nct_id"] in {"NCT00428597", "NCT00748189"}]
    cto = pl.DataFrame(
        {
            "nct_id": ["NCT00428597", "NCT00748189", "NCT03", "NCT04"],
            "probability": [0.95, 0.05, 0.7, 0.6],
            "p_value_signal": [1, 0, 1, -1],
        }
    )
    top = pl.DataFrame({"nct_id": ["NCT00428597", "NCT05"], "top_success": [True, False]})
    facts = lc.RegistryFacts(
        1000,
        150,
        40,
        100,
        7,
        {"phase 2 file": lc.WiderCto(500, 0.78, 0.22)},
        {"COMPLETED": 0.12, "TERMINATED": 0.06, "WITHDRAWN": 0.0},
    )
    text = lc.report(dev, {"phase 2 file": lc.CtoRead(cto, 0, 0)}, (top, 0), rows, facts)
    for heading in ("## 1.", "## 2.", "## 3.", "## 4.", "## 5.", "## 6."):
        assert heading in text
    assert "NCT00428597" in text
    assert "Outcome labels are provided by IQVIA" in text
    assert "always success" not in text and "always predict" not in text
    assert "7 (7.0%)" in text  # the one-sided count, with its denominator of 100


def test_the_cluster_table_covers_every_trial_once():
    frame = pl.DataFrame(
        {
            "probability": [0.02, 0.6, 0.65, 0.95, 0.99],
            "overall_status": ["TERMINATED", "COMPLETED", "COMPLETED", "COMPLETED", "COMPLETED"],
            "p_value_signal": [0, 0, 0, 1, 1],
            lc.OURS: [False, False, False, True, True],
        }
    )
    text = lc.cluster_table(frame)
    assert "| below 0.1 | 1 | 100.0% | 0.0% | 0.0% |" in text
    assert "| 0.55 to 0.75 | 2 | 0.0% | 0.0% | 0.0% |" in text
    assert "| 0.9 and above | 2 | 0.0% | 100.0% | 100.0% |" in text


def test_the_threshold_views_change_what_counts_as_cto_success():
    frame = pl.DataFrame(
        {
            "probability": [0.02, 0.6, 0.95],
            lc.OURS: [False, False, True],
        }
    )
    views = lc.threshold_frames(frame)
    at_half, at_seven, confident = views.values()
    assert at_half["theirs"].to_list() == [False, True, True]
    assert at_seven["theirs"].to_list() == [False, False, True]
    assert confident.height == 2
