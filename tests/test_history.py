from datetime import date

import polars as pl
import pytest

from trial_benchmark import history as hs
from trial_benchmark.dating import DATED

T_START = date(2015, 6, 10)


def study(nct, start, *, results=None, phase2=False, drugs=("name:drug",), conds=("mesh:c",),
          sponsor="acme", combination=False):  # fmt: skip
    """A registry row. A date may be a day, or a (first, last) pair for a month-only date."""
    first, last = start if isinstance(start, tuple) else (start, start)
    results_end = results[1] if isinstance(results, tuple) else results
    return {
        "nct_id": nct, "start": first, "start_end": last, "results_end": results_end,
        "sponsor": sponsor, "is_phase2": phase2, "combination": combination,
        "drug_keys": list(drugs), "condition_keys": list(conds),
    }  # fmt: skip


def registry(*rows):
    return pl.DataFrame(list(rows), schema={
        "nct_id": pl.String, "start": pl.Date, "start_end": pl.Date, "results_end": pl.Date,
        "sponsor": pl.String,
        "is_phase2": pl.Boolean, "combination": pl.Boolean,
        "drug_keys": pl.List(pl.String), "condition_keys": pl.List(pl.String),
    })  # fmt: skip


def labels(**by_id):
    """Development labels: id -> any_arm. An id ending in _test gets split "test"."""
    rows = [{"nct_id": k.removesuffix("_test"), "split": "test" if k.endswith("_test") else "dev",
             "any_arm": v} for k, v in by_id.items()]  # fmt: skip
    return pl.DataFrame(
        rows, schema={"nct_id": pl.String, "split": pl.String, "any_arm": pl.Boolean}
    )


def run(reg, lab, ids, **kwargs):
    trials = pl.DataFrame({"nct_id": ids, "split": ["dev"] * len(ids)})
    return hs.build_features(reg, lab, trials, **kwargs).sort("nct_id")


def row(table, nct):
    return table.filter(pl.col("nct_id") == nct).row(0, named=True)


def one_earlier_phase2(results, start=date(2014, 1, 1), **kwargs):
    reg = registry(
        study("T", T_START),
        study("U", start, results=results, phase2=True, **kwargs),
    )
    return row(run(reg, labels(U=True), ["T"]), "T")


# ---- timing: the label is known only once results are posted before the start


@pytest.mark.parametrize(
    ("posted", "known"),
    [(date(2015, 6, 9), True), (T_START, False), (date(2015, 6, 11), False), (None, False)],
    ids=["day before", "same day", "day after", "never posted"],
)
def test_label_is_known_only_if_results_were_posted_before_the_start(posted, known):
    got = one_earlier_phase2(posted)
    assert got["prior_phase2_labelled"] is known
    assert got["prior_phase2_succeeded"] is (True if known else None)


@pytest.mark.parametrize(
    ("u_start", "counted"),
    [(date(2015, 6, 9), True), (T_START, False), (date(2015, 6, 11), False)],
    ids=["day before", "same day", "day after"],
)
def test_an_earlier_trial_must_start_strictly_before(u_start, counted):
    got = one_earlier_phase2(date(2015, 6, 9), start=u_start)
    assert got["n_earlier_trials_same_drug"] == int(counted)
    assert got["n_earlier_trials_same_programme"] == int(counted)
    assert got["prior_phase2_labelled"] is counted


def test_date_range_of_a_month_is_the_whole_month():
    assert hs.date_range("2015-03") == (date(2015, 3, 1), date(2015, 3, 31))
    assert hs.date_range("2015-02-10") == (date(2015, 2, 10), date(2015, 2, 10))
    assert hs.date_range("") == (None, None)


MARCH = hs.date_range("2015-03")
DAY_15 = hs.date_range("2015-03-15")


@pytest.mark.parametrize(
    ("t_start", "u_start", "counted"),
    [
        (DAY_15, MARCH, False),  # U may have started on 20 March: not clearly before
        (MARCH, DAY_15, False),  # T may have started on 1 March
        (MARCH, MARCH, False),  # same month on both sides
        (hs.date_range("2015-04"), MARCH, True),  # every day of March is before every day of April
        (hs.date_range("2015-04-01"), MARCH, True),  # ... and before 1 April
        (DAY_15, hs.date_range("2015-02"), True),
        (DAY_15, hs.date_range("2015-03-14"), True),  # both full days: plain strict comparison
    ],
    ids=["u month only", "t month only", "both month", "months apart", "month then day",
         "earlier month", "both days"],
)  # fmt: skip
def test_started_before_holds_only_for_every_day_of_a_month_only_date(t_start, u_start, counted):
    reg = registry(study("T", t_start), study("U", u_start))
    assert row(run(reg, labels(), ["T"]), "T")["n_earlier_trials_same_drug"] == int(counted)


@pytest.mark.parametrize(
    ("posted", "known"),
    [
        (hs.date_range("2015-05"), True),  # the whole of May is before 10 June
        (hs.date_range("2015-06"), False),  # June may have been after the 10th
        (hs.date_range("2015-06-09"), True),
    ],
    ids=["month before", "month of the start", "day before"],
)
def test_results_posted_before_holds_only_for_every_day_of_a_month_only_date(posted, known):
    got = one_earlier_phase2(posted)
    assert got["prior_phase2_labelled"] is known


def test_results_posted_must_precede_every_day_of_a_month_only_start():
    reg = registry(
        study("T", hs.date_range("2015-06")),
        study("U", date(2014, 1, 1), results=hs.date_range("2015-06-02"), phase2=True),
    )
    assert row(run(reg, labels(U=True), ["T"]), "T")["prior_phase2_labelled"] is False


# ---- what counts as an earlier Phase 2 of the same programme


def test_a_phase2_3_trial_is_not_an_earlier_phase2():
    reg = registry(
        study("T", T_START),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=False),
    )
    got = row(run(reg, labels(U=True), ["T"]), "T")
    assert got["prior_phase2_labelled"] is False
    assert got["n_earlier_trials_same_drug"] == 1  # still an earlier trial of the drug


def test_a_study_never_matches_itself():
    reg = registry(study("T", T_START, results=date(2010, 1, 1), phase2=True))
    got = row(run(reg, labels(T=True), ["T"]), "T")
    assert got["n_earlier_trials_same_drug"] == 0
    assert got["prior_phase2_labelled"] is False


def test_the_programme_needs_a_shared_condition_but_the_drug_count_does_not():
    reg = registry(
        study("T", T_START, conds=("mesh:a",)),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, conds=("mesh:b",)),
    )
    got = row(run(reg, labels(U=True), ["T"]), "T")
    assert got["n_earlier_trials_same_drug"] == 1
    assert got["n_earlier_trials_same_programme"] == 0
    assert got["prior_phase2_labelled"] is False


def test_a_different_drug_does_not_match():
    reg = registry(
        study("T", T_START, drugs=("name:a",)),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, drugs=("name:b",)),
    )
    assert row(run(reg, labels(U=True), ["T"]), "T")["n_earlier_trials_same_drug"] == 0


def test_a_common_drug_name_is_not_a_key():
    saline = [study(f"X{i}", date(2014, 1, 1), drugs=("name:saline",)) for i in range(3)]
    reg = registry(
        study("T", T_START, drugs=("name:saline", "name:rare")),
        *saline,
        study("R", date(2014, 1, 1), drugs=("name:rare",)),
    )
    assert row(run(reg, labels(), ["T"], max_earlier=3), "T")["n_earlier_trials_same_drug"] == 4
    # three earlier trials share "saline", more than the limit of 2: only "rare" is left
    assert row(run(reg, labels(), ["T"], max_earlier=2), "T")["n_earlier_trials_same_drug"] == 1


def test_the_common_name_limit_counts_only_trials_that_started_before_the_trial():
    """A drug rare before T but common after: T must still see it. Counting the whole
    registry would drop it for T, which uses the future."""
    early = study("EARLY", date(2014, 1, 1), phase2=True, results=date(2014, 6, 1))
    later = [study(f"L{i}", date(2016, 1, 1) + (date(2016, 1, 2) - date(2016, 1, 1)) * i)
             for i in range(5)]  # fmt: skip
    reg = registry(study("T", T_START), early, *later)
    got = row(run(reg, labels(EARLY=True), ["T"], max_earlier=2), "T")
    assert got["n_earlier_trials_same_drug"] == 1
    assert got["prior_phase2_labelled"] is True
    # and a trial that comes after all of them does see the drug as common
    last = study("LAST", date(2017, 1, 1))
    got = row(
        run(registry(early, *later, last), labels(EARLY=True), ["LAST"], max_earlier=2), "LAST"
    )
    assert got["n_earlier_trials_same_drug"] == 0


def test_a_combination_trial_is_not_matched_to_its_part_for_labels_but_is_counted():
    reg = registry(
        study("T", T_START),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, combination=True),
    )
    got = row(run(reg, labels(U=True), ["T"]), "T")
    assert got["prior_phase2_labelled"] is False
    assert got["n_earlier_trials_same_programme"] == 1


# ---- the verdict


def test_succeeded_is_true_if_any_earlier_phase2_succeeded():
    def verdict(*outcomes):
        us = [
            study(f"U{i}", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True)
            for i in range(len(outcomes))
        ]
        lab = labels(**{f"U{i}": o for i, o in enumerate(outcomes)})
        return row(run(registry(study("T", T_START), *us), lab, ["T"]), "T")

    mixed = verdict(True, False)
    assert mixed["prior_phase2_succeeded"] is True
    failed = verdict(False, False)
    assert failed["prior_phase2_succeeded"] is False


def test_same_sponsor_is_the_stricter_join():
    reg = registry(
        study("T", T_START, sponsor="acme"),
        study("U1", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, sponsor="other"),
        study("U2", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, sponsor="acme"),
    )
    got = row(run(reg, labels(U1=True, U2=False), ["T"]), "T")
    assert got["prior_phase2_succeeded"] is True  # U1 succeeded
    assert got["prior_phase2_same_sponsor"] is True
    assert got["prior_phase2_same_sponsor_succeeded"] is False  # only U2 counts, and it failed


def test_a_missing_sponsor_never_matches():
    reg = registry(
        study("T", T_START, sponsor=None),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, sponsor=None),
    )
    got = row(run(reg, labels(U=True), ["T"]), "T")
    assert got["prior_phase2_labelled"] is True
    assert got["prior_phase2_same_sponsor"] is False


# ---- labels come from development data only


def test_a_test_split_trials_label_is_never_used():
    reg = registry(
        study("T", T_START),
        study("U_test", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True),
    ).with_columns(any_arm=pl.lit(True))  # a tempting column that must be ignored
    trials = pl.DataFrame({"nct_id": ["T"], "split": ["late"]})
    lab = labels(U_test=True)  # tagged "test" in the labels frame
    got = row(hs.build_features(reg, lab, trials), "T")
    assert got["prior_phase2_labelled"] is False
    assert got["prior_phase2_succeeded"] is None
    assert got["n_earlier_trials_same_drug"] == 1  # the trial still counts, label-free


def test_sealed_history_lists_trials_that_would_need_a_test_label():
    reg = registry(
        study("T", T_START),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True),
        study("V", date(2014, 1, 1), results=date(2016, 1, 1), phase2=True),  # posted too late
    )
    trials = pl.DataFrame({"nct_id": ["T", "U", "V"], "split": ["late", "test", "test"]})
    assert hs.sealed_history_trials(reg, trials) == ["T"]


# ---- nulls


def test_no_start_date_gives_null_features_and_a_missing_earlier_start_is_skipped():
    reg = registry(
        study("NOSTART", None),
        study("T", T_START),
        study("U", None, results=date(2014, 6, 1), phase2=True),
    )
    table = run(reg, labels(U=True), ["NOSTART", "T"])
    assert all(v is None for k, v in row(table, "NOSTART").items() if k not in ("nct_id", "split"))
    got = row(table, "T")
    assert got["n_earlier_trials_same_drug"] == 0
    assert got["prior_phase2_labelled"] is False
    assert got["prior_phase2_succeeded"] is None


def test_every_trial_gets_a_row_and_counts_are_zero_not_null_without_history():
    got = row(run(registry(study("T", T_START)), labels(), ["T"]), "T")
    assert got["n_earlier_trials_same_drug"] == 0
    assert got["prior_phase2_labelled"] is False


# ---- names and the contract


def test_drug_names_drop_doses_salts_and_brackets():
    assert hs.drug_name("Ruxolitinib 1.5% Cream QD") == "ruxolitinib"
    assert hs.drug_name("Mirabegron 50 mg tablet") == "mirabegron"
    assert hs.drug_name("Linaclotide Acetate") == "linaclotide"
    assert hs.drug_name("INT-747 (OCA)") == "int 747"
    assert hs.drug_name("NA") == ""


def test_device_words_do_not_make_a_drug_key():
    assert hs.drug_name("Ellipta device") == ""
    assert hs.drug_name("Inhaler") == ""
    assert hs.drug_name("Fluticasone furoate pen") == "fluticasone furoate"


@pytest.mark.parametrize(
    "name",
    ["Empagliflozin + Linagliptin", "LUM/IVA", "Drug A and Drug B", "Drug A with Drug B",
     "Fixed dose combination of A", "Drug A plus Drug B"],
)  # fmt: skip
def test_names_that_join_drugs_are_combinations(name):
    assert hs.names_a_combination(name)


@pytest.mark.parametrize(
    "name",
    ["Dulaglutide", None, "Drug X 10 mg/mL", "Drug X 5 mg/kg", "Drug X 1.5 mg/kg/day",
     "Drug X taken with food", "Drug X 10 mg + placebo"],
)  # fmt: skip
def test_doses_and_food_do_not_make_a_combination(name):
    assert not hs.names_a_combination(name)


def intervention(name, *arms):
    return {"name": name, "armGroupLabels": list(arms)}


def test_two_distinct_drugs_in_one_experimental_arm_are_a_combination():
    both = [intervention("Drug A", "Arm 1"), intervention("Drug B", "Arm 1")]
    assert hs.is_combination(both, ["Arm 1"])


def test_different_drugs_in_different_arms_or_doses_of_one_drug_are_not():
    two_arms = [intervention("Drug A", "Arm 1"), intervention("Drug B", "Arm 2")]
    assert not hs.is_combination(two_arms, ["Arm 1", "Arm 2"])
    doses = [intervention("Drug A 10 mg tablet", "Arm 1"), intervention("Drug A 20 mg", "Arm 1")]
    assert not hs.is_combination(doses, ["Arm 1"])


def test_the_sealed_list_ignores_combinations_like_the_labelled_pairs():
    reg = registry(
        study("T", T_START),
        study("U", date(2014, 1, 1), results=date(2014, 6, 1), phase2=True, combination=True),
    )
    trials = pl.DataFrame({"nct_id": ["T", "U"], "split": ["late", "test"]})
    assert hs.sealed_history_trials(reg, trials) == []


def test_features_contract():
    assert hs.TABLE == "history"
    assert set(hs.FEATURES.values()) == {DATED}
    table = run(registry(study("T", T_START)), labels(), ["T"])
    assert set(hs.FEATURES) <= set(table.columns)
