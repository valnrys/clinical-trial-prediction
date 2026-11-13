import math
from dataclasses import replace
from datetime import date
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials
from test_evaluate import lock, write_synthetic_project
from test_failures_summary import study

from elman_exercise import baselines as bl
from elman_exercise import evaluate as ev
from elman_exercise import heldout
from elman_exercise import metrics as mt
from elman_exercise import scorecard as sc
from elman_exercise.scorecard import (
    INSIDE,
    NO_RANGE,
    NOT_COMPUTED,
    OUTSIDE,
    OVERLAPS,
    Range,
    verdict,
)

# ---------------------------------------------------------------- the verdict rule

PREDICTED = Range(5, 9)


@pytest.mark.parametrize("value", [5, 7, 9])
def test_a_value_inside_the_range_including_both_ends_is_inside(value):
    assert verdict(value, 0, 20, PREDICTED) == INSIDE


@pytest.mark.parametrize("value, low, high", [(4.99, 3, 5.5), (9.01, 8.5, 12)])
def test_a_value_just_outside_with_an_interval_reaching_in_overlaps(value, low, high):
    assert verdict(value, low, high, PREDICTED) == OVERLAPS


def test_an_interval_that_only_touches_the_range_end_overlaps():
    assert verdict(3, 1, 5, PREDICTED) == OVERLAPS
    assert verdict(11, 9, 13, PREDICTED) == OVERLAPS


@pytest.mark.parametrize("value, low, high", [(3, 1, 4.99), (11, 9.01, 13)])
def test_an_interval_that_stops_short_of_the_range_is_outside(value, low, high):
    assert verdict(value, low, high, PREDICTED) == OUTSIDE


def test_an_interval_that_spans_the_whole_range_overlaps():
    assert verdict(2, 1, 12, PREDICTED) == OVERLAPS


def test_an_open_ended_range_has_no_far_side():
    under = Range(None, 0.01)
    assert verdict(-0.5, -0.6, -0.4, under) == INSIDE
    assert verdict(0.01, 0, 0.02, under) == INSIDE
    assert verdict(0.02, 0.005, 0.03, under) == OVERLAPS
    assert verdict(0.02, 0.011, 0.03, under) == OUTSIDE


def test_no_range_gives_no_verdict_and_no_value_is_not_computed():
    assert verdict(3, 1, 5, None) == NO_RANGE
    assert verdict(math.nan, math.nan, math.nan, PREDICTED) == NOT_COMPUTED


def test_a_missing_interval_cannot_rescue_a_value_outside_the_range():
    assert verdict(2, math.nan, math.nan, PREDICTED) == OUTSIDE


def test_a_strict_upper_end_is_not_inside_and_is_printed_as_below():
    under = Range(None, 0.01, high_included=False)
    assert verdict(0.0099, 0, 0.02, under) == INSIDE
    assert verdict(0.01, 0, 0.02, under) == OVERLAPS
    assert verdict(0.01, 0.01, 0.02, under) == OUTSIDE
    assert sc.show_range(sc.AUROC_DIFFERENCE, under) == "below +0.010"
    assert sc.show_range(sc.AUROC_DIFFERENCE, Range(None, 0.01)) == "up to +0.010"


def test_ranges_are_shown_in_the_units_of_their_row():
    assert sc.show_range(sc.POINTS, Range(-10, -3)) == "-10.0 to -3.0 points"
    assert sc.show_range(sc.PERCENT, Range(45, 50)) == "45.0% to 50.0%"
    assert sc.show_range(sc.RATIO, None) == "no range stated"


# ---------------------------------------------------------------- differences and their signs


def trials(successes: int, n: int, **columns) -> pl.DataFrame:
    """n trials, the first `successes` of them successful, with extra constant columns."""
    frame = pl.DataFrame({bl_label(): [i < successes for i in range(n)]})
    return frame.with_columns(**{k: pl.lit(v) for k, v in columns.items()})


def bl_label() -> str:
    return "any_arm"


def test_a_contrast_is_the_first_group_minus_the_second_in_points():
    forward = sc.contrast(trials(75, 100), trials(25, 100))
    backward = sc.contrast(trials(25, 100), trials(75, 100))
    assert forward.difference == pytest.approx(50)
    assert backward.difference == pytest.approx(-50)
    assert (forward.n_first, forward.n_second) == (100, 100)


def test_a_contrast_interval_is_newcombes_in_points_and_holds_the_estimate():
    c = sc.contrast(trials(30, 50), trials(20, 50))
    _, low, high = sc.newcombe_difference(30, 50, 20, 50)
    assert (c.low, c.high) == pytest.approx((low * 100, high * 100))
    assert c.low < c.difference < c.high


def test_a_contrast_refuses_trials_with_no_label():
    unlabelled = pl.DataFrame({"any_arm": [True, None]})
    with pytest.raises(ValueError):
        sc.contrast(unlabelled, trials(1, 4))
    with pytest.raises(ValueError):
        sc.contrast(trials(1, 4), unlabelled)


def test_an_empty_group_gives_no_contrast():
    c = sc.contrast(trials(1, 4), trials(0, 0))
    assert math.isnan(c.difference)


def frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def many(count: int, **columns) -> list[dict]:
    return [dict(columns) for _ in range(count)]


def test_trial_size_compares_under_100_with_300_or_more_and_leaves_the_middle_out():
    rows = (
        many(10, any_arm=True, enrollment=99, control_kind="placebo")
        + many(10, any_arm=False, enrollment=99, control_kind="placebo")
        + many(10, any_arm=True, enrollment=100, control_kind="placebo")  # 100 is in neither
        + many(10, any_arm=True, enrollment=299, control_kind="placebo")  # nor is 299
        + many(15, any_arm=True, enrollment=300, control_kind="placebo")
        + many(5, any_arm=False, enrollment=300, control_kind="placebo")
        + many(5, any_arm=False, enrollment=None, control_kind="placebo")
    )
    _, size = sc.prediction_5(frame(rows))
    assert size.value == pytest.approx(50 - 75)
    assert size.note.startswith("20 under 100 against 20 300 or more")


def test_the_control_comparison_is_active_minus_placebo():
    rows = many(10, any_arm=True, control_kind="active") + many(
        10, any_arm=False, control_kind="active"
    )
    rows += many(15, any_arm=True, control_kind="placebo") + many(
        5, any_arm=False, control_kind="placebo"
    )
    rows += many(40, any_arm=True, control_kind="no treatment")
    rows = [r | {"enrollment": 150} for r in rows]
    control, _ = sc.prediction_5(frame(rows))
    assert control.value == pytest.approx(50 - 75)
    assert control.predicted == Range(-10, -3)


def test_the_modality_gap_is_biologic_minus_small_molecule_and_says_if_zero_is_excluded():
    rows = many(70, any_arm=True, modality="biologic") + many(
        30, any_arm=False, modality="biologic"
    )
    rows += many(50, any_arm=True, modality="small molecule") + many(
        50, any_arm=False, modality="small molecule"
    )
    rows += many(50, any_arm=False, modality="vaccine")
    row = sc.prediction_4_gap(frame(rows))
    assert row.value == pytest.approx(20)
    assert row.predicted is None
    assert row.verdict == NO_RANGE
    assert "the interval excludes zero" in row.note
    few = many(3, any_arm=True, modality="biologic") + many(2, any_arm=False, modality="biologic")
    few += many(2, any_arm=True, modality="small molecule")
    few += many(3, any_arm=False, modality="small molecule")
    narrow = sc.prediction_4_gap(frame(few))
    assert "the interval does not exclude zero" in narrow.note


def test_trials_with_no_label_are_left_out_not_counted_as_failures():
    rows = many(10, any_arm=True, modality="biologic", control_kind="active", enrollment=50)
    rows += many(10, any_arm=None, modality="biologic", control_kind="active", enrollment=50)
    rows += many(
        10, any_arm=False, modality="small molecule", control_kind="placebo", enrollment=400
    )
    rows += many(
        10, any_arm=None, modality="small molecule", control_kind="placebo", enrollment=400
    )
    table = frame(rows)
    assert sc.prediction_4_gap(table).value == pytest.approx(100)
    control, size = sc.prediction_5(table)
    assert control.value == pytest.approx(100)
    assert control.note.startswith("10 active against 10 placebo")
    assert size.note.startswith("10 under 100 against 10 300 or more")


def genetic_rows() -> pl.DataFrame:
    def part(supported, successes, n, earlier):
        rows = many(n, supported_today=supported, lead_drug_earlier_trials=earlier)
        return [r | {"any_arm": i < successes} for i, r in enumerate(rows)]

    rows = part(True, 8, 10, 5) + part(False, 4, 10, 5)  # 5 earlier trials: few
    rows += part(True, 9, 10, 20) + part(False, 3, 10, 20)  # 20 earlier trials: established
    rows += part(True, 0, 10, 6)  # 6 earlier trials: established, supported only
    rows += part(None, 10, 10, 1)  # not mapped: never counted
    return frame(rows)


def test_genetic_support_is_supported_minus_not_among_mapped_trials():
    everyone, _ = sc.prediction_6(genetic_rows())
    # supported: 17 of 30; not supported: 7 of 20
    assert everyone.value == pytest.approx(100 * (17 / 30 - 7 / 20))
    assert everyone.note.startswith("30 supported against 20 not supported")
    assert everyone.predicted == Range(5, 13)


def test_few_earlier_trials_is_compared_with_more_than_five_as_genetics_summary_does():
    _, few = sc.prediction_6(genetic_rows())
    assert few.value == pytest.approx(100 * (8 / 10 - 4 / 10))  # the trial with 6 is left out
    assert few.predicted is None
    # established: supported 9 + 0 of 20, not supported 3 of 10
    established = 100 * (9 / 20 - 3 / 10)
    assert f"point estimate not lower than the {established:+.1f}" in few.note
    assert "20 supported against 10 not supported" in few.note
    assert "the intervals overlap" in few.note


def rare_scored() -> pl.DataFrame:
    rows = []
    for group, successes, n in (
        ("rare", 3, 10),
        ("not rare", 12, 20),
        ("unclear", 0, 10),
        ("unmapped", 0, 10),
    ):
        rows += [
            {"any_arm": i < successes, "group": group, "sponsor_class": "INDUSTRY"}
            for i in range(n)
        ]
    return frame(rows)


def recruitment_population() -> pl.DataFrame:
    """Phase 2, industry, one era: so the adjusted ratio is the crude one."""
    rows = []

    def add(group, oncology, n, recruitment):
        for i in range(n):
            rows.append(
                {
                    "group": group,
                    "oncology": oncology,
                    "recruitment": i < recruitment,
                    "phase": "phase2",
                    "sponsor": "industry",
                    "era": "2015 to 2020",
                    "sponsor_name": f"{group}-{oncology}-{i}",
                }
            )

    add("rare", False, 10, 2)
    add("not rare", False, 20, 2)
    add("rare", True, 10, 9)  # oncology: must not count
    add("not rare", True, 10, 1)
    return frame(rows)


def test_the_recruitment_ratio_is_rare_over_not_rare_outside_oncology_only():
    row = sc.prediction_7(rare_scored(), recruitment_population())[0]
    assert row.value == pytest.approx((2 / 10) / (2 / 20))
    assert "10 rare, 20 not rare" in row.note
    assert row.predicted is None  # the email says "about 1.2", with no range
    assert row.verdict == NO_RANGE
    assert "email: about 1.2; training-set figure 1.19 (1.00 to 1.40)" in row.note


def test_rare_success_is_rare_minus_not_rare_and_leaves_out_unclear_and_unmapped():
    row = sc.prediction_7(rare_scored(), recruitment_population())[1]
    assert row.value == pytest.approx(100 * (3 / 10 - 12 / 20))
    assert row.note.startswith("10 rare-disease against 20 not rare")


# ---------------------------------------------------------------- prediction 4b and 2b


def test_a_negative_modality_gain_is_inside_the_range_and_says_the_model_did_no_better():
    def difference(value):
        pair = mt.Difference(
            bl.REGISTRATION,
            bl.NO_MODALITY,
            value,
            mt.Interval(value - 0.01, value + 0.01),
            0,
            0,
            mt.Interval(0, 0),
        )
        return SimpleNamespace(
            differences=[pair, replace(pair, better=bl.LEAKY, baseline=bl.REGISTRATION)]
        )

    _, negative = sc.prediction_3_and_4_models(difference(-0.03))
    assert negative.verdict == INSIDE
    assert "did no better with modality" in negative.note
    _, positive = sc.prediction_3_and_4_models(difference(0.02))
    assert positive.verdict != INSIDE and positive.note == ""


def drop(column, loss):
    return ev.DropOne(column, 0.7 - loss, loss, mt.Interval(loss - 0.01, loss + 0.01))


def test_the_top_two_columns_are_found_whatever_the_order_of_the_list():
    separate = mt.Interval(0.001, 0.02)
    text = sc.top_two_table(
        [drop("masking", 0.0), drop("sponsor_class", 0.03), drop("phase", 0.05)], separate
    )
    assert "Observed: phase and sponsor_class. Matches" in text
    assert "above the third" in text
    other = sc.top_two_table(
        [drop("phase", 0.05), drop("masking", 0.03), drop("sponsor_class", 0.0)], separate
    )
    assert "Does not match" in other


def test_a_column_that_does_not_cost_auroc_cannot_matter_most():
    text = sc.top_two_table(
        [drop("phase", 0.05), drop("sponsor_class", -0.01), drop("masking", -0.02)], None
    )
    assert "Does not match" in text
    assert "could not be tested" in text


def test_2b_says_it_has_its_own_rule_and_whether_the_second_is_apart_from_the_third():
    close = mt.Interval(-0.01, 0.02)
    text = sc.top_two_table(
        [drop("phase", 0.05), drop("sponsor_class", 0.03), drop("x", 0.02)], close
    )
    assert "Rule for this row only" in text
    assert "not clearly above the third" in text
    assert "Prediction 2b is not judged by this rule" in sc.RULE


def test_the_second_over_third_interval_is_a_paired_bootstrap_of_the_losses(small_bootstrap):
    dev = synthetic_trials(400, seed=4)
    train, scored = dev[:250], dev[250:]
    drops = sorted(ev.drop_one_column(train, scored), key=lambda d: -d.loss)
    interval = sc.second_over_third(train, scored, drops)
    assert interval.low <= drops[1].loss - drops[2].loss <= interval.high
    assert sc.second_over_third(train, scored, drops[:2]) is None


# ---------------------------------------------------------------- the stopped-trial population


def test_a_population_in_a_later_window_gets_the_later_era_and_a_group_for_every_trial():
    studies = [study("NCT1", completed="2022-05"), study("NCT2", completed="2019-05")]
    population = sc.build_population(
        studies,
        lambda ids: pl.DataFrame({"nct_id": ids, "group": ["rare"] * len(ids)}),
        date(2021, 1, 1),
        date(2024, 1, 1),
    )
    assert population["nct_id"].to_list() == ["NCT1"]
    assert population["era"].to_list() == [sc.LATER_ERA]
    assert population["group"].to_list() == ["rare"]


def test_a_trial_the_flags_do_not_list_counts_as_unmapped():
    population = sc.build_population(
        [study("NCT1", completed="2022-05")],
        lambda ids: pl.DataFrame(
            {"nct_id": [], "group": []}, schema={"nct_id": pl.String, "group": pl.String}
        ),
        date(2021, 1, 1),
        date(2024, 1, 1),
    )
    assert population["group"].to_list() == [sc.UNMAPPED]


def test_flags_that_list_a_trial_twice_are_refused():
    with pytest.raises(ValueError):
        sc.build_population(
            [study("NCT1", completed="2022-05")],
            lambda ids: pl.DataFrame({"nct_id": ["NCT1"] * 2, "group": ["rare"] * 2}),
            date(2021, 1, 1),
            date(2024, 1, 1),
        )


# ---------------------------------------------------------------- the whole scorecard, small


@pytest.fixture(autouse=True)
def small_bootstrap(monkeypatch):
    monkeypatch.setattr(mt, "N_RESAMPLES", 30)


class Fake:
    """A small made-up project: git repo, trials, genetics, raw records and rare-disease groups."""

    def __init__(self, root):
        write_synthetic_project(root)
        self.root = root
        self.trials = pl.read_csv(root / "trials.csv")
        rng = np.random.default_rng(2)
        n = self.trials.height
        self.genetics_table = pl.DataFrame(
            {
                "nct_id": self.trials["nct_id"],
                "split": self.trials["split"],
                "mapped": rng.random(n) < 0.7,
                "supported_today": [
                    None if rng.random() < 0.3 else bool(rng.random() < 0.4) for _ in range(n)
                ],
                "lead_drug_earlier_trials": rng.integers(0, 12, n),
            }
        )
        self.genetics = root / "data" / "interim" / "genetics.parquet"
        self.genetics_table.write_parquet(self.genetics)
        self.flag_calls: list[list[str]] = []
        self.in_test_window = self._ids("test", "2022-05")
        self.studies = self._studies()

    def _ids(self, split, _date):
        return sorted(self.trials.filter(pl.col("split") == split)["nct_id"])[:45]

    def _studies(self):
        made = []
        dated = [(self.in_test_window, "2022-05"), (self._ids("late", ""), "2024-03")]
        dated.append((self._ids("dev", "")[:45], "2018-05"))
        for ids, completed in dated:
            for i, nct in enumerate(ids):
                status = ["COMPLETED", "TERMINATED", "WITHDRAWN"][i % 3]
                s = study(
                    nct, completed=completed, status=status, why="Slow accrual" if i % 3 else None
                )
                s["protocolSection"]["sponsorCollaboratorsModule"]["leadSponsor"]["name"] = f"S{i}"
                made.append(s)
        return made

    def flags(self, ids):
        self.flag_calls.append(list(ids))
        groups = ["rare", "not rare", "unclear"]
        return pl.DataFrame(
            {"nct_id": ids, "group": [groups[i % 3] for i in range(len(ids))]},
            schema={"nct_id": pl.String, "group": pl.String},
        )

    @property
    def sources(self):
        return sc.Sources(self.genetics, lambda: self.studies, self.flags)


@pytest.fixture
def fake(tmp_path):
    return Fake(tmp_path)


@pytest.fixture
def locked(fake):
    lock(fake.root)
    return fake


ROWS = ("1a", "1b", "2a", "3", "4a", "4b", "5a", "5b", "6a", "6b", "7a", "7b")


def test_the_real_run_logs_one_access_writes_the_file_and_has_a_row_for_every_prediction(locked):
    text = sc.run_test("final scorecard", locked.root, locked.sources)
    (entry,) = heldout.access_log(locked.root)
    assert entry["note"] == "final scorecard"
    assert entry["path"] == "data/heldout/test_labels.parquet"
    assert (locked.root / sc.TEST_REPORT).read_text() == text
    for number in ROWS:
        assert f"| {number} |" in text
    assert "scored 150 trials" in text  # the test split, not the late one
    assert "Access note: final scorecard" in text
    assert "opened 1 time(s), counting this run" in text
    assert "final scorecard |" in text  # the access table lists the note
    assert "## Prediction 2b" in text and "Rule for this row only" in text


def test_the_recruitment_population_is_drawn_from_the_test_window(locked):
    sc.run_test("window", locked.root, locked.sources)
    *_, real_call = locked.flag_calls
    assert real_call == locked.in_test_window  # not the 2018 dev trials or the 2024 late ones
    assert locked.flag_calls[0] == []  # the label-free check of the rare-disease files


def test_every_difference_says_which_group_comes_off_which(locked):
    text = sc.run_test("wording", locked.root, locked.sources)
    lines = [line for line in text.splitlines() if line.startswith("| ") and line[2:4] in ROWS]
    for line in lines:
        if line[2:4] not in ("1a", "2a"):
            assert " minus " in line or " over " in line, line


def test_the_second_opening_is_counted_and_both_are_listed(locked):
    sc.run_test("first", locked.root, locked.sources)
    text = sc.run_test("second", locked.root, locked.sources)
    assert "opened 2 time(s)" in text
    assert "| first |" in text and "| second |" in text


def test_a_dirty_working_tree_stops_the_real_run_before_anything_is_opened(locked):
    (locked.root / "stray.py").write_text("print('changed')\n")
    with pytest.raises(RuntimeError, match="uncommitted"):
        sc.run_test("dirty", locked.root, locked.sources)
    assert heldout.access_log(locked.root) == []
    assert not (locked.root / sc.TEST_REPORT).exists()


def break_genetics(fake):
    fake.genetics_table.filter(pl.col("nct_id") != fake.in_test_window[0]).write_parquet(
        fake.genetics
    )


def drop_a_feature_column(fake):
    path = fake.root / "data" / "interim" / "test_features.parquet"
    pl.read_parquet(path).drop("modality").write_parquet(path)


def break_the_flags_reader(fake):
    def refuse(ids):
        raise FileNotFoundError("clinical_report.parquet")

    fake.flags = refuse


@pytest.mark.parametrize(
    "breakage", [break_genetics, drop_a_feature_column, break_the_flags_reader]
)
def test_a_label_free_failure_cannot_burn_a_logged_opening(locked, breakage):
    breakage(locked)
    with pytest.raises((ValueError, FileNotFoundError, pl.exceptions.ColumnNotFoundError)):
        sc.run_test("should not open", locked.root, locked.sources)
    assert heldout.access_log(locked.root) == []


def test_before_the_tag_the_real_run_is_refused_and_logs_nothing(fake):
    with pytest.raises(heldout.HeldoutLocked):
        sc.run_test("too early", fake.root, fake.sources)
    assert heldout.access_log(fake.root) == []


def test_the_rehearsal_runs_the_loaders_and_cannot_reach_the_held_out_labels(fake, monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("the rehearsal reached the held-out labels")

    def guarded(read):
        def wrapper(path, *args, **kwargs):
            if "heldout" in str(path):
                refuse()
            return read(path, *args, **kwargs)

        return wrapper

    for module in (sc, heldout, ev):
        monkeypatch.setattr(module, "open_heldout", refuse, raising=False)
    monkeypatch.setattr(pl, "read_parquet", guarded(pl.read_parquet))
    monkeypatch.setattr(pl, "scan_parquet", guarded(pl.scan_parquet))
    text = sc.rehearse(fake.root, fake.sources)
    assert text.startswith("# REHEARSAL on development data. This is NOT the test set.")
    assert (fake.root / sc.REHEARSAL_REPORT).read_text() == text
    assert heldout.access_log(fake.root) == []
    for number in ROWS:
        assert f"| {number} |" in text
    *_, call = fake.flag_calls
    assert set(call) <= set(fake.trials.filter(pl.col("split") == "dev")["nct_id"])
