import subprocess
from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score
from synthetic import synthetic_trials

from elman_exercise import baselines as bl
from elman_exercise import dataset as ds
from elman_exercise import evaluate as ev
from elman_exercise import metrics as mt
from elman_exercise.heldout import HeldoutLocked, access_log, lock_prereg


@pytest.fixture(autouse=True)
def few_resamples(monkeypatch):
    """Keep the bootstrap quick; its behaviour is tested in test_metrics."""
    monkeypatch.setattr(mt, "N_RESAMPLES", 40)


def with_dates(dates):
    data = synthetic_trials(len(dates), seed=0)
    return data.with_columns(primary_completion_date=pl.Series(dates, dtype=pl.Date))


# ---------------------------------------------------------------- time-forward split


def test_the_time_forward_split_trains_before_2017_and_validates_from_2017_to_the_end_of_2020():
    dates = [
        date(2016, 12, 31), date(2017, 1, 1), date(2020, 12, 31), date(1999, 5, 1),
        date(2018, 6, 30),
    ]  # fmt: skip
    train, validation = ev.time_forward_split(with_dates(dates))
    assert sorted(train["primary_completion_date"]) == [date(1999, 5, 1), date(2016, 12, 31)]
    assert sorted(validation["primary_completion_date"]) == [
        date(2017, 1, 1), date(2018, 6, 30), date(2020, 12, 31),
    ]  # fmt: skip


def test_the_time_forward_split_refuses_trials_from_2021_on():
    data = with_dates([date(2019, 1, 1), date(2021, 1, 1)])
    with pytest.raises(ValueError, match="2021 or later"):
        ev.time_forward_split(data)


def test_every_trial_lands_in_exactly_one_side_of_the_time_forward_split():
    data = synthetic_trials(400, seed=1)
    train, validation = ev.time_forward_split(data)
    assert train.height + validation.height == data.height
    assert set(train["nct_id"]).isdisjoint(validation["nct_id"])


# ---------------------------------------------------------------- the two evaluations


def test_time_forward_scores_show_the_ladder_the_data_were_built_with():
    evaluation = ev.evaluate_time_forward(synthetic_trials(1500, seed=2))
    auroc = {name: s.auroc for name, s in evaluation.scores.items()}
    assert auroc[bl.CONSTANT] == pytest.approx(0.5)
    assert auroc[bl.PHASE_LOOKUP] > 0.55
    assert auroc[bl.REGISTRATION] > auroc[bl.PHASE_LOOKUP]
    assert auroc[bl.LEAKY] > auroc[bl.REGISTRATION]
    assert [d.better for d in evaluation.differences] == [
        bl.REGISTRATION,
        bl.LEAKY,
        bl.REGISTRATION,
    ]


def test_the_majority_floor_comes_from_the_training_trials_not_the_scored_ones():
    data = synthetic_trials(800, seed=3).with_columns(
        any_arm=pl.when(pl.col("primary_completion_date") < date(2017, 1, 1))
        .then(pl.int_range(pl.len()) % 10 != 0)  # 90% succeed before 2017
        .otherwise(False)  # none do after
    )
    evaluation = ev.evaluate_time_forward(data)
    assert evaluation.majority_class is True
    assert evaluation.scores[bl.CONSTANT].majority_accuracy == 0.0  # everything scored failed


def test_cross_validation_predicts_every_trial_once_with_a_model_that_did_not_see_it():
    data = synthetic_trials(500, seed=4)
    evaluation = ev.evaluate_cross_validated(data)
    for predicted in evaluation.predictions.values():
        assert predicted.shape == (data.height,)
        assert np.all((predicted > 0) & (predicted < 1))


def test_cross_validation_is_deterministic():
    data = synthetic_trials(400, seed=5)
    one = ev.evaluate_cross_validated(data)
    two = ev.evaluate_cross_validated(data)
    for name in bl.RUNGS:
        assert np.array_equal(one.predictions[name], two.predictions[name])
    assert one.scores == two.scores


# ---------------------------------------------------------------- subsets


def made_evaluation(table, outcomes):
    predictions = {name: np.linspace(0.05, 0.95, table.height) for name in bl.RUNGS}
    return ev.score_ladder(table.with_columns(any_arm=pl.Series(outcomes)), predictions, True)


def test_a_subset_with_fewer_than_30_trials_gets_counts_but_no_auroc():
    table = synthetic_trials(200, seed=6).with_columns(
        modality=pl.when(pl.int_range(pl.len()) < 29)
        .then(pl.lit("vaccine"))
        .otherwise(pl.lit("biologic"))
    )
    rng = np.random.default_rng(0)
    rows = ev.subset_rows(
        made_evaluation(table, rng.random(200) < 0.5), bl.REGISTRATION, [ev.MODALITY]
    )
    vaccine = next(r for r in rows if r.subset == "vaccine")
    assert (vaccine.n, vaccine.scores, vaccine.reason) == (29, None, "n < 30")
    table_text = ev.subsets_table(rows)
    vaccine_line = next(line for line in table_text.splitlines() if "| vaccine |" in line)
    assert vaccine_line.count("not computed (n < 30)") == 2  # AUROC and calibration gap
    assert not np.isnan(vaccine.observed_rate)
    biologic = next(r for r in rows if r.subset == "biologic")
    assert biologic.scores is not None and biologic.n == 171


def test_a_subset_with_one_class_gets_a_gap_interval_but_no_auroc():
    table = synthetic_trials(200, seed=7).with_columns(
        control_kind=pl.when(pl.int_range(pl.len()) < 40)
        .then(pl.lit("active"))
        .otherwise(pl.lit("placebo"))
    )
    outcomes = np.arange(200) >= 40  # every active-control trial failed
    rows = ev.subset_rows(made_evaluation(table, outcomes), bl.REGISTRATION, [ev.CONTROL_KIND])
    active = next(r for r in rows if r.subset == "active")
    assert (active.n, active.reason) == (40, "one class only")
    assert active.observed_rate == 0.0
    assert active.scores is not None and np.isnan(active.scores.auroc)
    line = next(x for x in ev.subsets_table(rows).splitlines() if "| active |" in x)
    assert "not computed (one class only)" in line and "to" in line.split("|")[-2]


def test_subsets_only_list_what_the_spec_names_and_sponsors_split_industry_from_the_rest():
    table = synthetic_trials(120, seed=8).with_columns(
        sponsor_class=pl.Series(["INDUSTRY", "NIH", "OTHER", "OTHER_GOV_OR_NETWORK"] * 30),
        control_kind=pl.lit("no treatment"),
    )
    outcomes = np.tile([True, False, True, False], 30)
    evaluation = made_evaluation(table, outcomes)
    rows = ev.subset_rows(evaluation, bl.REGISTRATION, [ev.SPONSOR, ev.CONTROL_KIND])
    counts = {(r.group, r.subset): r.n for r in rows}
    assert counts == {
        ("sponsor class", "industry"): 30,
        ("sponsor class", "not industry"): 90,
        ("control kind", "placebo"): 0,
        ("control kind", "active"): 0,
    }
    assert ev.not_shown(evaluation, [ev.CONTROL_KIND]) == ["control kind: no treatment (120)"]


def test_enrollment_bands_use_the_summary_cut_points():
    table = synthetic_trials(100, seed=9).with_columns(
        enrollment=pl.Series([99, 100, 299, 300, 999, 1000, None, 50, 50, 50] * 10)
    )
    rows = ev.subset_rows(
        made_evaluation(table, np.arange(100) % 2 == 0), bl.REGISTRATION, [ev.ENROLLMENT]
    )
    assert {r.subset: r.n for r in rows} == {"<100": 40, "100-299": 20, "300-999": 20, "1000+": 10}


# ---------------------------------------------------------------- the development report


def test_the_development_report_is_the_same_on_every_run_and_names_every_model():
    data = synthetic_trials(500, seed=10)
    first = ev.dev_report(data, "Header.")
    second = ev.dev_report(data, "Header.")
    assert first == second
    for name in bl.RUNGS:
        assert f"| {name} |" in first
    assert "registration minus phase lookup" in first
    assert "registration minus registration, no modality" in first
    assert "leaky minus registration" in first
    assert "## A. Time-forward validation" in first and "## B. Five-fold" in first
    assert "resamples of lead drugs" in first
    assert "converged" in first
    assert "coefficient" not in first.lower()


def test_the_development_report_says_which_slicing_columns_are_inputs_and_which_are_not():
    text = ev.dev_report(synthetic_trials(500, seed=10), "Header.")
    assert "used to slice, not as inputs" not in text
    assert "modality, phase, sponsor class are inputs to the model" in text
    assert "strongest input is held fixed" in text
    assert "control kind (it comes from the results section)" in text
    assert "enrollment (actual, post-hoc) (it is the actual figure, recorded after" in text


def test_control_kinds_that_are_not_shown_are_counted_and_explained():
    table = synthetic_trials(120, seed=8).with_columns(
        control_kind=pl.Series(["placebo", "active", "no treatment", "unspecified"] * 30)
    )
    evaluation = made_evaluation(table, np.arange(120) % 3 == 0)
    text = ev.ladder_section(evaluation, "## X", ev.LEAD_GROUPINGS)
    assert "control kind: no treatment (30), unspecified (30)" in text
    assert "not the placebo-against-active contrast" in text


def test_subset_skips_are_reported_apart_from_the_headline_tables():
    text = ev.dev_report(synthetic_trials(500, seed=10), "Header.")
    assert text.count("holding one class only, headline tables") == 2  # sections A and B
    assert text.count("holding one class only, subset table") == 1  # subsets only in A


def test_the_cross_validated_figures_are_scored_on_the_same_trials_as_the_time_forward_ones():
    data = synthetic_trials(900, seed=11)
    forward = ev.evaluate_time_forward(data)
    shuffled = ev.evaluate_cross_validated(data)
    rows = ev.same_trials_comparison(forward, shuffled)
    assert [r.model for r in rows] == list(bl.RUNGS)
    mask = (data["primary_completion_date"] >= ev.VALIDATION_START).to_numpy()
    for r in rows:
        assert r.forward.n == r.cross_validated.n == forward.table.height == mask.sum()
        assert r.forward.observed_rate == r.cross_validated.observed_rate
        by_hand = mt.auroc(forward.outcomes, shuffled.predictions[r.model][mask])
        assert r.cross_validated.auroc == pytest.approx(by_hand)
        assert r.difference.difference == pytest.approx(by_hand - r.forward.auroc)
        assert r.difference.gap_difference == pytest.approx(
            r.cross_validated.calibration_gap - r.forward.calibration_gap
        )


def test_the_comparison_section_says_what_a_random_split_hides():
    text = ev.dev_report(synthetic_trials(900, seed=11), "Header.")
    assert "## C. Time-forward against cross-validated, on the same trials" in text
    assert "A random split hides how over-optimistic the model is; it does not flatter" in text
    assert "rather than 0.5 because each fold predicts a slightly different constant" in text


def test_drop_one_column_has_a_row_per_registration_column_sorted_by_loss():
    data = synthetic_trials(900, seed=12)
    train, validation = ev.time_forward_split(data)
    rows = ev.drop_one_column(train, validation)
    assert sorted(r.column for r in rows) == sorted(bl.REGISTRATION_COLUMNS)
    losses = [r.loss for r in rows]
    assert losses == sorted(losses, reverse=True)
    full = bl.fit_logistic(train, bl.FEATURES[bl.REGISTRATION]).predict_probability(validation)
    sponsor = next(r for r in rows if r.column == "sponsor_class")
    without = bl.fit_logistic(
        train, bl.registration_features_without("sponsor_class")
    ).predict_probability(validation)
    outcomes = bl.training_outcomes(validation)
    assert sponsor.auroc_without == pytest.approx(mt.auroc(outcomes, without))
    assert sponsor.loss == pytest.approx(mt.auroc(outcomes, full) - mt.auroc(outcomes, without))
    assert sponsor.loss > 0.02  # the synthetic data make sponsor matter
    assert sponsor.interval.low < sponsor.loss < sponsor.interval.high


def test_the_development_report_refuses_data_from_2021_on():
    data = with_dates([date(2019, 1, 1), date(2021, 6, 1)])
    with pytest.raises(ValueError, match="2021 or later"):
        ev.dev_report(data, "Header.")


# ---------------------------------------------------------------- the test evaluation (synthetic only)


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def write_synthetic_project(root):
    """A temporary repo laid out like the real one, holding only made-up trials."""
    git(root, "init", "-q")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "user.name", "Test")
    (root / ".gitignore").write_text("data/\nresults/\n")
    (root / "PREREGISTRATION.md").write_text("# Pre-registration\n\nTODO\n")
    dev = synthetic_trials(500, seed=1)
    test = synthetic_trials(
        150, seed=2, split="test", first_day=date(2021, 1, 1), days=1000, prefix="NCT1"
    )
    late = synthetic_trials(
        60, seed=3, split="late", first_day=date(2024, 1, 1), days=700, prefix="NCT2"
    )
    held = pl.concat([test, late])
    not_labels = [c for c in held.columns if c not in ds.LABEL_COLUMNS]
    (root / "data" / "interim").mkdir(parents=True)
    (root / "data" / "heldout").mkdir(parents=True)
    dev.write_parquet(root / "data" / "interim" / "dev.parquet")
    held.select(not_labels).write_parquet(root / "data" / "interim" / "test_features.parquet")
    held.select("nct_id", "split", *ds.LABEL_COLUMNS).write_parquet(
        root / "data" / "heldout" / "test_labels.parquet"
    )
    pl.concat([dev, held]).select("nct_id", "split").sort("nct_id").write_csv(root / "trials.csv")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "init")


def lock(root):
    (root / "PREREGISTRATION.md").write_text("# Pre-registration\n\nI expect X because Y.\n")
    git(root, "commit", "-q", "-am", "prereg")
    lock_prereg(root)


@pytest.fixture
def project(tmp_path):
    write_synthetic_project(tmp_path)
    return tmp_path


@pytest.fixture
def locked_project(project):
    lock(project)
    return project


def test_the_test_evaluation_refuses_before_the_prereg_tag_exists(project):
    with pytest.raises(HeldoutLocked):
        ev.evaluate_on_test("too early", root=project)
    assert access_log(project) == []
    assert not (project / ev.TEST_REPORT).exists()


def test_after_the_tag_the_test_evaluation_runs_logs_the_access_and_writes_the_report(
    locked_project,
):
    text = ev.evaluate_on_test("final eval", root=locked_project)
    (entry,) = access_log(locked_project)
    assert entry["note"] == "final eval"
    assert entry["path"] == "data/heldout/test_labels.parquet"
    assert (locked_project / ev.TEST_REPORT).read_text() == text
    assert "Primary test set" in text and "Extended test set" not in text
    assert "Trials scored: 150 test." in text
    for name in bl.RUNGS:
        assert f"| {name} |" in text
    for grouping in ("control kind", "enrollment", "modality"):
        assert grouping in text
    assert "| phase |" not in text  # only the three lead subsets on the test set


def test_asking_for_the_late_trials_adds_the_extended_set(locked_project):
    text = ev.evaluate_on_test("extended eval", splits=("test", "late"), root=locked_project)
    assert "Primary test set" in text and "Extended test set" in text
    assert len(access_log(locked_project)) == 1


def test_the_test_evaluation_scores_the_trials_it_was_asked_for(locked_project):
    text = ev.evaluate_on_test("check counts", splits=("test", "late"), root=locked_project)
    assert "\n150 trials; observed success rate" in text
    assert "\n210 trials; observed success rate" in text


def test_the_test_evaluation_needs_the_primary_set(locked_project):
    with pytest.raises(ValueError, match="primary"):
        ev.evaluate_on_test("late only", splits=("late",), root=locked_project)
    assert access_log(locked_project) == []


def test_mismatched_feature_ids_stop_the_evaluation_before_the_labels_are_opened(locked_project):
    trials = pl.read_csv(locked_project / "trials.csv")
    trials.filter(pl.col("nct_id") != "NCT10000003").write_csv(locked_project / "trials.csv")
    with pytest.raises(ev.IdsDisagree, match="test_features.parquet"):
        ev.evaluate_on_test("mismatch", root=locked_project)
    assert access_log(locked_project) == []
    assert not (locked_project / ev.TEST_REPORT).exists()


def test_mismatched_label_ids_stop_the_evaluation_before_any_score(locked_project):
    path = locked_project / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    labels.filter(pl.col("nct_id") != "NCT10000003").write_parquet(path)
    with pytest.raises(ev.IdsDisagree, match="test_labels.parquet"):
        ev.evaluate_on_test("mismatch", root=locked_project)
    assert not (locked_project / ev.TEST_REPORT).exists()


def test_a_trial_whose_split_differs_between_files_stops_the_evaluation(locked_project):
    path = locked_project / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    moved = labels.with_columns(
        split=pl.when(pl.col("nct_id") == "NCT10000003")
        .then(pl.lit("late"))
        .otherwise(pl.col("split"))
    )
    moved.write_parquet(path)
    with pytest.raises(ev.IdsDisagree):
        ev.evaluate_on_test("mismatch", root=locked_project)


def test_a_clean_repo_gets_a_clean_test_report_even_though_opening_the_labels_writes_the_log(
    locked_project,
):
    text = ev.evaluate_on_test("provenance", root=locked_project)
    assert "(clean)" in text and "uncommitted" not in text
    again = ev.evaluate_on_test("provenance again", root=locked_project)
    assert "(clean)" in again


def test_the_test_report_still_flags_real_uncommitted_changes(locked_project):
    (locked_project / "stray.py").write_text("x = 1\n")
    text = ev.evaluate_on_test("dirty", root=locked_project)
    assert "(with uncommitted changes)" in text


def test_the_test_report_counts_every_time_the_labels_were_opened(locked_project):
    first = ev.evaluate_on_test("first look", root=locked_project)
    assert "opened 1 time(s)" in first
    second = ev.evaluate_on_test("second look", root=locked_project)
    assert "opened 2 time(s)" in second
    assert "first look" in second and "second look" in second
    assert (locked_project / ev.TEST_REPORT).read_text() == second
    entries = access_log(locked_project)
    for entry in entries:
        assert entry["at"] in second and entry["commit"][:12] in second


def test_the_late_set_is_reported_on_its_own_only_when_asked_for(locked_project):
    plain = ev.evaluate_on_test("plain", root=locked_project)
    assert "Late test set" not in plain and "late" not in plain.split("Access note")[0].lower()
    assert "Trials scored: 150 test." in plain
    extended = ev.evaluate_on_test("extended", splits=("test", "late"), root=locked_project)
    assert "Trials scored: 150 test, 60 late." in extended
    assert "Late test set (finished 2024 to 2025)" in extended
    assert "\n60 trials; observed success rate" in extended
    assert "\n150 trials; observed success rate" in extended
    assert "\n210 trials; observed success rate" in extended


def test_the_primary_test_scores_match_an_independent_calculation(locked_project):
    text = ev.evaluate_on_test("independent", root=locked_project)
    root = locked_project
    dev = pl.read_parquet(root / "data" / "interim" / "dev.parquet")
    features = pl.read_parquet(root / "data" / "interim" / "test_features.parquet")
    labels = pl.read_parquet(root / "data" / "heldout" / "test_labels.parquet")
    both = features.join(labels.select("nct_id", "any_arm"), on="nct_id").filter(
        pl.col("split") == "test"
    )
    truth = both["any_arm"].to_numpy().astype(bool)
    section = text.split("## Primary test set")[1].split("\n## ")[0]
    for rung in (bl.CONSTANT, bl.REGISTRATION, bl.LEAKY):
        predicted = bl.RUNGS[rung](dev).predict_probability(both)
        row = next(line for line in section.splitlines() if line.startswith(f"| {rung} |"))
        cells = [c.strip() for c in row.strip("|").split("|")]
        assert float(cells[3].split()[0]) == pytest.approx(
            roc_auc_score(truth, predicted), abs=5e-4
        )
        assert float(cells[4]) == pytest.approx(brier_score_loss(truth, predicted), abs=5e-4)


def test_a_repeated_trial_id_in_trials_csv_stops_the_evaluation_before_the_labels_are_opened(
    locked_project,
):
    path = locked_project / "trials.csv"
    trials = pl.read_csv(path)
    pl.concat([trials, trials.head(1)]).write_csv(path)
    with pytest.raises(ev.IdsDisagree, match="trials.csv lists 1 trial ids more than once"):
        ev.evaluate_on_test("repeat", root=locked_project)
    assert access_log(locked_project) == []


def test_a_repeated_trial_id_in_the_features_stops_the_evaluation_before_the_labels_are_opened(
    locked_project,
):
    path = locked_project / "data" / "interim" / "test_features.parquet"
    features = pl.read_parquet(path)
    pl.concat([features, features.head(1)]).write_parquet(path)
    with pytest.raises(ev.IdsDisagree, match="test_features lists 1 trial ids"):
        ev.evaluate_on_test("repeat", root=locked_project)
    assert access_log(locked_project) == []


def test_a_repeated_trial_id_in_the_labels_stops_the_evaluation_before_any_score(locked_project):
    path = locked_project / "data" / "heldout" / "test_labels.parquet"
    labels = pl.read_parquet(path)
    pl.concat([labels, labels.head(1)]).write_parquet(path)
    with pytest.raises(ev.IdsDisagree, match="test_labels.parquet lists 1 trial ids"):
        ev.evaluate_on_test("repeat", root=locked_project)
    assert not (locked_project / ev.TEST_REPORT).exists()


def test_the_label_join_is_on_trial_and_split_and_gives_one_row_per_trial():
    features = synthetic_trials(6, seed=1, split="test", with_labels=False)
    labels = synthetic_trials(6, seed=1, split="test").select("nct_id", "split", "any_arm")
    assert ev.join_labels(features, labels).height == 6
    moved = labels.with_columns(split=pl.lit("late"))  # same ids, another split
    with pytest.raises(ev.IdsDisagree, match="0 rows for 6 trials"):
        ev.join_labels(features, moved)
    doubled = pl.concat([labels, labels.head(1)])
    with pytest.raises(ev.IdsDisagree, match="7 rows for 6 trials"):
        ev.join_labels(features, doubled)
