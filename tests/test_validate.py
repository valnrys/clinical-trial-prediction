from datetime import date

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from trial_benchmark import metrics
from trial_benchmark import validate as v
from trial_benchmark.models.registry import ModelSpec, all_specs


def dated(*days: date) -> pl.DataFrame:
    return pl.DataFrame({"primary_completion_date": list(days)})


def test_a_round_trains_before_its_first_year_and_scores_its_two_years():
    frame = dated(
        date(2014, 12, 31),  # last day of training
        date(2015, 1, 1),  # first scored day
        date(2016, 12, 31),  # last scored day
        date(2017, 1, 1),  # after the round: neither
    )
    train, scored = v.time_round(frame, 2015, 2016)
    assert train.tolist() == [0]
    assert scored.tolist() == [1, 2]


def test_the_three_rounds_never_train_on_what_they_score_and_never_score_a_trial_twice():
    frame = synthetic_trials(500, days=365 * 14, first_day=date(2008, 1, 1))
    scored_before: set[int] = set()
    for first, last in v.ROUNDS:
        train, scored = v.time_round(frame, first, last)
        assert not set(train) & set(scored)
        assert not scored_before & set(scored)
        scored_before |= set(scored)
        years = frame["primary_completion_date"].dt.year().to_numpy()
        assert years[train].max() < first <= years[scored].min()
        assert years[scored].max() <= last


def test_the_rounds_are_the_ones_the_brief_names():
    assert v.ROUNDS == ((2015, 2016), (2017, 2018), (2019, 2020))


def test_grouped_folds_never_split_a_drug_and_cover_every_trial_once():
    rng = np.random.default_rng(0)
    groups = rng.integers(0, 60, 400).astype(str)
    folds = v.grouped_folds(groups, n_folds=10)
    assert len(folds) == 10
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])
    tested = np.concatenate([test for _, test in folds])
    assert sorted(tested.tolist()) == list(range(400))


def test_a_trial_with_no_known_drug_is_its_own_group():
    frame = pl.DataFrame({"nct_id": ["A", "B", "C"], "lead_drug": ["X", None, "X"]})
    assert v.drug_groups(frame).tolist() == ["X", "B", "X"]


def test_specs_with_a_missing_table_are_skipped_and_named(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "INTERIM", tmp_path)
    (tmp_path / "here.parquet").write_bytes(b"")
    ok = ModelSpec("ok", "t", "d", fit=lambda f: None, tables=("here",))
    bad = ModelSpec("bad", "t", "d", fit=lambda f: None, tables=("here", "gone"))
    plain = ModelSpec("plain", "t", "d", fit=lambda f: None)
    usable, missing = v.split_specs([ok, bad, plain])
    assert [s.name for s in usable] == ["ok", "plain"]
    assert missing == {"bad": ["gone"]}


def test_predict_all_hands_models_a_frame_without_label_columns():
    seen_columns = []

    class Spy:
        def predict(self, frame):
            seen_columns.append(set(frame.columns))
            return np.full(frame.height, 0.5)

    spec = ModelSpec("spy", "t", "d", fit=lambda train: Spy())
    train = synthetic_trials(50, seed=1)
    test = synthetic_trials(20, seed=2, prefix="NCT1")
    v.predict_all([spec], train, test)
    assert not {"any_arm", "all_arms", "highest_dose", "has_dose_ladder"} & seen_columns[0]


def test_predict_all_refuses_missing_predictions_and_names_the_model():
    class Broken:
        def predict(self, frame):
            return np.full(frame.height, np.nan)

    spec = ModelSpec("broken one", "t", "d", fit=lambda train: Broken())
    frame = synthetic_trials(20, seed=1)
    with pytest.raises(ValueError, match="broken one"):
        v.predict_all([spec], frame, frame)


def rounds_fixture(n_per_round: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    round_of = np.repeat([0, 1, 2], n_per_round)
    outcomes = rng.random(len(round_of)) < 0.4
    reference = rng.random(len(round_of)) + 0.5 * outcomes
    return outcomes, reference, round_of


def test_a_model_that_is_a_per_round_monotone_transform_of_the_reference_differs_by_exactly_zero():
    outcomes, reference, round_of = rounds_fixture()
    scale = np.array([0.1, 1.0, 5.0])[round_of]
    shifted = np.exp(reference * scale + round_of)  # a different increasing map in each round
    predictions = {v.REGISTRATION_NAME: reference, "shifted": shifted}
    evaluation = v.summarise(outcomes, predictions, round_of)
    difference = evaluation.differences["shifted"]
    assert difference.value == 0.0
    assert (difference.interval.low, difference.interval.high) == (0.0, 0.0)


def test_a_different_constant_each_round_scores_exactly_one_half_in_every_round():
    outcomes, reference, round_of = rounds_fixture()
    constant = np.array([0.30, 0.40, 0.50])[round_of]
    predictions = {v.REGISTRATION_NAME: reference, "constant": constant}
    evaluation = v.summarise(outcomes, predictions, round_of)
    assert evaluation.aurocs["constant"].per_group == (0.5, 0.5, 0.5)
    assert evaluation.aurocs["constant"].value == 0.5
    pooled = metrics.auroc(outcomes, constant)
    assert pooled != 0.5  # the pooled figure would have been fooled


def test_the_headline_auroc_is_the_mean_of_the_within_round_aurocs_with_an_interval_around_it():
    outcomes, reference, round_of = rounds_fixture()
    evaluation = v.summarise(outcomes, {v.REGISTRATION_NAME: reference}, round_of)
    mean = evaluation.aurocs[v.REGISTRATION_NAME]
    assert mean.value == pytest.approx(np.mean(mean.per_group))
    assert mean.interval.low < mean.value < mean.interval.high


def test_summarise_reports_a_difference_for_every_model_but_the_reference():
    outcomes, reference, round_of = rounds_fixture(100)
    rng = np.random.default_rng(1)
    predictions = {
        v.REGISTRATION_NAME: reference,
        "a": rng.random(len(reference)),
        "b": rng.random(len(reference)),
    }
    evaluation = v.summarise(outcomes, predictions, round_of)
    assert set(evaluation.differences) == {"a", "b"}


def test_models_marked_not_ranked_are_left_out_of_the_rankings_with_their_reason():
    plain = ModelSpec("plain", "t", "d", fit=lambda f: None)
    # The description says nothing about it: only the field decides.
    partial = ModelSpec(
        "partial", "t", "d", fit=lambda f: None, ranked=False, not_ranked_reason="only some trials"
    )
    lines = v.under_headings(
        [plain, partial], ["partial", "plain"], lambda names: [", ".join(names)]
    )
    assert "plain" in lines and "partial" not in lines  # only the table row is the bare name
    # Said once, not under every table: `under_headings` leaves it to `not_ranked_lines`.
    assert not any(line.startswith("Not ranked") for line in lines)
    assert v.not_ranked_lines([plain, partial]) == [
        "Not ranked in any table below (only some trials): partial.",
        "",
    ]


def test_a_model_cannot_be_unranked_without_a_reason():
    with pytest.raises(ValueError, match="gives no reason"):
        ModelSpec("m", "t", "d", fit=lambda f: None, ranked=False)


def test_the_science_only_and_language_model_specs_are_the_unranked_ones():
    unranked = {s.name for s in all_specs() if not s.ranked}
    assert unranked == {
        s.name
        for s in all_specs()
        if s.family == "llm" or (s.family == "biology" and "Defined on trials" in s.description)
    }
    assert any(s.family == "biology" for s in all_specs() if not s.ranked)


def test_specs_that_read_their_own_table_are_skipped_when_it_is_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(v, "INTERIM", tmp_path)
    own = ModelSpec("own", "t", "d", fit=lambda f: None, own_tables=("stored",))
    usable, missing = v.split_specs([own])
    assert usable == [] and missing == {"own": ["stored"]}
    (tmp_path / "stored.parquet").write_bytes(b"")
    assert v.split_specs([own])[0] == [own]


def _bootstrap_inputs(n=240, seed=3):
    rng = np.random.default_rng(seed)
    outcomes = rng.random(n) < 0.5
    predictions = {
        "a": np.clip(outcomes * 0.2 + rng.random(n) * 0.8, 0, 1),
        "b": rng.random(n),
    }
    groups = np.repeat([0, 1, 2], n // 3)
    return outcomes, predictions, groups


def test_mean_auroc_bootstrap_without_clusters_is_unchanged():
    """The old loop, copied here, gives the same intervals as the function without `clusters`."""
    outcomes, predictions, groups = _bootstrap_inputs()
    n_resamples = 50
    rows = [np.flatnonzero(groups == g) for g in np.unique(groups)]
    rng = np.random.default_rng(metrics.BOOTSTRAP_SEED)
    old = {name: np.empty((n_resamples, len(rows))) for name in predictions}
    for i in range(n_resamples):
        for j, r in enumerate(rows):
            chosen = r[rng.integers(0, len(r), size=len(r))]
            for name, p in predictions.items():
                old[name][i, j] = metrics.auroc(outcomes[chosen], p[chosen])
    means, diffs = v.mean_auroc_bootstrap(
        outcomes, predictions, groups, [("a", "b")], n_resamples=n_resamples
    )
    assert means["a"].interval == v.percentile_interval(old["a"].mean(axis=1))
    expected = v.percentile_interval(old["a"].mean(axis=1) - old["b"].mean(axis=1))
    assert diffs["a"].interval == expected


def test_mean_auroc_bootstrap_by_cluster_draws_whole_clusters_inside_each_round():
    outcomes, predictions, groups = _bootstrap_inputs()
    kwargs = {"pairs": [("a", "b")], "n_resamples": 200}
    one_cluster_per_round = np.array([f"drug{g}" for g in groups], dtype=object)
    means, _ = v.mean_auroc_bootstrap(
        outcomes, predictions, groups, clusters=one_cluster_per_round, **kwargs
    )
    # every resample is the whole round, so there is no spread at all
    assert means["a"].interval.low == pytest.approx(means["a"].value)
    assert means["a"].interval.high == pytest.approx(means["a"].value)
    # one trial per cluster (None = its own cluster) behaves like resampling trials: a real spread
    alone = np.array([None] * len(outcomes), dtype=object)
    spread, _ = v.mean_auroc_bootstrap(outcomes, predictions, groups, clusters=alone, **kwargs)
    assert spread["a"].interval.high - spread["a"].interval.low > 0.02
    with pytest.raises(ValueError, match="clusters has"):
        v.mean_auroc_bootstrap(outcomes, predictions, groups, clusters=alone[:5], **kwargs)


def test_lead_drug_clusters_use_the_trials_own_id_when_there_is_no_drug():
    frame = pl.DataFrame({"nct_id": ["N1", "N2", "N3"], "lead_drug": ["x", "x", None]})
    assert v.lead_drug_clusters(frame).tolist() == ["x", "x", "N3"]


def test_a_drug_spanning_rounds_is_in_or_out_of_every_round_together():
    # Outcomes and scores are built so a round's AUROC is 1 only if drug "A" is present in it.
    outcomes = np.array([True, False, True, False, True, False])
    predictions = {"m": np.array([0.9, 0.1, 0.9, 0.1, 0.5, 0.5])}
    groups = np.array([0, 0, 1, 1, 1, 1])
    clusters = np.array(["A", "A", "A", "A", "B", "C"], dtype=object)
    means, _ = v.mean_auroc_bootstrap(outcomes, predictions, groups, [], 400, clusters=clusters)
    per_round = means["m"].per_group
    assert per_round[0] == 1.0
    # round 0 holds only drug A, so any resample without A has no round-0 AUROC and is dropped;
    # in every kept resample A is in round 1 too, so round 1's AUROC is at least its share of A.
    assert means["m"].interval.low > 0.5
