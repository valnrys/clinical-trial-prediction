from datetime import date

import numpy as np
import polars as pl
from synthetic import synthetic_trials

from elman_exercise import split_study as ss
from elman_exercise.models.linear import REGISTRATION_NAME

CHEAP = (REGISTRATION_NAME, "lookup: phase", "lookup: constant")


def with_drugs(n: int, n_drugs: int, seed: int = 0) -> pl.DataFrame:
    """Synthetic trials spread over 2008-2020, each with one of `n_drugs` lead drugs."""
    frame = synthetic_trials(n, seed=seed, days=365 * 12, first_day=date(2008, 1, 1))
    drugs = np.random.default_rng(seed).integers(0, n_drugs, n)
    return frame.with_columns(pl.Series("lead_drug", [f"D{d}" for d in drugs]))


def test_chronological_holdout_has_no_trial_earlier_than_any_training_trial():
    frame = with_drugs(600, 80)
    pool, later = ss.chronological_split(frame, 2016)
    assert pool.height + later.height == frame.height
    assert pool["primary_completion_date"].max() < later["primary_completion_date"].min()
    assert later["primary_completion_date"].dt.year().min() >= 2016
    assert pool["primary_completion_date"].dt.year().max() < 2016


def test_random_by_trial_holdout_has_exactly_the_asked_size_with_no_repeats():
    held = ss.trial_holdout(500, 120, np.random.default_rng(1))
    assert len(held) == 120 == len(set(held))
    assert held.max() < 500


def test_random_by_drug_holdout_never_puts_a_drug_on_both_sides():
    groups = np.array([f"D{d}" for d in np.random.default_rng(2).integers(0, 60, 700)])
    for seed in range(10):
        held = ss.drug_holdout(groups, 150, np.random.default_rng(seed))
        rest = np.setdiff1d(np.arange(len(groups)), held)
        assert not set(groups[held]) & set(groups[rest])
        assert len(held) <= 150


def test_random_by_drug_holdout_lands_at_or_just_under_the_asked_size():
    groups = np.array([f"D{d}" for d in np.random.default_rng(3).integers(0, 200, 1000)])
    sizes = [len(ss.drug_holdout(groups, 200, np.random.default_rng(s))) for s in range(10)]
    assert min(sizes) >= 190 and max(sizes) <= 200


def test_study_runs_on_synthetic_trials_and_scores_sensibly():
    frame = with_drugs(500, 60)
    pool, later = ss.chronological_split(frame, 2016)
    result = ss.run_study(pool, later, CHEAP, n_seeds=3, cut_year=2016, jobs=1)
    assert set(result.auroc) == set(ss.REGIMES)
    assert result.auroc[ss.CHRONO][REGISTRATION_NAME] > 0.5
    # a model that says one number for everyone ranks nobody above anybody
    assert result.auroc[ss.CHRONO]["lookup: constant"] == 0.5
    # a random holdout is a mix of old and later trials; the chronological one is all later
    assert result.composition[ss.CHRONO]["share_older"] == 0.0
    assert 0 < result.composition[ss.BY_TRIAL]["share_older"] < 1
    # the random-split score is shown with the range of its seeds, which holds the mean
    low, high = result.auroc_ci[ss.BY_TRIAL][REGISTRATION_NAME]
    assert low <= result.auroc[ss.BY_TRIAL][REGISTRATION_NAME] <= high


def test_every_random_fit_trains_on_none_of_its_holdout_and_every_holdout_is_the_asked_size():
    frame = with_drugs(500, 60)
    groups = ss.drug_groups(frame)
    plan, holds = ss.build_plan(350, 150, groups, 4)
    for train_rows, test_rows in plan.values():
        assert not set(train_rows) & set(test_rows)
        assert len(train_rows) + len(test_rows) == 500
    assert all(len(h) == 150 for h in holds[ss.BY_TRIAL])
    assert all(0 < len(h) <= 150 for h in holds[ss.BY_DRUG])
    assert len(holds[ss.BY_TRIAL]) == len(holds[ss.BY_DRUG]) == 4


def test_a_fit_never_sees_the_labels_of_the_trials_it_predicts():
    # Whatever the regime, a scored trial is predicted by `_fit_predict`, which drops its labels.
    frame = with_drugs(400, 50)
    train, test = frame[:300], frame[300:]
    flipped = test.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    a = ss._fit_predict(list(CHEAP), train, test)
    b = ss._fit_predict(list(CHEAP), train, flipped)
    for name in CHEAP:
        assert np.array_equal(a[name], b[name])


def test_a_cache_file_is_not_reused_for_a_different_holdout(tmp_path):
    cache = tmp_path / "fits.pkl"
    frame = with_drugs(500, 60)
    pool, later = ss.chronological_split(frame, 2016)
    first = ss.run_study(pool, later, CHEAP, n_seeds=2, cut_year=2016, jobs=1, cache_file=cache)
    # a different holdout (cut a year later) with the same cache file and the same settings
    pool2, later2 = ss.chronological_split(frame, 2017)
    other = ss.run_study(pool2, later2, CHEAP, n_seeds=2, cut_year=2017, jobs=1, cache_file=cache)
    fresh = ss.run_study(pool2, later2, CHEAP, n_seeds=2, cut_year=2017, jobs=1)
    assert other.auroc == fresh.auroc
    assert other.auroc != first.auroc


def test_writing_a_subset_of_models_keeps_the_models_already_cached(tmp_path):
    cache = tmp_path / "fits.pkl"
    frame = with_drugs(400, 50)
    pool, later = ss.chronological_split(frame, 2016)
    ss.run_study(pool, later, CHEAP, n_seeds=1, cut_year=2016, jobs=1, cache_file=cache)
    ss.run_study(
        pool, later, (REGISTRATION_NAME,), n_seeds=1, cut_year=2016, jobs=1, cache_file=cache
    )
    import pickle

    kept = {name for _, name in pickle.loads(cache.read_bytes())}
    assert set(CHEAP) <= kept


def test_trials_with_no_lead_drug_are_each_their_own_drug_in_the_holdout_and_the_bootstrap():
    groups = np.array([None, np.nan, "A", "A", None, "B"], dtype=object)
    for seed in range(30):
        held = ss.drug_holdout(groups, 3, np.random.default_rng(seed))
        assert len(held) <= 3
    # many missing-drug trials can be split across the two sides, so a holdout of three can be
    # three missing-drug trials; one cluster of "missing" would have been all or nothing.
    many = np.array([None] * 20, dtype=object)
    assert len(ss.drug_holdout(many, 7, np.random.default_rng(0))) == 7
    run = ss.Run(
        np.array([True, False] * 10),
        many,
        {"m": np.linspace(0, 1, 20)},
    )
    draws = ss.bootstrap_draws([run], ["m"], np.random.default_rng(0))
    assert np.nanstd(draws[0, 0]) > 0  # singletons: resampling moves the score


def test_the_study_is_blind_to_the_holdout_outcomes():
    # Flipping every holdout label must change the scores, and shuffling nothing else must not:
    # the chronological predictions come from the pool alone.
    frame = with_drugs(400, 50)
    pool, later = ss.chronological_split(frame, 2016)
    base = ss.run_study(pool, later, CHEAP, n_seeds=2, cut_year=2016, jobs=1)
    flipped = later.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    again = ss.run_study(pool, flipped, CHEAP, n_seeds=2, cut_year=2016, jobs=1)
    assert (
        again.auroc[ss.CHRONO][REGISTRATION_NAME] < 0.5 < base.auroc[ss.CHRONO][REGISTRATION_NAME]
    )
    assert (
        abs(
            again.auroc[ss.CHRONO][REGISTRATION_NAME]
            - (1 - base.auroc[ss.CHRONO][REGISTRATION_NAME])
        )
        < 1e-9
    )


def test_report_has_one_table_and_four_bullets(tmp_path):
    frame = with_drugs(400, 50)
    pool, later = ss.chronological_split(frame, 2016)
    result = ss.run_study(pool, later, CHEAP, n_seeds=2, cut_year=2016, jobs=1)
    text = ss.write_report(result, 2016, tmp_path / "out.md")
    assert (tmp_path / "out.md").read_text() == text
    assert sum(line.startswith("- **") for line in text.splitlines()) == 4
    assert REGISTRATION_NAME in text
