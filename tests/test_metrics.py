import numpy as np
import pytest

from trial_benchmark import metrics as mt

# Four trials: the two failures are predicted 0.1 and 0.4, the two successes 0.35 and 0.8.
OUTCOMES = np.array([False, False, True, True])
PROBABILITIES = np.array([0.1, 0.4, 0.35, 0.8])


def test_auroc_on_four_trials_is_three_quarters_by_hand():
    # Of the four success-failure pairs, the model ranks three the right way round.
    assert mt.auroc(OUTCOMES, PROBABILITIES) == pytest.approx(0.75)


def test_auroc_counts_ties_as_half():
    assert mt.auroc(OUTCOMES, np.full(4, 0.58)) == pytest.approx(0.5)


def test_auroc_is_one_for_a_perfect_ranking_and_zero_for_a_reversed_one():
    assert mt.auroc(OUTCOMES, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert mt.auroc(OUTCOMES, np.array([0.9, 0.8, 0.2, 0.1])) == 0.0


def test_auroc_treats_integer_outcomes_as_classes_not_positions():
    assert mt.auroc(np.array([0, 1]), np.array([0.2, 0.8])) == 1.0
    assert mt.auroc(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.2, 0.1])) == pytest.approx(0.75)


def test_auroc_matches_scikit_learn_with_many_ties():
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(0)
    outcomes = rng.random(500) < 0.4
    probabilities = np.round(rng.random(500) * 0.5 + 0.3 * outcomes, 1)  # heavy ties
    assert mt.auroc(outcomes, probabilities) == pytest.approx(
        roc_auc_score(outcomes, probabilities)
    )


def test_the_bootstrap_gives_the_same_scores_for_integer_and_boolean_outcomes():
    outcomes, predictions = noisy_data()
    as_bool, _ = mt.bootstrap(outcomes, predictions, True, n_resamples=30)
    as_int, _ = mt.bootstrap(outcomes.astype(int), predictions, True, n_resamples=30)
    assert as_bool == as_int


def test_auroc_is_undefined_with_one_class():
    assert np.isnan(mt.auroc(np.array([True, True]), np.array([0.2, 0.9])))


def test_brier_score_by_hand():
    outcomes = np.array([True, False])
    assert mt.brier(outcomes, np.array([0.8, 0.4])) == pytest.approx((0.04 + 0.16) / 2)


def test_calibration_gap_is_positive_when_the_model_is_too_optimistic():
    outcomes = np.array([True, False, False, False])  # observed 25%
    assert mt.calibration_gap(outcomes, np.full(4, 0.35)) == pytest.approx(10.0)
    assert mt.calibration_gap(outcomes, np.full(4, 0.15)) == pytest.approx(-10.0)


def test_accuracy_counts_exactly_half_as_a_prediction_of_success():
    assert mt.accuracy(np.array([True, False]), np.array([0.5, 0.49])) == 1.0


def test_majority_accuracy_is_the_share_of_the_majority_class():
    outcomes = np.array([True, True, True, False])
    assert mt.majority_accuracy(outcomes, True) == 0.75
    assert mt.majority_accuracy(outcomes, False) == 0.25


def test_the_majority_class_comes_from_the_training_data_and_ties_go_to_success():
    assert mt.majority_class_of(np.array([True, False, False])) is False
    assert mt.majority_class_of(np.array([True, False])) is True


def noisy_data(seed=1, n=200):
    rng = np.random.default_rng(seed)
    outcomes = rng.random(n) < 0.5
    first = np.clip(0.5 * outcomes + rng.random(n) * 0.6, 0, 1)
    second = np.clip(0.3 * outcomes + rng.random(n) * 0.7, 0, 1)
    return outcomes, {"first": first, "second": second}


def test_the_bootstrap_gives_the_same_numbers_for_the_same_seed():
    outcomes, predictions = noisy_data()
    one, _ = mt.bootstrap(outcomes, predictions, True, n_resamples=100, seed=5)
    two, _ = mt.bootstrap(outcomes, predictions, True, n_resamples=100, seed=5)
    other, _ = mt.bootstrap(outcomes, predictions, True, n_resamples=100, seed=6)
    assert one == two
    assert one["first"].auroc_interval != other["first"].auroc_interval


def test_the_interval_surrounds_the_point_estimate():
    outcomes, predictions = noisy_data()
    scores, _ = mt.bootstrap(outcomes, predictions, True, n_resamples=300)
    first = scores["first"]
    assert first.auroc_interval.low < first.auroc < first.auroc_interval.high
    assert first.n == 200
    assert first.observed_rate == outcomes.mean()


def test_a_paired_difference_is_taken_within_each_resample():
    outcomes, predictions = noisy_data()
    pair = [("first", "second")]
    _, (difference,) = mt.bootstrap(outcomes, predictions, True, pair, n_resamples=50, seed=3)
    # Redo it by hand with the same random numbers: one resample, both models, then subtract.
    rng = np.random.default_rng(3)
    by_hand = []
    for _ in range(50):
        chosen = rng.integers(0, len(outcomes), size=len(outcomes))
        by_hand.append(
            mt.auroc(outcomes[chosen], predictions["first"][chosen])
            - mt.auroc(outcomes[chosen], predictions["second"][chosen])
        )
    low, high = np.percentile(by_hand, mt.INTERVAL_PERCENTILES)
    assert (difference.interval.low, difference.interval.high) == pytest.approx((low, high))
    assert difference.difference == pytest.approx(
        mt.auroc(outcomes, predictions["first"]) - mt.auroc(outcomes, predictions["second"])
    )


def test_two_identical_models_differ_by_exactly_zero_in_every_resample():
    outcomes, predictions = noisy_data()
    same = {"a": predictions["first"], "b": predictions["first"]}
    _, (difference,) = mt.bootstrap(outcomes, same, True, [("a", "b")], n_resamples=100)
    assert difference.interval == mt.Interval(0.0, 0.0)


def test_resamples_with_one_class_are_skipped_for_auroc_and_counted():
    outcomes = np.array([True] + [False] * 5)  # a resample often misses the one success
    predictions = {"m": np.array([0.9, 0.1, 0.2, 0.3, 0.4, 0.5])}
    scores, _ = mt.bootstrap(outcomes, predictions, False, n_resamples=200)
    assert scores["m"].skipped_resamples > 0
    assert not np.isnan(scores["m"].auroc_interval.low)
    assert not np.isnan(scores["m"].accuracy_interval.low)  # accuracy is not skipped


# ---- clustered bootstrap

_Y = np.array([0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1], dtype=bool)  # fmt: skip
_P = np.array([0.32, 0.43, 0.6, 0.13, 0.72, 0.25, 0.26, 0.38, 0.07, 0.71, 0.34, 0.39, 0.4, 0.4, 0.4, 0.04, 0.73, 0.43, 0.74, 0.54, 0.14, 0.3, 0.32, 0.36, 0.42, 0.14, 0.11, 0.62, 0.7, 0.48, 0.08, 0.54, 0.45, 0.21, 0.29, 0.01, 0.64, 0.46, 0.41, 0.74])  # fmt: skip


def _run(outcomes, p, **kwargs):
    preds = {"a": p, "b": np.full(len(p), 0.4)}
    return mt.bootstrap(outcomes, preds, True, [("a", "b")], n_resamples=50, seed=7, **kwargs)


def test_the_default_bootstrap_is_unchanged_by_the_cluster_option():
    # Hard-coded from the code before `clusters` existed, same data and seed.
    scores, (diff,) = _run(_Y, _P)
    assert scores["a"].auroc_interval.low == pytest.approx(0.851118170622481)
    assert scores["a"].auroc_interval.high == pytest.approx(0.9957999999999999)
    assert scores["a"].gap_interval.low == pytest.approx(-4.524374999999995)
    assert scores["a"].gap_interval.high == pytest.approx(15.340624999999996)
    assert scores["a"].accuracy_interval.low == pytest.approx(0.85)
    assert scores["a"].accuracy_interval.high == pytest.approx(0.994375)
    assert diff.interval.low == pytest.approx(0.3511181706224809)
    assert diff.gap_difference_interval.high == pytest.approx(5.517499999999997)


def test_one_trial_per_cluster_gives_exactly_the_unclustered_result():
    # Clusters are ordered by first appearance, so cluster k is trial k and the same
    # random draws pick the same trials.
    labels = np.array([f"drug{i}" for i in range(len(_Y))], dtype=object)
    assert _run(_Y, _P, clusters=labels) == _run(_Y, _P)


def test_clustering_by_drug_is_wider_when_each_trial_is_repeated_within_its_cluster():
    outcomes, probabilities = np.repeat(_Y, 10), np.repeat(_P, 10)
    labels = np.repeat(np.arange(len(_Y)).astype(str), 10).astype(object)
    naive, _ = _run(outcomes, probabilities)
    clustered, _ = _run(outcomes, probabilities, clusters=labels)
    width = lambda i: i.high - i.low
    assert width(clustered["a"].auroc_interval) > 1.5 * width(naive["a"].auroc_interval)
    assert width(clustered["a"].gap_interval) > 1.5 * width(naive["a"].gap_interval)


def test_each_missing_cluster_label_is_its_own_cluster():
    labels = np.array(["x", None, np.nan, "x", None], dtype=object)
    members = mt.cluster_members(labels, 5)
    assert sorted(m.tolist() for m in members) == [[0, 3], [1], [2], [4]]
    all_missing = np.array([None] * len(_Y), dtype=object)
    assert _run(_Y, _P, clusters=all_missing) == _run(_Y, _P)


def test_clustered_resamples_with_one_class_are_skipped_and_counted():
    outcomes = np.array([True, True, False, False])
    labels = np.array(["a", "a", "b", "b"], dtype=object)  # a cluster is a whole class
    scores, _ = mt.bootstrap(
        outcomes, {"m": np.array([0.9, 0.8, 0.2, 0.1])}, True, n_resamples=200, clusters=labels
    )
    assert scores["m"].skipped_resamples > 0
    assert scores["m"].auroc_interval.low == 1.0


def test_cluster_labels_must_match_the_trials():
    with pytest.raises(ValueError):
        mt.bootstrap(_Y, {"a": _P}, True, n_resamples=3, clusters=np.array(["x"], dtype=object))


def test_float_nan_of_any_width_is_a_missing_label():
    for dtype in (np.float32, np.float64):
        labels = np.array([1.0, np.nan, np.nan, 1.0], dtype=dtype)
        members = mt.cluster_members(labels, 4)
        assert sorted(m.tolist() for m in members) == [[0, 3], [1], [2]]
