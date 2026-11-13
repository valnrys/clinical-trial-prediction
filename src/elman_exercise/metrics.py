"""Scores for predicted probabilities of success, with bootstrap intervals.

All scores are for the headline label (a boolean per trial) against a predicted
probability of success per trial:

- AUROC: the chance a random successful trial gets a higher probability than a
  random failed one (ties count half). 0.5 is a coin toss.
- Brier score: the mean squared gap between probability and outcome. Lower is
  better; always guessing 50% scores 0.25.
- Accuracy: how often "probability of 0.5 or more" matches the outcome. Read it
  next to the majority-class accuracy: what you would score by always saying
  whatever was more common in the training data.
- Calibration gap: mean predicted probability minus observed success rate, in
  percentage points. Positive means the model is too optimistic.

Intervals are 95% percentile intervals from a bootstrap over trials: 1,000
resamples with replacement, at a fixed seed. By default the bootstrap treats
trials as independent. They are not (the same drugs and sponsors recur), so the
intervals are somewhat too narrow. Pass `clusters` (for example the lead drug)
to resample whole clusters instead.

Every score in one bootstrap call comes from the same resamples, so a paired
difference between two models is the difference within each resample, not the
difference of two unrelated intervals. A resample holding one class only has no
AUROC; it is skipped for AUROC (and for AUROC differences) and counted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

N_RESAMPLES = 1000
BOOTSTRAP_SEED = 20261110
THRESHOLD = 0.5
INTERVAL_PERCENTILES = (2.5, 97.5)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Ranks from 1, with tied values sharing the average of the ranks they span."""
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    last_rank = np.cumsum(counts)
    return (last_rank - (counts - 1) / 2)[inverse.ravel()]


def auroc(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Area under the ROC curve, from ranks. NaN if only one class is present."""
    outcomes = np.asarray(outcomes, dtype=bool)  # 0/1 integers would index, not mask
    positives = int(outcomes.sum())
    negatives = len(outcomes) - positives
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = _average_ranks(probabilities)
    return float(
        (ranks[outcomes].sum() - positives * (positives + 1) / 2) / (positives * negatives)
    )


def brier(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    return float(np.mean((probabilities - outcomes) ** 2))


def accuracy(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Share of trials where predicting success at THRESHOLD or above matches the outcome."""
    return float(np.mean((probabilities >= THRESHOLD) == outcomes))


def calibration_gap(outcomes: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean predicted probability minus observed success rate, in points. Positive: optimistic."""
    return float((probabilities.mean() - outcomes.mean()) * 100)


def majority_accuracy(outcomes: np.ndarray, majority_class: bool) -> float:
    """Accuracy of saying `majority_class` for every trial."""
    return float(np.mean(outcomes == majority_class))


def majority_class_of(training_outcomes: np.ndarray) -> bool:
    """The more common outcome in the training data (success, if the two are tied)."""
    return bool(training_outcomes.mean() >= 0.5)


@dataclass(frozen=True)
class Interval:
    low: float
    high: float


@dataclass(frozen=True)
class Scores:
    """Every score for one model on one set of trials, with bootstrap intervals."""

    n: int
    observed_rate: float
    mean_predicted: float
    calibration_gap: float  # points
    auroc: float
    brier: float
    accuracy: float
    majority_accuracy: float
    gap_interval: Interval
    auroc_interval: Interval
    accuracy_interval: Interval
    skipped_resamples: int  # resamples with one class only, left out of the AUROC interval


@dataclass(frozen=True)
class Difference:
    """AUROC and calibration gap of one model minus another, with intervals from shared resamples."""

    better: str
    baseline: str
    difference: float
    interval: Interval
    skipped_resamples: int
    gap_difference: float  # calibration gap of `better` minus that of `baseline`, in points
    gap_difference_interval: Interval


def _interval(values: np.ndarray) -> Interval:
    """Percentile interval of the non-NaN values; NaN if there are none."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return Interval(float("nan"), float("nan"))
    low, high = np.percentile(values, INTERVAL_PERCENTILES)
    return Interval(float(low), float(high))


def _is_missing(label: object) -> bool:
    """None, or a float NaN of any width (a missing label in an object or float array)."""
    return label is None or (isinstance(label, float | np.floating) and bool(np.isnan(label)))


def cluster_codes(clusters: np.ndarray, n: int) -> np.ndarray:
    """An integer code per trial: trials with the same label share a code, codes count up in
    order of first appearance.

    A missing label (None, or NaN of any float type) gets a code of its own for that trial, so
    trials with no known cluster are never lumped together.
    """
    if len(clusters) != n:
        raise ValueError(f"clusters has {len(clusters)} labels for {n} trials")
    seen: dict[object, int] = {}
    codes = np.empty(n, dtype=np.intp)
    for i, label in enumerate(clusters):
        key = ("missing", i) if _is_missing(label) else ("label", label)
        codes[i] = seen.setdefault(key, len(seen))
    return codes


def cluster_members(clusters: np.ndarray, n: int) -> list[np.ndarray]:
    """Trial positions of each cluster, in `cluster_codes` order."""
    codes = cluster_codes(clusters, n)
    order = np.argsort(codes, kind="stable")
    cuts = np.flatnonzero(np.diff(codes[order])) + 1
    return [np.asarray(part, dtype=np.intp) for part in np.split(order, cuts)] if n else []


def bootstrap(
    outcomes: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    majority_class: bool,
    pairs: Sequence[tuple[str, str]] = (),
    n_resamples: int = N_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    clusters: np.ndarray | None = None,
) -> tuple[dict[str, Scores], list[Difference]]:
    """Scores for each named model, and AUROC differences for each (better, baseline) pair.

    All models and pairs share the same resamples of the trials. The same seed
    gives the same numbers.

    With `clusters` (one label per trial, for example the lead drug), each resample
    draws whole clusters with replacement, as many as there are clusters, and takes
    all their trials. A trial with a missing label (None or NaN) is its own cluster.
    """
    outcomes = np.asarray(outcomes, dtype=bool)
    n = len(outcomes)
    rng = np.random.default_rng(seed)
    members = None if clusters is None else cluster_members(clusters, n)
    names = list(predictions)
    aurocs = {name: np.full(n_resamples, np.nan) for name in names}
    accuracies = {name: np.empty(n_resamples) for name in names}
    gaps = {name: np.empty(n_resamples) for name in names}
    for i in range(n_resamples):
        if members is None:
            chosen = rng.integers(0, n, size=n)
        else:
            picks = rng.integers(0, len(members), size=len(members))
            chosen = np.concatenate([members[k] for k in picks])
        resampled = outcomes[chosen]
        for name in names:
            probabilities = predictions[name][chosen]
            aurocs[name][i] = auroc(resampled, probabilities)
            accuracies[name][i] = accuracy(resampled, probabilities)
            gaps[name][i] = calibration_gap(resampled, probabilities)

    scores = {}
    for name in names:
        probabilities = predictions[name]
        scores[name] = Scores(
            n=n,
            observed_rate=float(outcomes.mean()),
            mean_predicted=float(probabilities.mean()),
            calibration_gap=calibration_gap(outcomes, probabilities),
            auroc=auroc(outcomes, probabilities),
            brier=brier(outcomes, probabilities),
            accuracy=accuracy(outcomes, probabilities),
            majority_accuracy=majority_accuracy(outcomes, majority_class),
            gap_interval=_interval(gaps[name]),
            auroc_interval=_interval(aurocs[name]),
            accuracy_interval=_interval(accuracies[name]),
            skipped_resamples=int(np.isnan(aurocs[name]).sum()),
        )
    differences = []
    for better, baseline in pairs:
        paired = aurocs[better] - aurocs[baseline]
        gap_difference = scores[better].calibration_gap - scores[baseline].calibration_gap
        differences.append(
            Difference(
                better=better,
                baseline=baseline,
                difference=scores[better].auroc - scores[baseline].auroc,
                interval=_interval(paired),
                skipped_resamples=int(np.isnan(paired).sum()),
                gap_difference=gap_difference,
                gap_difference_interval=_interval(gaps[better] - gaps[baseline]),
            )
        )
    return scores, differences
