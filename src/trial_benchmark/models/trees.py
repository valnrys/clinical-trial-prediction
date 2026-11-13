"""Tree models on the registration columns.

Same inputs as the registration logistic, different learner: they can find
interactions and thresholds the logistic cannot.

- boosted trees: scikit-learn's HistGradientBoostingClassifier, shallow trees,
  small steps, fixed settings.
- random forest: 500 trees, fixed settings.
- boosted trees, tuned: the boosted model with depth, step size and number of
  rounds chosen on an inner split of the training table (oldest 75% fit, newest
  25% score), then refit on the whole training table.
"""

from __future__ import annotations

from itertools import product

import polars as pl
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier

from trial_benchmark import baselines as bl
from trial_benchmark.models.registry import (
    ModelSpec,
    Predictor,
    best_by_inner_split,
    fit_pipeline,
)

DEPTHS = (2, 4)
STEP_SIZES = (0.03, 0.1)
ROUNDS = (100, 300)


def boosted(
    depth: int = 3, step: float = 0.05, rounds: int = 200
) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(
        max_depth=depth,
        learning_rate=step,
        max_iter=rounds,
        min_samples_leaf=40,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=bl.RANDOM_STATE,
    )


def fit_boosted(train: pl.DataFrame) -> Predictor:
    return fit_pipeline(train, bl.FEATURES[bl.REGISTRATION], boosted())


def fit_forest(train: pl.DataFrame) -> Predictor:
    forest = RandomForestClassifier(
        n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=bl.RANDOM_STATE
    )
    return fit_pipeline(train, bl.FEATURES[bl.REGISTRATION], forest)


def fit_boosted_tuned(train: pl.DataFrame) -> Predictor:
    features = bl.FEATURES[bl.REGISTRATION]
    best = best_by_inner_split(
        train,
        list(product(DEPTHS, STEP_SIZES, ROUNDS)),
        lambda table, setting: fit_pipeline(table, features, boosted(*setting)),
    )
    return fit_pipeline(train, features, boosted(*best))


SPECS = [
    ModelSpec(
        "boosted trees: registration",
        "trees",
        "Gradient-boosted shallow trees on the registration columns, fixed settings.",
        fit_boosted,
    ),
    ModelSpec(
        "random forest: registration",
        "trees",
        "A random forest of 500 trees on the registration columns, fixed settings.",
        fit_forest,
    ),
    ModelSpec(
        "boosted trees: registration, tuned",
        "trees",
        "Boosted trees on the registration columns, depth and step size chosen on an inner split.",
        fit_boosted_tuned,
    ),
]
