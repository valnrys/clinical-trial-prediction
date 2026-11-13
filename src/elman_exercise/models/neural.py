"""A small neural network on the registration columns.

One scikit-learn MLPClassifier: two hidden layers (32 and 16 units) and a modest
penalty. It reads the benchmark's own encoding of the registration columns, which
already scales the numbers, so there is no second scaler. The 0/1 missing-value
columns are left as they are, as in the logistic.

Early stopping has no say in how long it trains. The number of iterations is chosen
from a short list on the registry's time-ordered inner split (oldest 75% of the
training table fits, newest 25% scores), then the network is refitted on all of the
training table. Nothing is looked at outside the training table.
"""

from __future__ import annotations

import warnings

import polars as pl
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

from elman_exercise import baselines as bl
from elman_exercise.models.registry import ModelSpec, Predictor, best_by_inner_split, fit_pipeline

HIDDEN_LAYERS = (32, 16)
ITERATIONS = (20, 50, 100, 200)  # candidate numbers of passes over the training table
FEATURES = bl.FEATURES[bl.REGISTRATION]


def network(iterations: int) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        alpha=1e-2,
        early_stopping=False,
        max_iter=iterations,
        random_state=bl.RANDOM_STATE,
    )


def fit_with(train: pl.DataFrame, iterations: int) -> Predictor:
    with warnings.catch_warnings():
        # Stopping at the cap is the point of the cap; scikit-learn warns about it every time.
        warnings.simplefilter("ignore", ConvergenceWarning)
        return fit_pipeline(train, FEATURES, network(iterations))


def fit_network(train: pl.DataFrame) -> Predictor:
    best = best_by_inner_split(train, list(ITERATIONS), fit_with)
    return fit_with(train, best)


SPECS = [
    ModelSpec(
        "neural network: registration",
        "neural",
        "A small two-layer neural network on the registration columns; training length chosen on an inner split.",
        fit_network,
    ),
]
