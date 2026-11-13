"""Partial pooling: borrow strength for sponsors.

The registration logistic plus one extra number per trial: how far the lead
sponsor's success rate sits from the overall rate. The rate is shrunk towards the
overall rate (see `lookup.shrunk_rate`), so a sponsor with two trials barely moves
the prediction and one with two hundred moves it a lot. That is partial pooling
done by hand. (The therapeutic area is not pooled this way: the registration
columns already carry it, one column per area.)

The number must not be computed from the trial it describes, or the model would
learn from its own label. And it should mean the same thing in training as in use.
When the model predicts a trial, the sponsor's rate comes from every training
trial, all of which finished earlier. So in training, each trial's number comes
only from training trials that finished in an EARLIER calendar year than it did.
The first year has no earlier trials and gets 0 (no information). The sponsor
name is today's name on the record (see `trial_benchmark.sponsor`).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark.models.lookup import look_up, shrunk_rates
from trial_benchmark.models.registry import ModelSpec, PipelinePredictor, Predictor, fit_pipeline

SOURCE = "sponsor_key"
EFFECT = "sponsor_effect"  # the new column


def sponsor_rates(train: pl.DataFrame, outcomes: np.ndarray) -> tuple[dict, float]:
    """The shrunk success rate of every sponsor in `train`, and the overall rate."""
    overall = float(outcomes.mean())
    known = train[SOURCE].is_not_null().to_numpy()
    return shrunk_rates(train[SOURCE].filter(known), outcomes[known], overall), overall


def add_effect(frame: pl.DataFrame, rates: dict, overall: float) -> pl.DataFrame:
    """Add the effect column: the sponsor's shrunk rate minus the overall rate (0 if unseen)."""
    return frame.with_columns(pl.Series(EFFECT, look_up(frame[SOURCE], rates, overall) - overall))


def add_earlier_only_effect(train: pl.DataFrame, outcomes: np.ndarray) -> pl.DataFrame:
    """Like `add_effect`, but each row's rate comes only from trials completed in earlier years."""
    years = train["primary_completion_date"].dt.year().to_numpy()
    effect = np.zeros(train.height)
    for year in np.unique(years):
        now, earlier = years == year, years < year
        if earlier.any():
            rates, overall = sponsor_rates(train.filter(earlier), outcomes[earlier])
            effect[now] = look_up(train.filter(now)[SOURCE], rates, overall) - overall
    return train.with_columns(pl.Series(EFFECT, effect))


@dataclass
class PooledPredictor:
    inner: PipelinePredictor
    rates: dict
    overall: float

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.inner.predict(add_effect(frame, self.rates, self.overall))


def fit_pooled(train: pl.DataFrame) -> Predictor:
    outcomes = bl.training_outcomes(train)
    base = bl.FEATURES[bl.REGISTRATION]
    features = replace(base, numeric=(*base.numeric, EFFECT))
    inner = fit_pipeline(add_earlier_only_effect(train, outcomes), features)
    return PooledPredictor(inner, *sponsor_rates(train, outcomes))


SPECS = [
    ModelSpec(
        "logistic: registration + pooled sponsor",
        "pooling",
        "The registration logistic plus the lead sponsor's shrunk success rate in trials that finished in earlier years.",
        fit_pooled,
        tables=("sponsor",),
    ),
]
