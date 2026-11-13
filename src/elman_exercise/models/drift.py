"""Models that allow for the success rate changing over time.

Success rates drift. These try three ways of letting the training table speak
more for the recent past. None of them looks at the completion date of the trial
being predicted: they read only the completion years of the TRAINING trials, and
project from those to "the period just after training". That is the one thing a
forecaster standing at the cut could do.

- last N years: the registration logistic fitted on only the newest N years of
  the training table (N = 4 and 8).
- recency-weighted: the registration logistic with each training trial weighted by
  0.5 ** (age / 4 years), age measured back from the newest training trial. The
  weights are scaled to average 1, so the penalty keeps its usual strength.
- trend-shifted: the registration logistic with its intercept moved so that its
  average prediction on the training trials equals the success rate projected
  forward by a straight line through the yearly success rates of the last
  TREND_YEARS complete calendar years. The line is fitted with each year counted
  by its number of trials. The shift is found by bisection, because moving every
  log-odds by one amount does not move the average by the same amount.

What the trend-shifted model assumes. The look-ahead is a parameter of the model,
`look_ahead_years`: how far past the END of the training years the projection
looks. The default, 1.0, is the middle of a two-year scoring window that starts
right after training. A forecaster who scores a different window should change it.
The yearly rates are placed at mid-year, and the training years end at the close
of the newest complete year.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from elman_exercise import baselines as bl
from elman_exercise.models.registry import (
    ModelSpec,
    Predictor,
    completion_years,
    fit_pipeline,
)

FEATURES = bl.FEATURES[bl.REGISTRATION]
HALF_LIFE_YEARS = 4.0
TREND_YEARS = 8
LOOK_AHEAD_YEARS = 1.0  # years past the end of training: the middle of a two-year window
BISECTION_STEPS = 60
SHIFT_RANGE = 20.0  # the shift is searched between minus and plus this on the log-odds scale
NEWEST_YEAR_COMPLETE_FROM_MONTH = 12  # the newest year counts only if the table reaches December
RATE_RANGE = (0.05, 0.95)  # a projected rate is kept inside this, so the logit stays finite


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def newest_years(train: pl.DataFrame, n_years: int) -> pl.DataFrame:
    """The trials completing in the last `n_years` calendar years of the table."""
    years = train["primary_completion_date"].dt.year()
    return train.filter(years > years.max() - n_years)


def fit_last_years(train: pl.DataFrame, n_years: int) -> Predictor:
    return fit_pipeline(newest_years(train, n_years), FEATURES)


def recency_weights(years: np.ndarray, half_life: float = HALF_LIFE_YEARS) -> np.ndarray:
    """Halving every `half_life` years further back, scaled to average 1 over the trials."""
    weights = 0.5 ** ((years.max() - years) / half_life)
    return weights / weights.mean()


def fit_recency_weighted(train: pl.DataFrame) -> Predictor:
    return fit_pipeline(train, FEATURES, weights=recency_weights(completion_years(train)))


def complete_years(train: pl.DataFrame) -> np.ndarray:
    """Calendar years of the training trials, without the newest year if the table stops short of it."""
    dates = train["primary_completion_date"]
    years = dates.dt.year().to_numpy()
    newest = years.max()
    if dates.max().month < NEWEST_YEAR_COMPLETE_FROM_MONTH:
        return years[years < newest]
    return years


def projected_rate(train: pl.DataFrame, look_ahead_years: float) -> float:
    """Success rate that a trend through the recent yearly rates gives `look_ahead_years` past training.

    The line is fitted over the last TREND_YEARS complete calendar years, each year counted
    by its number of trials. With fewer than two such years the overall rate is returned.
    """
    outcomes = bl.training_outcomes(train)
    years = train["primary_completion_date"].dt.year().to_numpy()
    whole = np.unique(complete_years(train))
    whole = whole[whole > whole.max() - TREND_YEARS] if len(whole) else whole
    if len(whole) < 2:
        return float(outcomes.mean())
    counts = np.array([(years == y).sum() for y in whole])
    rates = np.array([outcomes[years == y].mean() for y in whole])
    slope, intercept = np.polyfit(whole + 0.5, rates, deg=1, w=np.sqrt(counts))  # mid-year
    end_of_training = whole.max() + 1
    rate = intercept + slope * (end_of_training + look_ahead_years)
    return float(np.clip(rate, *RATE_RANGE))


def shift_to_mean(predictions: np.ndarray, target: float) -> float:
    """The log-odds shift that makes the average of the shifted predictions equal `target`."""
    odds = logit(np.clip(predictions, 1e-6, 1 - 1e-6))
    low, high = -SHIFT_RANGE, SHIFT_RANGE
    for _ in range(BISECTION_STEPS):
        middle = (low + high) / 2
        if expit(odds + middle).mean() < target:
            low = middle
        else:
            high = middle
    return (low + high) / 2


@dataclass
class ShiftedPredictor:
    """Wraps a predictor, adding `shift` to every prediction on the log-odds scale."""

    inner: Predictor
    shift: float

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        odds = logit(np.clip(self.inner.predict(frame), 1e-6, 1 - 1e-6)) + self.shift
        return expit(odds)


def fit_trend_shifted(train: pl.DataFrame, look_ahead_years: float = LOOK_AHEAD_YEARS) -> Predictor:
    model = fit_pipeline(train, FEATURES)
    target = projected_rate(train, look_ahead_years)
    return ShiftedPredictor(model, shift_to_mean(model.predict(train), target))


def _last_years(n_years: int):
    return lambda train: fit_last_years(train, n_years)


SPECS = [
    ModelSpec(
        f"logistic: last {n} years only",
        "drift",
        f"The registration logistic fitted on only the newest {n} years of training trials.",
        _last_years(n),
    )
    for n in (4, 8)
] + [
    ModelSpec(
        "logistic: recency-weighted",
        "drift",
        "The registration logistic with newer training trials counting more (half-life 4 years).",
        fit_recency_weighted,
    ),
    ModelSpec(
        "logistic: trend-shifted intercept",
        "drift",
        "The registration logistic, shifted to the success rate projected from the yearly trend.",
        fit_trend_shifted,
    ),
]
