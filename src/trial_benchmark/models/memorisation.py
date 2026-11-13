"""Models that only remember: how often did this sponsor, or this drug, succeed before?

Each model predicts from one name alone, using the training table only:

- sponsor memory: the success rate of the trial's lead sponsor among training
  trials that have a sponsor, shrunk towards their rate (see `lookup.shrunk_rates`). The sponsor
  name is today's name on the record, which may differ from the one at registration.
- drug memory: the same for the trial's lead drug (from genetics.parquet).

Whether each model is honest is worked out from how the column it reads is dated
(`sponsor.FEATURES`, `genetics.FEATURES`): the sponsor name is "protocol", the lead drug
is "after the start" (it comes from today's drug mapping), so drug memory is not honest.

A trial with no sponsor or drug, or one not in the training table, gets the overall
rate. These show how much of the signal is "the same programmes keep succeeding". In a
time-ordered check the training trials all finished before the scoring window, but some
finished after the trial being predicted started; that is true of every model in this
benchmark, which forecasts as of the split date, not as of each trial's start. In a check
that keeps each drug on one side, drug memory has nothing to remember.

The sponsor table (`sponsor_key`) is built by `trial_benchmark.sponsor`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import genetics, sponsor
from trial_benchmark.dating import HONEST
from trial_benchmark.models.lookup import PRIOR_STRENGTH, look_up, shrunk_rates
from trial_benchmark.models.registry import ModelSpec, Predictor


@dataclass(frozen=True)
class NameMemory:
    """Shrunk training success rate for each value of `column`; the overall rate if unseen."""

    column: str
    overall_rate: float
    rates: dict

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return look_up(frame[self.column], self.rates, self.overall_rate)


def fit_memory(train: pl.DataFrame, column: str) -> NameMemory:
    outcomes = bl.training_outcomes(train)
    overall = float(outcomes.mean())
    known = train[column].is_not_null().to_numpy()
    known_rate = float(outcomes[known].mean()) if known.any() else overall
    rates = shrunk_rates(train[column].filter(known), outcomes[known], known_rate, PRIOR_STRENGTH)
    return NameMemory(column, overall, rates)


def _sponsor_memory(train: pl.DataFrame) -> Predictor:
    return fit_memory(train, "sponsor_key")


def _drug_memory(train: pl.DataFrame) -> Predictor:
    return fit_memory(train, "lead_drug")


SPECS = [
    ModelSpec(
        "memory: lead sponsor",
        "memorisation",
        "Says the lead sponsor's success rate in the training trials (shrunk towards the rate of trials that have one), nothing else.",
        _sponsor_memory,
        tables=("sponsor",),
        honest=sponsor.FEATURES["sponsor_key"] in HONEST,
    ),
    ModelSpec(
        "memory: lead drug",
        "memorisation",
        "Says the lead drug's success rate in the training trials (shrunk towards the rate of trials that have one), nothing else.",
        _drug_memory,
        tables=("genetics",),
        honest=genetics.FEATURES["lead_drug"] in HONEST,
    ),
]
