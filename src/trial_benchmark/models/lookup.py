"""Lookup models: no learning beyond counting.

- constant: the training success rate for every trial.
- phase: the success rate within the trial's phase.
- phase x sponsor class: the rate within phase and sponsor class, pulled towards
  the phase rate when the cell is small.

"Pulled towards" is `shrunk_rate`: a cell with n trials and s successes, whose
parent group has rate r, gets (s + K * r) / (n + K). It is as if K extra trials at
the parent rate were added to every cell. A cell with no trials gets exactly r;
a big cell is close to its own rate. K is fixed at 10, not tuned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark.models.registry import BaselinePredictor, ModelSpec, Predictor

PRIOR_STRENGTH = 10.0  # K: how many pretend trials at the parent rate every cell starts with


def shrunk_rate(successes: float, trials: float, parent_rate: float, strength: float) -> float:
    """(successes + strength * parent_rate) / (trials + strength)."""
    return (successes + strength * parent_rate) / (trials + strength)


def shrunk_rates(
    keys: pl.Series, outcomes: np.ndarray, parent_rate: float, strength: float = PRIOR_STRENGTH
) -> dict:
    """Shrunk success rate for each distinct key. A null key is its own group, `None`."""
    table = (
        pl.DataFrame({"key": keys, "outcome": outcomes.astype(float)})
        .group_by("key")
        .agg(pl.col("outcome").sum().alias("successes"), pl.len().alias("trials"))
    )
    return {
        key: shrunk_rate(successes, trials, parent_rate, strength)
        for key, successes, trials in table.iter_rows()
    }


def look_up(keys: pl.Series, rates: dict, default: float | np.ndarray) -> np.ndarray:
    """The rate for each key; `default` (a number, or one per row) where the key was never seen."""
    defaults = np.broadcast_to(np.asarray(default, dtype=float), (len(keys),))
    return np.array([rates.get(key, d) for key, d in zip(keys, defaults, strict=True)])


@dataclass(frozen=True)
class PhaseSponsorModel:
    overall_rate: float
    phase_rates: dict  # phase -> rate, shrunk towards the overall rate
    cell_rates: dict  # (phase, sponsor class) -> rate, shrunk towards the phase rate

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        by_phase = look_up(frame["phase"], self.phase_rates, self.overall_rate)
        cells = zip(frame["phase"], frame["sponsor_class"], strict=True)
        return np.array(
            [
                self.cell_rates.get(cell, fallback)
                for cell, fallback in zip(cells, by_phase, strict=True)
            ]
        )


def fit_phase_sponsor(train: pl.DataFrame) -> PhaseSponsorModel:
    outcomes = bl.training_outcomes(train)
    overall = float(outcomes.mean())
    phase_rates = shrunk_rates(train["phase"], outcomes, overall)
    table = (
        train.select("phase", "sponsor_class")
        .with_columns(pl.Series("outcome", outcomes.astype(float)))
        .group_by("phase", "sponsor_class")
        .agg(pl.col("outcome").sum().alias("successes"), pl.len().alias("trials"))
    )
    cell_rates = {
        (phase, sponsor): shrunk_rate(
            successes, trials, phase_rates.get(phase, overall), PRIOR_STRENGTH
        )
        for phase, sponsor, successes, trials in table.iter_rows()
    }
    return PhaseSponsorModel(overall, phase_rates, cell_rates)


def _constant(train: pl.DataFrame) -> Predictor:
    return BaselinePredictor(bl.fit_constant(train))


def _phase(train: pl.DataFrame) -> Predictor:
    return BaselinePredictor(bl.fit_phase_lookup(train))


SPECS = [
    ModelSpec(
        "lookup: constant",
        "lookup",
        "Says the training success rate for every trial; knows nothing about the trial.",
        _constant,
    ),
    ModelSpec(
        "lookup: phase",
        "lookup",
        "Says the training success rate of the trial's phase.",
        _phase,
    ),
    ModelSpec(
        "lookup: phase x sponsor class",
        "lookup",
        "Says the training success rate of the trial's phase and sponsor class, "
        "pulled towards the phase rate when few trials share the cell.",
        fit_phase_sponsor,
    ),
]
