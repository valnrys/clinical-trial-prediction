"""Separate logistics for different kinds of trial.

One registration logistic per group, instead of one for everybody, so every
coefficient can differ by group:

- company vs other: trials with an industry sponsor, and all the rest.
- by phase: Phase 2, and Phase 3.

A group with fewer than MIN_TRIALS training trials, or with only one outcome, is
not given its own model; its trials use one logistic fitted on everything. Trials
whose group is unknown or was never seen in training do the same.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import polars as pl

from elman_exercise import baselines as bl
from elman_exercise.models.registry import ModelSpec, Predictor, fit_pipeline

MIN_TRIALS = 100
FEATURES = bl.FEATURES[bl.REGISTRATION]


def company_or_other(frame: pl.DataFrame) -> pl.Series:
    """ "company" for an industry-sponsored trial, "other" otherwise, null if the sponsor is unknown."""
    return frame.select(
        pl.when(pl.col("sponsor_class").is_null())
        .then(None)
        .otherwise(
            pl.when(pl.col("sponsor_class") == "INDUSTRY")
            .then(pl.lit("company"))
            .otherwise(pl.lit("other"))
        )
        .alias("group")
    )["group"]


def phase_group(frame: pl.DataFrame) -> pl.Series:
    return frame["phase"].alias("group")


@dataclass
class GroupedPredictor:
    group_of: Callable[[pl.DataFrame], pl.Series]
    by_group: dict[str, Predictor]
    fallback: Predictor

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        predictions = self.fallback.predict(frame)
        groups = self.group_of(frame)
        for group, model in self.by_group.items():
            rows = (groups == group).fill_null(False).to_numpy()
            if rows.any():
                predictions[rows] = model.predict(frame.filter(rows))
        return predictions


def fit_grouped(train: pl.DataFrame, group_of: Callable[[pl.DataFrame], pl.Series]) -> Predictor:
    groups = group_of(train)
    outcomes = bl.training_outcomes(train)
    by_group = {}
    for group in groups.drop_nulls().unique().sort():
        rows = (groups == group).fill_null(False).to_numpy()
        if rows.sum() >= MIN_TRIALS and 0 < outcomes[rows].sum() < rows.sum():
            by_group[group] = fit_pipeline(train.filter(rows), FEATURES)
    return GroupedPredictor(group_of, by_group, fit_pipeline(train, FEATURES))


SPECS = [
    ModelSpec(
        "logistic: separate for company and other sponsors",
        "grouped",
        "One registration logistic for industry-sponsored trials and another for all the rest.",
        lambda train: fit_grouped(train, company_or_other),
    ),
    ModelSpec(
        "logistic: separate by phase",
        "grouped",
        "One registration logistic for Phase 2 trials and another for Phase 3.",
        lambda train: fit_grouped(train, phase_group),
    ),
]
