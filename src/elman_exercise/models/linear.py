"""Logistic models on the registration columns.

- "logistic: registration" is the pre-registered baseline, unchanged: it is
  `baselines.fit_logistic` on `baselines.FEATURES["registration"]`.
- "logistic: registration + interactions" adds phase x sponsor class and
  phase x therapeutic area, so the effect of one can differ by the other.
- "logistic: registration, tuned penalty" chooses the penalty strength and its
  mix of L1 and L2 on an inner split of the training table (oldest 75% fit,
  newest 25% score), then refits on the whole training table.
"""

from __future__ import annotations

from dataclasses import replace

import polars as pl
from sklearn.linear_model import LogisticRegression

from elman_exercise import baselines as bl
from elman_exercise.models.registry import (
    BaselinePredictor,
    ModelSpec,
    Predictor,
    best_by_inner_split,
    fit_pipeline,
)

REGISTRATION_NAME = "logistic: registration"
INTERACTIONS = (("phase", "sponsor_class"), ("phase", "therapeutic_area"))
PENALTY_STRENGTHS = (0.03, 0.1, 0.3, 1.0)  # C: larger means a weaker penalty
L1_SHARES = (0.0, 0.5, 1.0)  # 0 is pure L2 (ridge), 1 is pure L1 (lasso)


def fit_registration(train: pl.DataFrame) -> Predictor:
    return BaselinePredictor(bl.fit_logistic(train, bl.FEATURES[bl.REGISTRATION]))


def interaction_name(first: str, second: str) -> str:
    return f"{first} x {second}"


def add_interactions(frame: pl.DataFrame) -> pl.DataFrame:
    """One text column per pair: the two values joined. Null if either is null."""
    return frame.with_columns(
        pl.concat_str(first, second, separator=" x ").alias(interaction_name(first, second))
        for first, second in INTERACTIONS
    )


def fit_interactions(train: pl.DataFrame) -> Predictor:
    base = bl.FEATURES[bl.REGISTRATION]
    extra = tuple(interaction_name(*pair) for pair in INTERACTIONS)
    features = replace(base, categorical=base.categorical + extra)
    return fit_pipeline(train, features, derive=add_interactions)


def _penalised(strength: float, l1_share: float) -> LogisticRegression:
    return LogisticRegression(
        C=strength,
        l1_ratio=l1_share,
        solver="saga",
        max_iter=bl.MAX_ITERATIONS,
        random_state=bl.RANDOM_STATE,
    )


def fit_tuned(train: pl.DataFrame) -> Predictor:
    features = bl.FEATURES[bl.REGISTRATION]
    candidates = [(c, share) for c in PENALTY_STRENGTHS for share in L1_SHARES]
    strength, share = best_by_inner_split(
        train,
        candidates,
        lambda table, setting: fit_pipeline(table, features, _penalised(*setting)),
    )
    return fit_pipeline(train, features, _penalised(strength, share))


SPECS = [
    ModelSpec(
        REGISTRATION_NAME,
        "linear",
        "The pre-registered baseline: logistic regression on what is known at registration.",
        fit_registration,
    ),
    ModelSpec(
        "logistic: registration + interactions",
        "linear",
        "The registration logistic plus phase-by-sponsor and phase-by-area terms.",
        fit_interactions,
    ),
    ModelSpec(
        "logistic: registration, tuned penalty",
        "linear",
        "The registration logistic with penalty strength and L1/L2 mix chosen on an inner split.",
        fit_tuned,
    ),
]
