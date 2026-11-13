"""The list of models the benchmark compares.

Every module named in `FAMILY_MODULES` defines `SPECS` (a list of `ModelSpec`).
`all_specs()` collects them and refuses duplicate names. To add a family, write
the module and add its name to the list.

A `ModelSpec` says how to fit a model, which extra feature tables it needs,
whether it is meant to be honest (it reads nothing that could have been
learned after the trial started) and whether it can be ranked against the others
on all trials.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import polars as pl
from sklearn.pipeline import Pipeline

from elman_exercise import baselines as bl
from elman_exercise.dataset import LABEL_COLUMNS
from elman_exercise.metrics import auroc


class Predictor(Protocol):
    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Probability of success for each row of `frame`."""
        ...


@dataclass(frozen=True)
class ModelSpec:
    name: str  # unique and readable, e.g. "logistic: registration"
    family: str  # e.g. "lookup", "linear", "trees"
    description: str  # one plain sentence: what it knows and how it predicts
    # training frame (features, extra tables joined, label `any_arm`) -> predictor
    fit: Callable[[pl.DataFrame], Predictor]
    tables: tuple[str, ...] = ()  # extra feature tables: data/interim/<name>.parquet
    # Tables the model reads itself (not one row per trial, so not joined to the frame). The
    # harness only checks that they exist and skips the model, by name, if they do not.
    own_tables: tuple[str, ...] = ()
    honest: bool = True  # False for deliberately leaky models
    pretrained: bool = False  # True if any pretrained model produced an input (shown in reports)
    # False for a model defined on only some trials (it predicts one constant for the rest), so
    # its score on all trials means nothing. Reports leave it out of the rankings and say why.
    ranked: bool = True
    not_ranked_reason: str = ""  # required when `ranked` is False

    def __post_init__(self):
        if not self.ranked and not self.not_ranked_reason:
            raise ValueError(f"Model {self.name!r} is not ranked but gives no reason.")


FAMILY_MODULES = (
    "lookup",
    "linear",
    "trees",
    "neural",
    "drift",
    "grouped",
    "memorisation",
    "pooling",
    "ladder",
    "text",
    "embedded",
    "biology",
    "llm",
    "precedent",
    "whathow",
    "hierarchical",
    "leaky",
)

# Families a plain `validate` run leaves out. "hierarchical": each model is a NUTS run of minutes
# and needs an optional install (`--extra hierarchical`). "whathow": its three link versions each
# carry a table with the same `wh_*` column names, which `validate`'s one-frame join refuses; it
# is scored by `whathow_report`. Ask for a family by name (`validate --only hierarchical`);
# `all_specs(include_opt_in=True)` lists them all.
OPT_IN_FAMILIES = ("hierarchical", "whathow")

# ---------------------------------------------------------------- shared helpers
# Small tools several families use: fit the benchmark's logistic pipeline (or the same
# encoding with another learner), and split a training table by time.

INNER_FRACTION = 0.75  # share of the training table, oldest first, used to fit when tuning


@dataclass
class PipelinePredictor:
    """A fitted scikit-learn pipeline, optionally preceded by a step that adds derived columns."""

    pipeline: Pipeline
    derive: Callable[[pl.DataFrame], pl.DataFrame] | None = None

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        if self.derive is not None:
            frame = self.derive(frame)
        return self.pipeline.predict_proba(frame)[:, 1]


class BaselinePredictor:
    """Adapts a model from baselines.py (which says `predict_probability`) to a Predictor."""

    def __init__(self, model):
        self.model = model

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.predict_probability(frame)


def features_from_frame(frame: pl.DataFrame, columns: list[str] | tuple[str, ...]) -> bl.Features:
    """Sort `columns` into text, yes/no and number by their type in `frame`.

    Used for tables the dataset schema does not describe. A column that holds
    nothing but nulls in `frame` is left out: there is nothing to learn from it.
    """
    kinds: dict[str, list[str]] = {"categorical": [], "boolean": [], "numeric": []}
    for name in columns:
        dtype = frame.schema[name]
        if frame[name].null_count() == frame.height:
            continue
        if dtype == pl.String:
            kinds["categorical"].append(name)
        elif dtype == pl.Boolean:
            kinds["boolean"].append(name)
        elif dtype.is_numeric():
            kinds["numeric"].append(name)
        else:
            raise ValueError(f"No way to model column {name!r} of type {dtype}.")
    return bl.Features(**{kind: tuple(names) for kind, names in kinds.items()})


def fit_pipeline(
    train: pl.DataFrame,
    features: bl.Features,
    estimator=None,
    derive: Callable[[pl.DataFrame], pl.DataFrame] | None = None,
    weights: np.ndarray | None = None,
) -> PipelinePredictor:
    """The benchmark's encoding of `features` followed by `estimator` (default: its logistic).

    With the default estimator this is `baselines.fit_logistic` with a dense matrix. `derive`
    adds columns (such as interactions) to `train` now and to every frame later.
    """
    outcomes = bl.training_outcomes(train)
    pipeline = bl.build_pipeline(features)
    # Always dense. By default the column encoder may return a sparse matrix, and the benchmark's
    # pipeline then crashes on a small training window in which a number column has no missing
    # value (its missing-value indicator has no columns). baselines.py is not changed for this.
    pipeline.set_params(encode__sparse_threshold=0.0)
    if estimator is not None:
        pipeline.set_params(fit=estimator)
    table = derive(train) if derive is not None else train
    fit_arguments = {} if weights is None else {"fit__sample_weight": weights}
    pipeline.fit(table, outcomes, **fit_arguments)
    return PipelinePredictor(pipeline, derive)


def inner_split(train: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Oldest `INNER_FRACTION` of the training table by completion date, then the newest rest.

    For choosing settings without touching anything outside the training table. The cut falls
    between dates: trials that share the boundary date all go to the newer part. Trials with no
    completion date cannot be placed in time, so they are left out of both parts.
    """
    dated = train.filter(pl.col("primary_completion_date").is_not_null()).sort(
        "primary_completion_date", maintain_order=True
    )
    if dated.is_empty():
        raise ValueError("No training trial has a completion date to cut at.")
    cut_date = dated["primary_completion_date"][int(dated.height * INNER_FRACTION)]
    older = dated.filter(pl.col("primary_completion_date") < cut_date)
    newer = dated.filter(pl.col("primary_completion_date") >= cut_date)
    if older.is_empty():
        raise ValueError("The training trials do not span enough dates to cut in two.")
    return older, newer


def best_by_inner_split(train: pl.DataFrame, candidates: list, fit_candidate: Callable):
    """The candidate setting that scores best (AUROC) on the newest part of the training table.

    `fit_candidate(table, candidate)` returns a Predictor. Only `train` is looked at. A
    candidate whose score is undefined (NaN) is never chosen; if all are, this raises.
    """
    fit_part, tune_part = inner_split(train)
    outcomes = bl.training_outcomes(tune_part)
    unlabelled = tune_part.drop(*LABEL_COLUMNS, strict=False)  # a model must not see the answer
    scores = np.array(
        [
            auroc(outcomes, fit_candidate(fit_part, candidate).predict(unlabelled))
            for candidate in candidates
        ],
        dtype=float,
    )
    if np.isnan(scores).all():
        raise ValueError("No candidate could be scored on the newest part of the training table.")
    return candidates[int(np.nanargmax(scores))]


def completion_years(frame: pl.DataFrame) -> np.ndarray:
    """Completion date as a fractional year (2016.5 is mid-2016). Training tables only."""
    dates = frame["primary_completion_date"]
    return (dates.dt.year() + (dates.dt.ordinal_day() - 0.5) / 366).to_numpy()


def require_unique_names(specs: list[ModelSpec]) -> None:
    names = [s.name for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise ValueError(f"duplicate model names: {duplicates}")


def all_specs(include_opt_in: bool = False) -> list[ModelSpec]:
    """Every spec from every module in `FAMILY_MODULES`, with names checked for duplicates.

    Families in `OPT_IN_FAMILIES` are left out (and never imported) unless `include_opt_in`.
    """
    specs: list[ModelSpec] = []
    for name in FAMILY_MODULES:
        if name in OPT_IN_FAMILIES and not include_opt_in:
            continue
        module = importlib.import_module(f"elman_exercise.models.{name}")
        specs.extend(module.SPECS)
    require_unique_names(specs)
    return specs
