"""Plain predictors that show what the benchmark reads.

The benchmark is the ruler; these are deliberately simple things to measure with
it. Each rung is a function from a training table to a fitted model, and a
fitted model has `predict_probability(table)`: the chance, per trial, that the
headline label (`any_arm`) is true.

The ladder, and what each rung is for:

- constant: the training success rate for every trial. What "knowing how often
  trials succeed" is worth. Its AUROC is 0.5 by construction.
- phase lookup: the training success rate for the trial's phase.
- registration: logistic regression on REGISTRATION_COLUMNS: the `design`
  columns (things known when a trial is registered), less any in
  REGISTRATION_EXCLUDED. Every rung below, and the drop-one-column table in
  evaluate.py, builds on that one list, so excluding a column is a one-line change.
- registration, no modality: the same without `modality` and `is_vaccine`. Tests
  the prediction that knowing the modality adds little.
- leaky: registration plus the `post_hoc` columns. Deliberately wrong: it shows
  what a model scores when it can read fields edited after the trial ended.
- results structure: registration plus `control_kind` and `n_arms`. Also
  deliberately wrong, more subtly: those come from the posted results.

No tuning, no feature search. Logistic regression is scikit-learn's default
(L2, default strength), with a fixed seed.

A model only ever sees the columns in its `Features`, whatever else the table
holds, so passing it a table with post-hoc or label columns is safe.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from trial_benchmark.dataset import SCHEMA, columns_in_group

LABEL = "any_arm"
MISSING = "(missing)"
LOG_COLUMNS = ("enrollment",)  # counts with a long tail: modelled as log(1 + count)
MAX_ITERATIONS = 1000
RANDOM_STATE = 0

CONSTANT = "constant"
PHASE_LOOKUP = "phase lookup"
REGISTRATION = "registration"
NO_MODALITY = "registration, no modality"
LEAKY = "leaky"
RESULTS_STRUCTURE = "results structure"


def training_outcomes(table: pl.DataFrame) -> np.ndarray:
    """The label as a boolean array. Refuses a table with undetermined labels."""
    labels = table[LABEL]
    if labels.null_count() > 0:
        raise ValueError(f"Cannot train: {labels.null_count()} trials have no {LABEL} label.")
    if len(labels) == 0:
        raise ValueError("Cannot train on an empty table.")
    return labels.to_numpy().astype(bool)


# ---------------------------------------------------------------- rungs 0 and 1


@dataclass(frozen=True)
class ConstantModel:
    rate: float

    def predict_probability(self, table: pl.DataFrame) -> np.ndarray:
        return np.full(table.height, self.rate)


def fit_constant(train: pl.DataFrame) -> ConstantModel:
    return ConstantModel(rate=float(training_outcomes(train).mean()))


@dataclass(frozen=True)
class PhaseLookupModel:
    """The training success rate within each phase. A phase never seen gets the overall rate."""

    rate_by_phase: dict[str | None, float]
    overall_rate: float

    def predict_probability(self, table: pl.DataFrame) -> np.ndarray:
        return np.array([self.rate_by_phase.get(p, self.overall_rate) for p in table["phase"]])


def fit_phase_lookup(train: pl.DataFrame) -> PhaseLookupModel:
    outcomes = training_outcomes(train)
    rates = (
        train.with_columns(pl.Series("_outcome", outcomes))
        .group_by("phase")
        .agg(pl.col("_outcome").mean())
    )
    return PhaseLookupModel(
        rate_by_phase=dict(rates.iter_rows()), overall_rate=float(outcomes.mean())
    )


# ---------------------------------------------------------------- logistic rungs


@dataclass(frozen=True)
class Features:
    """The columns a logistic model may read, by how each is turned into numbers."""

    categorical: tuple[str, ...]  # one-hot, with an explicit "(missing)" category
    boolean: tuple[str, ...]  # 0/1, with a missing indicator
    numeric: tuple[str, ...]  # standardised, missing filled with the median plus an indicator

    @property
    def columns(self) -> tuple[str, ...]:
        return self.categorical + self.boolean + self.numeric


def features_from(columns: tuple[str, ...]) -> Features:
    """Sort columns into the three kinds by their type in the dataset schema."""
    kinds: dict[str, list[str]] = {"categorical": [], "boolean": [], "numeric": []}
    for name in columns:
        dtype = SCHEMA[name]
        if dtype == pl.String:
            kinds["categorical"].append(name)
        elif dtype == pl.Boolean:
            kinds["boolean"].append(name)
        elif dtype.is_numeric():
            kinds["numeric"].append(name)
        else:
            raise ValueError(f"No way to model column {name!r} of type {dtype}.")
    return Features(**{kind: tuple(names) for kind, names in kinds.items()})


# The one place the registration columns are defined. Name a column here to take it out of
# every logistic rung, the no-modality rung and the drop-one-column table at once.
#
# start_year is excluded on purpose. Among trials that finished in a given window, a later
# start means a shorter trial, so start year stands in for trial duration, which is only
# recorded after the fact. And the test set lies entirely beyond the training range of years,
# so the model would be extrapolating a trend. It added about 0.001 AUROC and 2.4 points of
# over-optimism on the 2017-2020 rehearsal. The column stays in the dataset.
REGISTRATION_EXCLUDED: tuple[str, ...] = ("start_year",)
REGISTRATION_COLUMNS = tuple(
    c for c in columns_in_group("design") if c not in REGISTRATION_EXCLUDED
)


def registration_features_without(*dropped: str) -> Features:
    """The registration model's inputs less the named columns."""
    unknown = set(dropped) - set(REGISTRATION_COLUMNS)
    if unknown:
        raise ValueError(f"Not registration columns: {sorted(unknown)}.")
    return features_from(tuple(c for c in REGISTRATION_COLUMNS if c not in dropped))


FEATURES = {
    REGISTRATION: registration_features_without(),
    NO_MODALITY: registration_features_without("modality", "is_vaccine"),
    LEAKY: features_from(REGISTRATION_COLUMNS + columns_in_group("post_hoc")),
    RESULTS_STRUCTURE: features_from(REGISTRATION_COLUMNS + columns_in_group("results_structure")),
}


def prepare(table: pl.DataFrame, features: Features) -> pl.DataFrame:
    """Select the model's columns, make non-text ones float, fill missing text, log the counts."""
    text = [pl.col(c).fill_null(MISSING) for c in features.categorical]
    numbers = [
        (pl.col(c).cast(pl.Float64).log1p() if c in LOG_COLUMNS else pl.col(c).cast(pl.Float64))
        for c in features.boolean + features.numeric
    ]
    return table.select(*text, *numbers)


def _category_name(column: str, category: str) -> str:
    return f"{column}={category}"


def build_pipeline(features: Features) -> Pipeline:
    """Prepare, encode, then logistic regression: one scikit-learn pipeline."""
    encode = ColumnTransformer(
        [
            (
                "text",
                OneHotEncoder(handle_unknown="ignore", feature_name_combiner=_category_name),
                list(features.categorical),
            ),
            (
                "boolean",
                SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
                list(features.boolean),
            ),
            # Scale the filled-in values only. A 0/1 missing indicator is left as it is, so a
            # rare one is not stretched to look bigger than the others.
            (
                "numeric",
                make_pipeline(SimpleImputer(strategy="median"), StandardScaler()),
                list(features.numeric),
            ),
            ("numeric_missing", MissingIndicator(error_on_new=False), list(features.numeric)),
        ]
    )
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare, kw_args={"features": features})),
            ("encode", encode),
            ("fit", LogisticRegression(max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)),
        ]
    )


@dataclass
class LogisticModel:
    features: Features
    pipeline: Pipeline

    def predict_probability(self, table: pl.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(table)[:, 1]

    def feature_names(self) -> list[str]:
        """One name per coefficient, such as "sponsor_class=INDUSTRY" or "n_registered_arms"."""
        raw = self.pipeline.named_steps["encode"].get_feature_names_out()
        return [_readable(name) for name in raw]

    def iterations(self) -> int:
        return int(self.pipeline.named_steps["fit"].n_iter_[0])


def _readable(name: str) -> str:
    """ "text__phase_phase3" to "phase=phase3"; "numeric__missingindicator_x" to "x is missing"."""
    _, _, rest = name.partition("__")
    if "missingindicator_" in rest:
        return rest.replace("missingindicator_", "") + " is missing"
    return rest


Model = ConstantModel | PhaseLookupModel | LogisticModel


def fit_logistic(train: pl.DataFrame, features: Features) -> LogisticModel:
    outcomes = training_outcomes(train)
    pipeline = build_pipeline(features).fit(train, outcomes)
    return LogisticModel(features=features, pipeline=pipeline)


RUNGS: dict[str, Callable[[pl.DataFrame], Model]] = {
    CONSTANT: fit_constant,
    PHASE_LOOKUP: fit_phase_lookup,
    REGISTRATION: partial(fit_logistic, features=FEATURES[REGISTRATION]),
    NO_MODALITY: partial(fit_logistic, features=FEATURES[NO_MODALITY]),
    LEAKY: partial(fit_logistic, features=FEATURES[LEAKY]),
    RESULTS_STRUCTURE: partial(fit_logistic, features=FEATURES[RESULTS_STRUCTURE]),
}


def fit_ladder(train: pl.DataFrame) -> dict[str, Model]:
    """Fit every rung on the same training table."""
    return {name: fit(train) for name, fit in RUNGS.items()}
