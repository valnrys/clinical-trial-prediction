"""Models that read the science table: what the drug's target is like, in public databases.

- "biology: logistic, all sources" and "biology: boosted trees, all sources" read only the
  science columns: no registration columns at all.
- "biology: logistic, <source> only" (five models) each read one source group (DepMap,
  expression, networks, annotations, chemistry), so each source's own score can be seen.
- "biology: registration + <source>" (five models) is the registration logistic plus one source's
  columns: what each source adds to what registration already says.
- "biology: registration + all sources" is the registration logistic plus every science column.
- "biology: ladder, all four tables + science" is the information ladder's all-tables logistic
  (`models.ladder`) plus every science column.

Which columns: every science column whose dating is honest (`science.FEATURES`; timeless,
dated or protocol). `has_target` and `n_target_genes` are "after the start" and are never read. A
spec's `honest` flag comes from that rule, not from a hand-set value.

Missing values must not say "mapped"
------------------------------------
Every science column is empty for a trial whose drug has no mapped target, and being mapped is
hindsight (see `science`). Any way of filling the blanks that is not neutral lets a model see it.
Filling with the commonest category, for example, makes every rare category a sign that the trial
was mapped, and trees can isolate a pile of identical filled rows. So:

1. Text columns become one yes/no column per level (levels seen in at least `MIN_LEVEL`
   training trials). Yes/no columns and numbers stay as they are.
2. Each column is centred on its mean over the training trials that have a value.
3. Anything missing becomes zero.

A trial with no science value is then zero in every science column: it adds nothing to the
linear score, whatever the model learned. The benchmark's pipeline normally adds a "was this
missing" indicator for numbers; none is created here because nothing is missing by the time the
pipeline sees the columns (`test_models_biology.py` checks the fitted pipelines for one). Because
the centred columns have mean zero over the training rows, the pipeline's own standardisation
leaves the zeros where they are.

Models that read science columns only
-------------------------------------
These are DEFINED, FITTED and REPORTED on trials that have at least one science value. For a trial
with none they predict one constant: the model's own prediction at the centring point (every
encoded column zero), not the base rate, so nothing separates an unmapped trial from one that
sits at the training mean. The score they get on "all trials" is not meaningful: it compares a
constant with real predictions. Boosted trees are fitted on the trials with a value only, so no
tree can learn to route filled rows. A trial with a value in some sources and none in others
sees zeros for the missing ones, which marks "gene not in that source", not "drug not mapped".

The trees can still read which sources are empty for a trial. No chemistry value, for example,
usually means the lead drug is a biologic, so the science-only boosted trees partly encode drug
type. That is a property of the data, not a leak of the label; the trees' lead over the logistic
model should not be read as biology alone (`results/dev_science.md`, section 7, checks that it
is not the number of targets either).

Models that also read registration columns
------------------------------------------
Registration-plus-science and ladder-plus-science are logistic models: the zero fill is exactly
neutral, so an unmapped trial is predicted from its registration columns alone.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import dating, science
from trial_benchmark.models import ladder, text
from trial_benchmark.models.registry import (
    ModelSpec,
    Predictor,
    features_from_frame,
    fit_pipeline,
)
from trial_benchmark.models.trees import boosted

TABLE = "science"
MIN_LEVEL = 30  # a text level seen in fewer training trials than this gets no column


def honest_columns(sources: tuple[str, ...] | None = None) -> list[str]:
    """Science columns with an honest dating, for the named sources (default: all), in order."""
    names = sources or tuple(science.SOURCES)
    return [
        column
        for source in names
        for column, label in science.SOURCES[source].FEATURES.items()
        if label in dating.HONEST
    ]


def is_honest(columns: list[str]) -> bool:
    """True if every column read has an honest dating."""
    return all(science.FEATURES[c] in dating.HONEST for c in columns)


def has_value(frame: pl.DataFrame, columns: list[str]) -> pl.Series:
    """True for each row that has at least one non-null science value among `columns`."""
    return frame.select(pl.any_horizontal(pl.col(c).is_not_null() for c in columns)).to_series()


# ---------------------------------------------------------------- centring


@dataclass
class Centring:
    """What the training table says about each science column, for the three steps above.

    `numbers`: column -> mean over trials with a value. `yes_no`: the same for yes/no columns
    (as 0/1). `levels`: text column -> {level: share of valued trials having it}.
    """

    numbers: dict[str, float]
    yes_no: dict[str, float]
    levels: dict[str, dict[str, float]]

    def level_names(self) -> list[str]:
        return [f"{column}={level}" for column, shares in self.levels.items() for level in shares]

    def apply(self, frame: pl.DataFrame) -> pl.DataFrame:
        """`frame` with every learned column centred, missing as zero, text as level columns."""
        centred = [
            (pl.col(name).cast(pl.Float64) - mean).fill_null(0.0).alias(name)
            for name, mean in {**self.numbers, **self.yes_no}.items()
        ]
        levels = [
            pl.when(pl.col(column).is_null())
            .then(0.0)
            .otherwise((pl.col(column) == level).cast(pl.Float64) - share)
            .alias(f"{column}={level}")
            for column, shares in self.levels.items()
            for level, share in shares.items()
        ]
        return frame.with_columns(*centred, *levels)

    def features(self, registration: bl.Features | None = None) -> bl.Features:
        """The pipeline's column lists: registration columns (if any), then the centred ones.

        Numbers go where the pipeline standardises; yes/no and level columns go where it does
        not, so a rare level is not stretched.
        """
        base = registration or bl.Features((), (), ())
        return bl.Features(
            base.categorical,
            base.boolean + tuple(self.yes_no) + tuple(self.level_names()),
            base.numeric + tuple(self.numbers),
        )


def fit_centring(train: pl.DataFrame, columns: list[str]) -> Centring:
    """Means and level shares from the trials in `train` that have a value in each column.

    A column with no value anywhere in `train` is left out.
    """
    centring = Centring({}, {}, {})
    for name in columns:
        values = train[name].drop_nulls()
        if values.is_empty():
            continue
        if values.dtype == pl.String:
            counts = values.value_counts().filter(pl.col("count") >= MIN_LEVEL).sort(name)
            centring.levels[name] = {
                level: count / values.len()
                for level, count in zip(counts[name], counts["count"], strict=True)
            }
        elif values.dtype == pl.Boolean:
            centring.yes_no[name] = float(values.cast(pl.Float64).mean())
        else:
            centring.numbers[name] = float(values.cast(pl.Float64).mean())
    return centring


@dataclass
class CentredPredictor:
    """Centres the science columns of any frame, then asks the fitted model."""

    model: Predictor
    centring: Centring

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.predict(self.centring.apply(frame))


# ---------------------------------------------------------------- fitting


def fit_science_only(
    train: pl.DataFrame, columns: list[str], boosted_trees: bool = False
) -> Predictor:
    """Logistic (or boosted trees) on `columns` alone, fitted on trials with a science value.

    A trial with no value in `columns` is zero in every encoded column, so it gets exactly the
    prediction of a trial at the centring point: one constant, with nothing to tell it from
    a trial that sits at the training mean.
    """
    centring = fit_centring(train, columns)
    valued = centring.apply(train.filter(has_value(train, columns)))
    model = fit_pipeline(valued, centring.features(), boosted() if boosted_trees else None)
    return CentredPredictor(model, centring)


def fit_with_registration(train: pl.DataFrame, columns: list[str]) -> Predictor:
    """The registration logistic plus `columns`, fitted on every training trial."""
    centring = fit_centring(train, columns)
    centred = centring.apply(train)
    registration = features_from_frame(centred, bl.REGISTRATION_COLUMNS)
    return CentredPredictor(fit_pipeline(centred, centring.features(registration)), centring)


def fit_ladder(
    train: pl.DataFrame, columns: list[str], tables: tuple[str, ...] = ladder.ALL_FOUR
) -> Predictor:
    """The information ladder's logistic over `tables` (honest columns) plus `columns`.

    Mirrors `ladder.fit_rung`, which cannot take centred columns: unknown genetics is filled as
    "no support", and the word columns are read when the words table is among `tables`.
    """
    centring = fit_centring(train, columns)
    filled = ladder.fill_unknown_as_no_support(centring.apply(train))
    read = ladder.columns_read(tables, dating.HONEST)
    words = tuple(n for n in read if n in ladder.WORD_FIELDS and ladder.WORD_TABLE in tables)
    others = [name for name in read if name not in words]
    registration = features_from_frame(filled, [*bl.REGISTRATION_COLUMNS, *others])
    features = centring.features(registration)
    if words:
        model = text.fit_features_and_words(filled, features, words)
    else:
        model = fit_pipeline(filled, features)
    return CentredPredictor(ladder.FilledPredictor(model), centring)


# ---------------------------------------------------------------- the specs

ALL = honest_columns()
SOURCE_LABELS = {
    "depmap": "DepMap",
    "expression": "expression",
    "networks": "networks",
    "annotations": "annotations",
    "chemistry": "chemistry",
}
ONLY_MAPPED = " Defined on trials with a mapped target; the others get one constant."
NOT_RANKED = (
    "not meaningful on all trials: they predict one constant where a trial has no mapped target; "
    "scored on mapped trials in `results/dev_science.md`"
)
SCIENCE_ONLY = {"ranked": False, "not_ranked_reason": NOT_RANKED}


def source_spec(source: str, label: str) -> ModelSpec:
    """The logistic model that reads one source's columns only."""
    columns = honest_columns((source,))
    return ModelSpec(
        f"biology: logistic, {label} only",
        "biology",
        f"Logistic regression on the {label} columns alone." + ONLY_MAPPED,
        lambda train: fit_science_only(train, columns),
        tables=(TABLE,),
        honest=is_honest(columns),
        **SCIENCE_ONLY,
    )


def increment_spec(source: str, label: str) -> ModelSpec:
    """The registration logistic plus one source's columns: what that source adds."""
    columns = honest_columns((source,))
    return ModelSpec(
        f"biology: registration + {label}",
        "biology",
        f"The registration logistic plus the {label} columns.",
        lambda train: fit_with_registration(train, columns),
        tables=(TABLE,),
        honest=is_honest(columns),
    )


SPECS = [
    ModelSpec(
        "biology: logistic, all sources",
        "biology",
        "Logistic regression on the science columns alone (no registration columns)." + ONLY_MAPPED,
        lambda train: fit_science_only(train, ALL),
        tables=(TABLE,),
        honest=is_honest(ALL),
        **SCIENCE_ONLY,
    ),
    ModelSpec(
        "biology: boosted trees, all sources",
        "biology",
        "Boosted trees on the science columns alone (no registration columns)." + ONLY_MAPPED,
        lambda train: fit_science_only(train, ALL, boosted_trees=True),
        tables=(TABLE,),
        honest=is_honest(ALL),
        **SCIENCE_ONLY,
    ),
    *[source_spec(source, label) for source, label in SOURCE_LABELS.items()],
    *[increment_spec(source, label) for source, label in SOURCE_LABELS.items()],
    ModelSpec(
        "biology: registration + all sources",
        "biology",
        "The registration logistic plus every science column.",
        lambda train: fit_with_registration(train, ALL),
        tables=(TABLE,),
        honest=is_honest(ALL),
    ),
    ModelSpec(
        "biology: ladder, all four tables + science",
        "biology",
        "The information ladder's all-tables logistic plus every science column.",
        lambda train: fit_ladder(train, ALL),
        tables=(*ladder.ALL_FOUR, TABLE),
        honest=is_honest(ALL),
    ),
]
