"""The information ladder: the registration logistic, given more to read each rung.

Each rung adds a table of features to the registration columns:

- + endpoint: what the trial's primary endpoint is (endpoint.parquet)
- + history: the sponsor's and drug's track record before the trial (history.parquet)
- + genetics: human genetic evidence for the drug's target (genetics.parquet)
- + words: the words of the title, conditions and primary outcomes (text.parquet)
- + all four
- boosted trees on the first three (the word counts are too many columns for trees)
- + all four, with the "after the start" columns too (NOT honest, see below)

Every feature table says, per column, how it is dated (see `trial_benchmark.dating`).
A rung reads a column only if its dating is in the rung's allow-list. The honest
rungs allow `dating.HONEST`: timeless, dated and protocol columns. The last rung
also allows "after the start" columns (and the brief summary, as words); it shows
how much a model gains from reading later knowledge into the past.

One rule decides honesty everywhere: a rung is honest if every column it reads has
a dating in `dating.HONEST`. It is worked out from the columns, not set by hand.
The column lists come from each table's module (`FEATURES`).

Unknown genetics is treated as "no support"
--------------------------------------------
`mapped` (could Open Targets link the drug at all?) is labelled "after the start": being
catalogued is partly hindsight. The at-start genetics columns are null for unmapped trials,
and the benchmark's pipeline adds a "was this missing" indicator for numbers and yes/no
columns (trees see a null directly). Left alone, every honest rung that reads genetics would
read `mapped` through the back door. `lead_drug_earlier_trials` has the same problem: it is
null for a trial with no lead drug, and the lead drug comes from today's mapping.

So an honest rung fills every null in `NO_SUPPORT`'s columns, before fitting and before
predicting, with the "no support" value: score 0, supported False, 0 earlier trials of the
lead drug. In words: unknown genetics is treated as no known genetic support, whether or not
the trial is mapped, and nothing marks which trials were unknown. The fill does not look at
`mapped`, so an honest rung needs no "after the start" column to be present. The "after the
start" rung is not given the fill: it is meant to see the raw columns.

Columns no rung reads
---------------------
`IDENTIFIER_COLUMNS` (the lead drug's id, the target ids) are names, not measurements. One-hot
encoded they would become thousands of categories that let a model memorise drugs, so no
rung reads them, whatever its dating allows. Memorising drugs is tested separately, in
`models.memorisation`.

Boosted trees cannot be combined with words (the word counts are too many columns for
trees); asking for both raises an error instead of quietly dropping one.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass

import numpy as np
import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import dating
from trial_benchmark.models import text
from trial_benchmark.models.registry import (
    ModelSpec,
    Predictor,
    features_from_frame,
    fit_pipeline,
)
from trial_benchmark.models.trees import boosted

WORD_TABLE = text.TABLE
# Genetics columns whose nulls would show which trials are unmapped, and their "no support" value.
NO_SUPPORT = {
    "genetic_score_at_start": 0.0,
    "supported_at_start": False,
    "supported_at_start_investigational": False,
    "lead_drug_earlier_trials": 0,
}
# Names rather than measurements: never read by any rung (see the docstring).
IDENTIFIER_COLUMNS = frozenset(
    {"lead_drug", "target_ids", "top_target_today", "top_target_at_start"}
)
# The text fields the word features read: these, then the summary if after-the-start is allowed.
WORD_FIELDS = text.WORD_FIELDS + ("summary",)


def fill_unknown_as_no_support(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill every null in the `NO_SUPPORT` columns with its "no support" value (see the docstring)."""
    return frame.with_columns(
        pl.col(name).fill_null(value) for name, value in NO_SUPPORT.items() if name in frame.columns
    )


@dataclass
class FilledPredictor:
    """Applies `fill_unknown_as_no_support` to a frame, then asks the fitted model."""

    model: Predictor

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.model.predict(fill_unknown_as_no_support(frame))


def table_features(table: str) -> dict[str, str]:
    """Column name -> dating, for one feature table. Refuses a label `dating` does not know."""
    module = importlib.import_module(f"trial_benchmark.{table}")
    if not hasattr(module, "FEATURES"):
        raise ValueError(f"trial_benchmark.{table} does not list its FEATURES.")
    dating.check(module.FEATURES)
    return module.FEATURES


def columns_read(tables: tuple[str, ...], allowed: frozenset[str]) -> dict[str, str]:
    """Column -> dating for everything a rung reads from `tables`, given its allow-list."""
    read = {}
    for table in tables:
        if table == WORD_TABLE:
            wanted = {f: d for f, d in table_features(table).items() if f in WORD_FIELDS}
        else:
            wanted = table_features(table)
        read.update(
            {
                name: label
                for name, label in wanted.items()
                if label in allowed and name not in IDENTIFIER_COLUMNS
            }
        )
    return read


def fit_rung(
    train: pl.DataFrame,
    tables: tuple[str, ...],
    allowed: frozenset[str] = dating.HONEST,
    boosted_trees: bool = False,
) -> Predictor:
    fill = allowed <= dating.HONEST  # only honest rungs; the other is meant to see raw columns
    if fill:
        train = fill_unknown_as_no_support(train)
    read = columns_read(tables, allowed)
    words = tuple(name for name in read if name in WORD_FIELDS and WORD_TABLE in tables)
    if words and boosted_trees:
        raise ValueError("Boosted trees cannot read word features; choose one.")
    others = [name for name in read if name not in words]
    features = features_from_frame(train, [*bl.REGISTRATION_COLUMNS, *others])
    if words:
        model = text.fit_features_and_words(train, features, words)
    else:
        model = fit_pipeline(train, features, boosted() if boosted_trees else None)
    return FilledPredictor(model) if fill else model


def rung(
    name: str,
    tables: tuple[str, ...],
    description: str,
    allowed: frozenset[str] = dating.HONEST,
    boosted_trees: bool = False,
) -> ModelSpec:
    unknown = allowed - set(dating.ALL)
    if unknown:
        raise ValueError(f"unknown dating labels: {sorted(unknown)}")
    honest = all(label in dating.HONEST for label in columns_read(tables, allowed).values())
    return ModelSpec(
        name,
        "ladder",
        description,
        lambda train: fit_rung(train, tables, allowed, boosted_trees),
        tables=tables,
        honest=honest,
    )


FIRST_THREE = ("endpoint", "history", "genetics")
ALL_FOUR = (*FIRST_THREE, WORD_TABLE)
EVERYTHING = frozenset(dating.ALL)
SPECS = [
    rung(
        "ladder: registration + endpoint",
        ("endpoint",),
        "The registration logistic plus features of the primary endpoint.",
    ),
    rung(
        "ladder: registration + history",
        ("history",),
        "The registration logistic plus the sponsor's and drug's record before the trial.",
    ),
    rung(
        "ladder: registration + genetics",
        ("genetics",),
        "The registration logistic plus genetic evidence dated before the trial started.",
    ),
    rung(
        "ladder: registration + words",
        (WORD_TABLE,),
        "The registration logistic plus words of the title, conditions and primary outcomes.",
    ),
    rung(
        "ladder: registration + endpoint + history + genetics + words",
        ALL_FOUR,
        "The registration logistic plus all four tables, honest columns only.",
    ),
    rung(
        "ladder: boosted trees, endpoint + history + genetics",
        FIRST_THREE,
        "Boosted trees on the registration columns plus endpoint, history and genetics, honest columns only.",
        boosted_trees=True,
    ),
    rung(
        "ladder: all four tables plus after-the-start columns (not honest)",
        ALL_FOUR,
        "The all-tables logistic that also reads columns, and the summary, that may include later knowledge.",
        allowed=EVERYTHING,
    ),
]
