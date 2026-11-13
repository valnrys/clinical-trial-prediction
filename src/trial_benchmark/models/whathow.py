"""WHAT versus HOW: four logistic models per link version.

- HOW: the registration columns less therapeutic area (`whathow.HOW`). It does not depend on the
  links; one copy per version keeps every version's table complete.
- WHAT: therapeutic area, shrunk success rates for target, disease and target x disease, the
  strict precedent flag (experimental-arm drugs only), genetic support when the trial started,
  generic target biology and "has a mapped target / disease" flags (`whathow` says exactly which).
- WHAT + phase: WHAT with the trial's phase added, so WHAT is not asked to explain that a Phase 3
  succeeds more often than a Phase 2.
- HOW + WHAT: both blocks in one model.

The three WHAT models are L2 logistic regressions with a fixed penalty (C = 0.1), nothing tuned.
A trial's group rates are made from the training trials only, under the rule in `whathow`: each
counts only labelled trials whose results were first posted before the scored trial started, and
never the trial itself. The same rule applies to the training rows' own rates, so the model sees
at fit time the kind of rate it will see at predict time.

The WHAT models read a drug-to-target release published after the trials started, so they are not
marked honest. The tables they need are built by `python -m trial_benchmark.whathow`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from trial_benchmark import baselines as bl
from trial_benchmark import whathow
from trial_benchmark.models.registry import ModelSpec, Predictor, fit_pipeline
from trial_benchmark.science import links

PENALTY = 0.1  # C of the three WHAT models: larger means a weaker penalty
LINKS_TEXT = {
    "today": "today's links",
    "2020": "2020 links",
    "2020_names": "2020 names and links",
}
BLOCKS = ("HOW", "WHAT", "WHAT + phase", "HOW + WHAT")
EXCLUDED_TEXT = ", own lead drug excluded"


def column_sets(block: str) -> tuple[list[str], list[str], list[str]]:
    """(text, yes/no, number) columns a block reads, as named in the frame."""
    text, yes_no, numbers = whathow.column_groups()
    if block == "WHAT":
        return text, yes_no, numbers
    if block == "WHAT + phase":
        return [*text, "phase"], yes_no, numbers
    if block == "HOW + WHAT":
        how = whathow.HOW
        return [*text, *how.categorical], [*yes_no, *how.boolean], [*numbers, *how.numeric]
    raise ValueError(f"No logistic columns for block {block!r}.")


def build_pipeline(text: list[str], yes_no: list[str], numbers: list[str]) -> Pipeline:
    """One-hot text, 0/1 flags, standardised numbers; missing values filled with an indicator."""

    def prepare(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.select(
            *[pl.col(c).cast(pl.String).fill_null("(missing)") for c in text],
            *[pl.col(c).cast(pl.Float64) for c in [*yes_no, *numbers]],
        )

    encode = ColumnTransformer(
        [
            ("text", OneHotEncoder(handle_unknown="ignore"), text),
            (
                "yes_no",
                SimpleImputer(strategy="constant", fill_value=0.0, add_indicator=True),
                yes_no,
            ),
            ("number", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numbers),
            ("number_missing", MissingIndicator(error_on_new=False), numbers),
        ]
    )
    return Pipeline(
        [
            ("prepare", FunctionTransformer(prepare)),
            ("encode", encode),
            ("fit", LogisticRegression(C=PENALTY, max_iter=5000)),
        ]
    )


@dataclass
class WhatPredictor:
    """A fitted WHAT-containing model, with the labelled pool its group rates draw on."""

    pipeline: Pipeline
    pool: pl.DataFrame  # nct_id, wh_posted_last, lead_drug, y: the training trials
    keys: pl.DataFrame
    exclude: str | None

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        """Probability of success per row. Group rates come from the pool only, under the rule."""
        scored = whathow.what_frame(frame, self.pool, self.keys, self.exclude)
        return self.pipeline.predict_proba(scored)[:, 1]


def fit_what(
    train: pl.DataFrame,
    block: str,
    version: str,
    exclude: str | None = None,
    keys: pl.DataFrame | None = None,
) -> Predictor:
    """Fit one block on `train`. `keys` defaults to the version's table on disk."""
    if block == "HOW":
        return fit_pipeline(train, whathow.HOW)
    keys = whathow.load_keys(version) if keys is None else keys
    pool = whathow.pool_of(train)
    rated = whathow.what_frame(train, pool, keys, exclude)
    pipeline = build_pipeline(*column_sets(block)).fit(rated, bl.training_outcomes(train))
    return WhatPredictor(pipeline, pool, keys, exclude)


def describe(block: str, version: str, exclude: str | None) -> str:
    """One plain sentence for the model's description."""
    rule = (
        "Group rates count only labelled trials whose results were first posted before the "
        "scored trial started"
        + (
            "; trials of the scored trial's own lead drug are left out of every rate"
            if exclude
            else ""
        )
        + "."
    )
    what = (
        "therapeutic area, shrunk success rates for target, disease and target x disease, "
        "the strict precedent flag, genetic support at the start, target biology and mapped flags"
    )
    if block == "HOW":
        return "The registration columns less therapeutic area (how the trial is run)."
    extra = {
        "WHAT": f"{what}.",
        "WHAT + phase": f"{what}, plus phase.",
        "HOW + WHAT": f"{what}, plus the how columns.",
    }[block]
    return f"Logistic on {extra} {LINKS_TEXT[version].capitalize()}. {rule}"


def spec_name(block: str, version: str, exclude: str | None = None) -> str:
    return f"logistic: {block} ({LINKS_TEXT[version]}{EXCLUDED_TEXT if exclude else ''})"


def make_spec(block: str, version: str, exclude: str | None = None) -> ModelSpec:
    return ModelSpec(
        spec_name(block, version, exclude),
        "whathow",
        describe(block, version, exclude),
        partial(fit_what, block=block, version=version, exclude=exclude),
        tables=("genetics", whathow.table_name(version)),
        own_tables=(whathow.keys_name(version),),
        honest=block == "HOW",
    )


def make_specs(version: str, exclude: str | None = None) -> list[ModelSpec]:
    """The four models for one link version (HOW has no links, so it is left out when excluding)."""
    blocks = BLOCKS[1:] if exclude else BLOCKS
    return [make_spec(block, version, exclude) for block in blocks]


SPECS = [spec for version in links.VERSIONS for spec in make_specs(version)]
