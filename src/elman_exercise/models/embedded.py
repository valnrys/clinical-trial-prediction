"""Sentence-embedding models: do pretrained text vectors add anything to plain words?

These are the only models that use a pretrained model (see `elman_exercise.embeddings`),
so every spec here is marked `pretrained=True`. They read a table of vectors built in
advance. This module never imports torch or sentence-transformers, and a spec whose table
has not been built is skipped by the validation harness.

Per encoder (general, biomedical) there are three models, each a logistic regression:

- alone: the vector only.
- registration plus vector: the registration baseline's inputs beside the vector.
- registration plus vector plus words: as above, plus the TF-IDF word features of
  `models.text`. Compare it with "ladder: registration + words" to see whether the
  vector adds anything to the words.

The vector columns are standardised, like every number column in the benchmark; the word
features are scaled up by `WORD_WEIGHT` so one penalty suits both. The one tuned setting, the
strength of the penalty, is chosen on the training trials only: fit on the oldest three quarters, score the newest quarter (`registry.best_by_inner_split`), then refit on
all of them. Word features learn their vocabulary from the training trials inside `fit`.

Warning. A pretrained encoder may have seen these trials, even their results, while it was
trained. Small sentence encoders memorise little and none was trained to predict outcomes,
but the time split does not protect against it. See `elman_exercise.embeddings`.

A limit on the comparison. The chosen penalty in these models is small (about C = 0.003 to
0.01), so the registration columns are shrunk far harder here than in "ladder: registration +
words" (C = 1). "Embedding plus words versus words" therefore mixes what the vector adds with
heavier shrinkage of the registration columns. Read it as "no evidence the vector adds
anything", not as a like-for-like ablation.

A second difference in the same comparison. The text that is turned into a vector includes the
intervention (drug) names, but the word features (`models.text.WORD_FIELDS`) read only the title,
conditions and primary outcomes. So the vector models see drug names and "ladder: registration +
words" does not.

A model is honest only if every column it reads is dated honestly. The vectors are dated
"protocol", like the text they come from, so all six are honest in that sense.
"""

from __future__ import annotations

from functools import partial

import polars as pl
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from elman_exercise.baselines import (
    FEATURES as BASELINE_FEATURES,
)
from elman_exercise.baselines import (
    MAX_ITERATIONS,
    RANDOM_STATE,
    REGISTRATION,
    Features,
    build_pipeline,
    training_outcomes,
)
from elman_exercise.dating import HONEST
from elman_exercise.embeddings import BIOMEDICAL, GENERAL, EmbeddingModel
from elman_exercise.models.registry import ModelSpec, PipelinePredictor, best_by_inner_split
from elman_exercise.models.text import WORD_FIELDS, word_features
from elman_exercise.text import TABLE as TEXT_TABLE

PENALTY_STRENGTHS = (0.001, 0.003, 0.01, 0.03, 0.1)  # C: larger means a weaker penalty
# Word features are scaled up before the penalty. The word models elsewhere use C = 1 (a fixed
# default in `models.text`, not tuned). One small C is chosen here for the whole model, which
# would shrink TF-IDF columns far too hard; ten times the scale (a hundred times the penalty
# weight) keeps the word block on roughly that footing when C is near 0.01.
WORD_WEIGHT = 10.0


def feature_columns(model: EmbeddingModel, with_registration: bool) -> Features:
    """The vector columns (as numbers), with the registration columns when asked."""
    vector = Features((), (), tuple(model.columns))
    if not with_registration:
        return vector
    registration = BASELINE_FEATURES[REGISTRATION]
    return Features(
        registration.categorical,
        registration.boolean,
        registration.numeric + vector.numeric,
    )


def fit_with_strength(
    train: pl.DataFrame,
    model: EmbeddingModel,
    with_registration: bool,
    with_words: bool,
    strength: float,
) -> PipelinePredictor:
    """The chosen inputs, encoded as in the benchmark, then a logistic regression."""
    columns = build_pipeline(feature_columns(model, with_registration))[:-1]  # all but the fit
    columns.set_params(encode__sparse_threshold=0.0)  # dense, as in registry.fit_pipeline
    parts = [("columns", columns)]
    if with_words:
        parts.append(("word", word_features(WORD_FIELDS)))
    logistic = LogisticRegression(C=strength, max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE)
    union = FeatureUnion(parts, transformer_weights={"word": WORD_WEIGHT} if with_words else None)
    pipeline = Pipeline([("features", union), ("fit", logistic)])
    return PipelinePredictor(pipeline.fit(train, training_outcomes(train)))


def fit_embedded(
    train: pl.DataFrame, model: EmbeddingModel, with_registration: bool, with_words: bool
) -> PipelinePredictor:
    """Choose the penalty on the training trials' own time split, then fit on all of them."""
    strength = best_by_inner_split(
        train,
        list(PENALTY_STRENGTHS),
        lambda table, c: fit_with_strength(table, model, with_registration, with_words, c),
    )
    return fit_with_strength(train, model, with_registration, with_words, strength)


def reads_honestly(model: EmbeddingModel) -> bool:
    return model.dating in HONEST


def embedded_spec(
    model: EmbeddingModel, label: str, description: str, with_registration: bool, with_words: bool
) -> ModelSpec:
    needs = (TEXT_TABLE,) if with_words else ()  # the word features read the text table
    return ModelSpec(
        f"embeddings: {model.key} {label}",
        "embeddings",
        description,
        partial(
            fit_embedded,
            model=model,
            with_registration=with_registration,
            with_words=with_words,
        ),
        tables=(model.table, *needs),
        honest=reads_honestly(model),
        pretrained=True,
    )


def specs_for(model: EmbeddingModel) -> list[ModelSpec]:
    return [
        embedded_spec(
            model,
            "vector alone",
            f"Logistic regression on the {model.key} sentence vector of the registration text.",
            with_registration=False,
            with_words=False,
        ),
        embedded_spec(
            model,
            "vector plus registration",
            f"Registration columns beside the {model.key} sentence vector, then a logistic regression.",
            with_registration=True,
            with_words=False,
        ),
        embedded_spec(
            model,
            "vector plus registration plus words",
            f"Registration columns, the {model.key} sentence vector and word features, then a logistic regression.",
            with_registration=True,
            with_words=True,
        ),
    ]


SPECS = [spec for model in (GENERAL, BIOMEDICAL) for spec in specs_for(model)]
