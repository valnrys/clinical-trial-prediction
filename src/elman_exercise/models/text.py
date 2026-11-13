"""Word-count models: what the words of a registration say about success.

No pretrained model of any kind. Each model counts words and two-word phrases
(TF-IDF), then fits a logistic regression. Settings are fixed and not tuned.
The vocabulary is learned from the training trials only, inside `fit`, so a
word that appears only in the trials being predicted has no effect.

The models, and what each is for:

- words: title, conditions and primary outcomes. The plain text baseline.
- drug names only: the intervention names alone. A memorisation probe. It can
  only work by recognising drugs it saw in training.
- words without drug names: as "words", but each trial's own drug names are removed
  from its text, and the interventions are not read. A drug name here means a word in
  the names of the trial's interventions, after dose, unit, route and form words, numbers
  and a short list of plain words (placebo, standard, control...) are set aside. So
  "placebo", "oral" and "week" stay; "zorbex" goes. The gap to "words" shows how much
  was name recognition.
- words with summary (leaky): "words" plus the brief summary, which sponsors can
  rewrite with results. Deliberately not honest.

Whether a model is honest is worked out, not declared: it is honest only if every
text field it reads is dated honestly in `elman_exercise.text.FEATURES`. Only the
summary is not (sponsors can rewrite it once results are in).

(The registration baseline plus words is the ladder rung "ladder: registration + words";
`fit_features_and_words` below is what that rung uses.)

Words never join across fields or across items of a list: two-word phrases are
built inside one title, one condition or one outcome only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import partial

import numpy as np
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from elman_exercise.baselines import (
    MAX_ITERATIONS,
    RANDOM_STATE,
    Features,
    build_pipeline,
    training_outcomes,
)
from elman_exercise.dating import HONEST
from elman_exercise.history import drug_name
from elman_exercise.models.registry import ModelSpec
from elman_exercise.text import FEATURES as TEXT_FEATURES
from elman_exercise.text import SEPARATOR, TABLE

# Fixed settings.
MIN_DOCUMENT_FREQUENCY = 5  # a term must appear in at least this many training trials
DRUG_MIN_DOCUMENT_FREQUENCY = 3  # drug names are rarer: at least three trials
MAX_FEATURES = 20000
REGULARISATION = 1.0  # scikit-learn's default strength for the inverse penalty C
WORD_FIELDS = ("title", "conditions", "primary_outcomes")
WORD = re.compile(r"[a-z0-9]+")
BOUNDARY = "\n"  # between fields and between items of a list; no phrase crosses it
# Plain words that can stand in an intervention name without being a drug's name.
# (Dose, unit, route and form words are dropped by `history.drug_name`.)
# fmt: off
NOT_NAMES = frozenset({
    "placebo", "matching", "sham", "control", "standard", "care", "usual", "treatment",
    "therapy", "drug", "vehicle", "saline", "intravenous", "subcutaneous", "intramuscular",
    "topical", "iv", "sc", "im", "po", "week", "weeks", "day", "days", "once", "twice",
    "weekly", "monthly", "dose", "doses", "arm", "group", "study",
})
# fmt: on


def _tokens(text: str | None) -> list[str]:
    return WORD.findall((text or "").lower())


def drug_tokens(interventions: str | None) -> set[str]:
    """The words that name the drugs in a trial's intervention names (joined with SEPARATOR).

    Words in brackets count: they are often the brand name.
    """
    words: set[str] = set()
    for name in (interventions or "").split(SEPARATOR):
        for part in [name, *re.findall(r"\(([^)]*)\)", name)]:
            words.update(_tokens(drug_name(part)))
    return {w for w in words if w not in NOT_NAMES and not w.isdigit()}


def documents(frame: pl.DataFrame, fields: tuple[str, ...], drop_own_drugs: bool) -> list[str]:
    """One string per trial: the chosen fields, split at BOUNDARY, own drug names optionally removed."""
    own_drugs = [drug_tokens(text) for text in frame["interventions"]] if drop_own_drugs else None
    texts = zip(*(frame[field].to_list() for field in fields), strict=True)
    result = []
    for i, parts in enumerate(texts):
        segments = [item for part in parts for item in (part or "").split(SEPARATOR)]
        if own_drugs is not None:
            segments = [" ".join(t for t in _tokens(x) if t not in own_drugs[i]) for x in segments]
        else:
            segments = [" ".join(_tokens(x)) for x in segments]
        result.append(BOUNDARY.join(segments))
    return result


_WITHIN_ONE_SEGMENT = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").build_analyzer()


def _terms(document: str) -> list[str]:
    """Words and two-word phrases of each segment of `document`, never across a boundary."""
    return [term for segment in document.split(BOUNDARY) for term in _WITHIN_ONE_SEGMENT(segment)]


def _vectoriser(min_document_frequency: int) -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer=_terms,
        min_df=min_document_frequency,
        max_features=MAX_FEATURES,
        sublinear_tf=True,
    )


def word_features(
    fields: tuple[str, ...], drop_own_drugs: bool = False, min_df: int = MIN_DOCUMENT_FREQUENCY
) -> Pipeline:
    """Frame -> documents -> TF-IDF counts. The vocabulary is learned when this is fitted."""
    to_documents = FunctionTransformer(
        documents, kw_args={"fields": fields, "drop_own_drugs": drop_own_drugs}
    )
    return Pipeline([("documents", to_documents), ("tfidf", _vectoriser(min_df))])


@dataclass
class WordModel:
    """A fitted word-count model. `top_terms` shows what it learned."""

    pipeline: Pipeline

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(frame)[:, 1]

    def weights(self) -> dict[str, float]:
        """Coefficient of each input, named "word:term" or "registration:column=value"."""
        names = [
            f"{part}:{name.split('__', 1)[-1]}"  # drop the encoder's "text__" style prefix
            for part, transformer in self.pipeline.named_steps["features"].transformer_list
            for name in transformer[-1].get_feature_names_out()
        ]
        coefficients = self.pipeline.named_steps["fit"].coef_[0]
        return dict(zip(names, map(float, coefficients), strict=True))

    def top_terms(self, n: int = 15) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
        """The n highest and the n lowest weighted terms."""
        ranked = sorted(self.weights().items(), key=lambda item: item[1])
        return ranked[::-1][:n], ranked[:n]


def _fit(train: pl.DataFrame, features: FeatureUnion) -> WordModel:
    pipeline = Pipeline(
        [
            ("features", features),
            (
                "fit",
                LogisticRegression(
                    C=REGULARISATION, max_iter=MAX_ITERATIONS, random_state=RANDOM_STATE
                ),
            ),
        ]
    )
    return WordModel(pipeline.fit(train, training_outcomes(train)))


def fit_words(
    train: pl.DataFrame,
    fields: tuple[str, ...] = WORD_FIELDS,
    drop_own_drugs: bool = False,
    min_df: int = MIN_DOCUMENT_FREQUENCY,
) -> WordModel:
    return _fit(train, FeatureUnion([("word", word_features(fields, drop_own_drugs, min_df))]))


def fit_features_and_words(
    train: pl.DataFrame, features: Features, fields: tuple[str, ...] = WORD_FIELDS
) -> WordModel:
    """The benchmark's encoding of `features` beside word features of `fields`, then a logistic."""
    columns = build_pipeline(features)[:-1]  # all but the regression
    # Always dense, as in registry.fit_pipeline: a sparse result crashes on a small window.
    columns.set_params(encode__sparse_threshold=0.0)
    return _fit(train, FeatureUnion([("registration", columns), ("word", word_features(fields))]))


def reads_honestly(fields: tuple[str, ...]) -> bool:
    """True if every field is dated as something that could be known when the trial started."""
    return all(TEXT_FEATURES[field] in HONEST for field in fields)


def text_spec(name: str, description: str, fit, fields: tuple[str, ...]) -> ModelSpec:
    """A text model, honest only if every field it reads is (see `text.FEATURES`)."""
    return ModelSpec(name, "text", description, fit, tables=(TABLE,), honest=reads_honestly(fields))


SPECS = [
    text_spec(
        "text: words",
        "Logistic regression on words and two-word phrases of the title, conditions and primary outcomes.",
        fit_words,
        WORD_FIELDS,
    ),
    text_spec(
        "text: drug names only",
        "Logistic regression on the intervention names alone: it can only work by recognising drugs.",
        partial(fit_words, fields=("interventions",), min_df=DRUG_MIN_DOCUMENT_FREQUENCY),
        ("interventions",),
    ),
    text_spec(
        "text: words without drug names",
        "As the words model, with each trial's own drug names removed from its text.",
        partial(fit_words, drop_own_drugs=True),
        (*WORD_FIELDS, "interventions"),
    ),
    text_spec(
        "text: words with summary (leaky)",
        "The words model plus the brief summary, which sponsors can rewrite with results.",
        partial(fit_words, fields=(*WORD_FIELDS, "summary")),
        (*WORD_FIELDS, "summary"),
    ),
]
