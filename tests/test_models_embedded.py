import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise import dataset as ds
from elman_exercise import embeddings as emb
from elman_exercise.models import embedded
from elman_exercise.models.registry import all_specs

TINY = emb.EmbeddingModel("tiny", "none/none", "0" * 40, 4, "emb_t")
WORDS = ["asthma inhaler", "cancer infusion", "diabetes pill", "stroke device"]


def frame(n: int, seed: int, prefix: str) -> pl.DataFrame:
    """Synthetic trials with made-up vector and text columns. The label is tied to the vector."""
    trials = synthetic_trials(n, seed=seed, prefix=prefix)
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, TINY.dimension))
    label = vectors[:, 0] + 0.5 * rng.normal(size=n) > 0
    columns = pl.DataFrame(vectors, schema=TINY.columns)
    words = pl.DataFrame(
        {
            "title": rng.choice(WORDS, n).tolist(),
            "conditions": rng.choice(["asthma", "cancer"], n).tolist(),
            "primary_outcomes": rng.choice(["fev1", "survival"], n).tolist(),
        }
    )
    return pl.concat([trials, columns, words], how="horizontal").with_columns(
        any_arm=pl.Series(label), all_arms=pl.Series(label)
    )


SPECS = embedded.specs_for(TINY)


def test_the_registered_specs_all_use_the_embeddings_family_and_a_pretrained_model():
    real = [s for s in all_specs() if s.family == "embeddings"]
    assert len(real) == 6
    assert all(s.pretrained and s.honest for s in real)
    assert {t for s in real for t in s.tables} == {
        "embeddings_general",
        "embeddings_biomedical",
        "text",
    }


def test_only_the_models_with_words_need_the_text_table():
    for spec in SPECS:
        assert ("text" in spec.tables) == spec.name.endswith("plus words")
        assert TINY.table in spec.tables


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_predictions_ignore_the_label_and_the_completion_date_of_the_predicted_trials(spec):
    model = spec.fit(frame(400, 1, "NCT0"))
    target = frame(80, 2, "NCT1")
    base = model.predict(target)
    assert base.shape == (80,) and base.min() >= 0 and base.max() <= 1

    flipped = target.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    no_labels = target.drop([c for c in ds.LABEL_COLUMNS if c in target.columns])
    later = target.with_columns(pl.col("primary_completion_date").dt.offset_by("6y"))
    for changed in (flipped, no_labels, later):
        np.testing.assert_allclose(model.predict(changed), base, rtol=1e-9)


def test_the_vector_carries_the_signal_when_the_label_is_tied_to_it():
    spec = next(s for s in SPECS if s.name.endswith("vector alone"))
    target = frame(300, 2, "NCT1")
    scores = spec.fit(frame(600, 1, "NCT0")).predict(target)
    assert np.corrcoef(scores, target[TINY.columns[0]].to_numpy())[0, 1] > 0.8


def test_nothing_is_fitted_on_the_rows_being_predicted():
    spec = next(s for s in SPECS if s.name.endswith("plus words"))
    train = frame(300, 1, "NCT0")
    target = frame(60, 2, "NCT1")
    first = spec.fit(train).predict(target)
    # Predicting does not change the model, and a different batch of rows leaves a row's score alone.
    model = spec.fit(train)
    model.predict(frame(60, 5, "NCT2"))
    np.testing.assert_allclose(model.predict(target), first, rtol=1e-9)
    np.testing.assert_allclose(model.predict(target.head(1)), first[:1], rtol=1e-9)
