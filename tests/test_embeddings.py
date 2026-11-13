import subprocess
import sys

import numpy as np
import polars as pl
import pytest

from trial_benchmark import embeddings as emb
from trial_benchmark import text
from trial_benchmark.dating import HONEST

TINY = emb.EmbeddingModel("tiny", "none/none", "0" * 40, 3, "emb_t")
TEXT = pl.DataFrame(
    {
        "nct_id": ["NCT1", "NCT2", "NCT3"],
        "split": ["dev", "test", "late"],
        "title": ["A study of Zorbex", "Another study", None],
        "conditions": ["asthma", None, None],
        "interventions": ["Zorbex; Placebo", "Quenlimab", None],
        "primary_outcomes": ["FEV1", "Survival", None],
        "eligibility": ["adults only", "none", None],
        "summary": ["the study met its primary endpoint", "s", None],
    }
)


def fake_encoder(texts):
    """A stand-in for the model: a fixed vector from each text's length, no randomness."""
    return np.array([[len(t), len(t) % 7, 1.0] for t in texts])


def test_document_labels_each_field_and_leaves_missing_ones_out():
    row = TEXT.row(0, named=True)
    assert emb.document(row) == (
        "Title: A study of Zorbex\nConditions: asthma\n"
        "Interventions: Zorbex; Placebo\nPrimary outcomes: FEV1"
    )
    assert emb.document(TEXT.row(1, named=True)).startswith("Title: Another study\nInterventions")
    assert emb.document(TEXT.row(2, named=True)) == ""


def test_the_summary_and_the_eligibility_text_are_never_embedded():
    documents = emb.documents(TEXT)
    assert not any("primary endpoint" in d or "adults only" in d for d in documents)


def test_every_embedded_field_is_dated_as_honest_in_the_text_table():
    assert all(text.FEATURES[field] in HONEST for field in emb.FIELD_LABELS)
    assert set(emb.FEATURES.values()) == {"protocol"}


def test_table_has_one_row_per_trial_in_order_with_ids_split_and_vector_columns():
    table = emb.embedding_table(TEXT, TINY, fake_encoder)
    assert table.columns == ["nct_id", "split", "emb_t_000", "emb_t_001", "emb_t_002"]
    assert table["nct_id"].to_list() == ["NCT1", "NCT2", "NCT3"]
    assert table["split"].to_list() == ["dev", "test", "late"]
    assert table.schema["emb_t_000"] == pl.Float32
    assert table["emb_t_002"].to_list() == [1.0, 1.0, 1.0]


def test_a_trial_s_vector_does_not_depend_on_the_other_trials():
    both = emb.embedding_table(TEXT, TINY, fake_encoder)
    alone = emb.embedding_table(TEXT.head(1), TINY, fake_encoder)
    assert alone.row(0) == both.row(0)


def test_an_encoder_of_the_wrong_width_is_refused():
    with pytest.raises(ValueError, match="Expected"):
        emb.embedding_table(TEXT, emb.EmbeddingModel("x", "n", "r", 5, "emb_x"), fake_encoder)


def test_pinned_revisions_are_full_commit_hashes_and_columns_are_unique():
    for model in emb.MODELS:
        assert len(model.revision) == 40
        assert len(set(model.columns)) == model.dimension
    assert len(emb.FEATURES) == sum(m.dimension for m in emb.MODELS)


def test_importing_the_model_registry_does_not_import_torch():
    code = (
        "import sys\n"
        "from trial_benchmark.models.registry import all_specs\n"
        "assert any(s.family == 'embeddings' for s in all_specs())\n"
        "assert 'torch' not in sys.modules and 'sentence_transformers' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def cached(model: emb.EmbeddingModel) -> bool:
    from huggingface_hub import try_to_load_from_cache

    found = try_to_load_from_cache(model.hf_name, "config.json", revision=model.revision)
    return isinstance(found, str)


def test_the_real_encoder_gives_the_same_vectors_twice_and_for_a_trial_alone():
    pytest.importorskip("sentence_transformers")
    if not cached(emb.GENERAL):
        pytest.skip("the model files are not downloaded")
    encode = emb.load_encoder(emb.GENERAL)
    texts = ["Title: asthma inhaler", "Title: cancer infusion", ""]
    first, second = encode(texts), encode(texts)
    np.testing.assert_array_equal(first, second)
    assert first.shape == (3, emb.GENERAL.dimension)
    np.testing.assert_allclose(encode(texts[:1])[0], first[0], atol=1e-5)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, atol=1e-5)
