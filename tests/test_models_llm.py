import numpy as np
import polars as pl
import pytest

from elman_exercise import dating, validate
from elman_exercise.llm_predictor import PROMPT_FIELDS, TABLE_SCHEMA
from elman_exercise.models import llm
from elman_exercise.models.registry import all_specs


def stored_table(tmp_path, monkeypatch, extra=()):
    rows = [
        {"nct_id": "A", "model": "haiku", "variant": "masked", "run": 0, "probability": 0.2},
        {"nct_id": "B", "model": "haiku", "variant": "masked", "run": 0, "probability": 0.6},
        {"nct_id": "C", "model": "haiku", "variant": "masked", "run": 0, "probability": None},
        {"nct_id": "A", "model": "haiku", "variant": "masked", "run": 1, "probability": 0.9},
        {"nct_id": "A", "model": "sonnet", "variant": "masked", "run": 0, "probability": 0.99},
        # a test and a late trial: stored, but they must not move the fallback constant
        {"nct_id": "T", "model": "haiku", "variant": "masked", "run": 0, "probability": 0.95},
        {"nct_id": "L", "model": "haiku", "variant": "masked", "run": 0, "probability": 0.95},
        *extra,
    ]
    path = tmp_path / "p.parquet"
    pl.DataFrame(rows, schema=TABLE_SCHEMA).write_parquet(path)
    monkeypatch.setattr(llm, "PREDICTIONS_PATH", path)
    sample = pl.DataFrame(
        {"nct_id": ["A", "B", "C", "T", "L"], "split": ["dev", "dev", "dev", "test", "late"]}
    )
    monkeypatch.setattr(llm, "read_sample", lambda: sample)


def test_there_is_one_spec_per_model_and_variant_and_all_are_pretrained_and_honest():
    specs = [s for s in all_specs() if s.family == "llm"]
    assert len(specs) == 6  # two models, three prompts
    assert all(s.pretrained and s.honest and s.tables == () for s in specs)


def test_the_specs_are_not_ranked_and_say_why():
    specs = [s for s in all_specs() if s.family == "llm"]
    assert all(not s.ranked and "dev_llm.md" in s.not_ranked_reason for s in specs)


def test_without_the_predictions_table_the_harness_skips_the_specs_and_names_the_table(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(validate, "INTERIM", tmp_path)
    specs = [s for s in all_specs() if s.family == "llm"]
    usable, missing = validate.split_specs(specs)
    assert usable == [] and set(missing) == {s.name for s in specs}
    assert set(map(tuple, missing.values())) == {("llm_predictions",)}
    (tmp_path / "llm_predictions.parquet").write_bytes(b"")
    assert validate.split_specs(specs)[0] == specs
    # The table has several rows per trial, so the harness must not join it to the frame.
    assert {t for s in specs for t in s.tables} == set()


def test_honesty_comes_from_the_datings_of_the_prompt_fields():
    assert all(PROMPT_FIELDS[f] in dating.HONEST for f in llm.fields_read("identified"))
    assert all(PROMPT_FIELDS[f] in dating.HONEST for f in llm.fields_read("blinded"))
    shown = {v: set(llm.fields_read(v)) for v in ("identified", "masked", "blinded")}
    assert shown["identified"] - shown["masked"] == {"registry_id", "title", "sponsor_name"}
    assert shown["masked"] - shown["blinded"] == {"start_date"}


def test_a_spec_returns_the_stored_first_run_probability_and_a_constant_elsewhere(
    tmp_path, monkeypatch
):
    stored_table(tmp_path, monkeypatch)
    spec = llm.llm_spec("haiku", "masked")
    model = spec.fit(pl.DataFrame({"nct_id": []}))
    p = model.predict(pl.DataFrame({"nct_id": ["B", "A", "Z", "C"]}))
    # A and B use run 0 (not the repeat); C had a failed prediction; Z was never sampled.
    assert p[0] == 0.6 and p[1] == 0.2
    # the model's own mean over its sampled development trials: not pulled up by test and late
    assert p[2] == p[3] == pytest.approx(0.4)


def test_a_spec_with_nothing_stored_says_so(tmp_path, monkeypatch):
    stored_table(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No stored predictions"):
        llm.llm_spec("sonnet", "identified").fit(pl.DataFrame({"nct_id": []}))
    assert np.isfinite(
        llm.StoredProbabilities({"A": 0.5}, 0.5).predict(pl.DataFrame({"nct_id": ["Q"]}))
    ).all()


def test_the_constant_needs_development_predictions_and_a_repeated_trial_is_refused(
    tmp_path, monkeypatch
):
    stored_table(tmp_path, monkeypatch)
    only_late = pl.DataFrame({"nct_id": ["L"], "split": ["late"]})
    monkeypatch.setattr(llm, "read_sample", lambda: only_late)
    with pytest.raises(ValueError, match="development"):
        llm.llm_spec("haiku", "masked").fit(pl.DataFrame({"nct_id": []}))
    twice = {"nct_id": "A", "model": "haiku", "variant": "masked", "run": 0, "probability": 0.5}
    stored_table(tmp_path, monkeypatch, extra=[twice])
    with pytest.raises(ValueError, match="twice"):
        llm.llm_spec("haiku", "masked").fit(pl.DataFrame({"nct_id": []}))


def test_the_descriptions_say_recall_is_not_bounded():
    specs = [s for s in all_specs() if s.family == "llm"]
    assert all("Recall is not bounded" in s.description for s in specs)


def test_a_spec_cannot_return_a_stored_test_or_late_prediction(tmp_path, monkeypatch):
    stored_table(tmp_path, monkeypatch)
    model = llm.llm_spec("haiku", "masked").fit(pl.DataFrame({"nct_id": []}))
    assert set(model.probabilities) == {"A", "B"}  # T and L are stored but not kept
    p = model.predict(pl.DataFrame({"nct_id": ["T", "L"]}))
    assert list(p) == [pytest.approx(0.4)] * 2  # the development mean, not their stored 0.95
