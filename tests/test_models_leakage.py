"""No model may read the label, or the completion date, of the trials it predicts."""

import importlib

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from trial_benchmark import baselines as bl
from trial_benchmark import dataset as ds
from trial_benchmark.models import ladder
from trial_benchmark.models.registry import ModelSpec, all_specs

N_TRAIN, N_PREDICT = 500, 120


def table_columns(table: str, n: int, rng: np.random.Generator) -> dict:
    """Made-up columns for a feature table, shaped by the table's own column list."""
    if table == "sponsor":
        return {"sponsor_key": rng.choice([f"sponsor {i}" for i in range(15)], n).tolist()}
    columns = {}
    for name in ladder.table_features(table):
        kind = genetics_kind(name) if table == "genetics" else None
        columns[name] = kind(rng, n) if kind else rng.normal(size=n)
    if table == "genetics":
        columns["lead_drug"] = rng.choice([f"drug {i}" for i in range(40)], n).tolist()
    return columns


def genetics_kind(name: str):
    if name == "drug_type":
        return lambda rng, n: rng.choice(["small molecule", "antibody"], n).tolist()
    if name in {"mapped", "few_targets", "investigational_mapped"} or name.startswith("supported"):
        return lambda rng, n: rng.random(n) < 0.5
    return None


def synthetic_frame(n: int, seed: int, prefix: str, tables: set[str]) -> pl.DataFrame:
    frame = synthetic_trials(n, seed=seed, prefix=prefix)
    rng = np.random.default_rng(seed)
    for table in sorted(tables | {"genetics"}):
        frame = frame.with_columns(pl.DataFrame(table_columns(table, n, rng)))
    return frame


def buildable(spec: ModelSpec) -> bool:
    """Whether made-up columns for this spec's tables can be written from their column lists."""
    if spec.own_tables:
        # Stored lookups (the language models) read no frame column, and their table is git-ignored,
        # so a clean checkout has none to fit. `tests/test_models_llm.py` covers them.
        return False
    for table in spec.tables:
        try:
            importlib.import_module(f"trial_benchmark.{table}")
        except ImportError:
            return False
        if table not in {"sponsor", "genetics"} and not hasattr(
            importlib.import_module(f"trial_benchmark.{table}"), "FEATURES"
        ):
            return False
    # Text features are not plain numbers; the text tests cover them.
    return spec.family != "text" and "text" not in spec.tables


SPECS = [s for s in all_specs() if buildable(s)]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: s.name)
def test_predictions_do_not_change_with_the_label_or_the_completion_date_of_the_predicted_trials(
    spec,
):
    tables = set(spec.tables)
    train = synthetic_frame(N_TRAIN, 1, "NCT0", tables)
    target = synthetic_frame(N_PREDICT, 2, "NCT1", tables)
    model = spec.fit(train)
    base = model.predict(target)

    assert base.shape == (N_PREDICT,)
    assert np.all(np.isfinite(base)) and base.min() >= 0 and base.max() <= 1

    flipped = target.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    no_labels = target.drop([c for c in ds.LABEL_COLUMNS if c in target.columns])
    later = target.with_columns(pl.col("primary_completion_date").dt.offset_by("6y"))
    for changed in (flipped, no_labels, later):
        np.testing.assert_allclose(model.predict(changed), base, rtol=1e-9)


def test_the_leaky_models_do_read_their_extra_columns_so_the_test_can_tell_the_difference():
    spec = next(s for s in all_specs() if s.name == "leaky: registration + post-hoc fields")
    train = synthetic_frame(N_TRAIN, 1, "NCT0", set())
    target = synthetic_frame(N_PREDICT, 2, "NCT1", set())
    model = spec.fit(train)
    changed = target.with_columns(pl.col("enrollment") * 50)
    assert not np.allclose(model.predict(changed), model.predict(target))
    assert not spec.honest
    assert bl.LEAKY in bl.FEATURES
