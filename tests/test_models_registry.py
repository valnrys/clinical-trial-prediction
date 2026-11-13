import pkgutil
from datetime import date

import numpy as np
import polars as pl
import pytest

from trial_benchmark import models
from trial_benchmark.models.registry import (
    FAMILY_MODULES,
    ModelSpec,
    all_specs,
    best_by_inner_split,
    inner_split,
    require_unique_names,
)


def spec(name: str) -> ModelSpec:
    return ModelSpec(name, "test", "A test model.", fit=lambda frame: None)


def test_every_family_module_with_specs_is_discovered():
    families = {s.family for s in all_specs()}
    assert {"lookup", "linear", "trees", "neural", "drift", "memorisation"} <= families
    assert {"pooling", "grouped", "ladder", "leaky", "text"} <= families


def test_every_module_in_the_package_is_listed_or_is_the_registry():
    on_disk = {m.name for m in pkgutil.iter_modules(models.__path__)}
    assert on_disk == set(FAMILY_MODULES) | {"registry"}


def test_model_names_are_unique_across_all_families():
    names = [s.name for s in all_specs()]
    assert len(names) == len(set(names))


def test_the_registration_logistic_is_there_and_honest():
    (registration,) = [s for s in all_specs() if s.name == "logistic: registration"]
    assert registration.honest
    assert registration.tables == ()


def test_the_leaky_family_and_the_after_the_start_rung_are_not_honest():
    not_honest = {s.name for s in all_specs() if not s.honest}
    assert {n for n in not_honest if n.startswith("leaky:")} == {
        "leaky: registration + post-hoc fields",
        "leaky: registration + results structure",
    }
    assert any("after-the-start" in n for n in not_honest)
    assert "text: words with summary (leaky)" in not_honest


def test_the_honest_ladder_rungs_and_text_models_are_honest():
    honest = {s.name for s in all_specs() if s.honest}
    assert "ladder: registration + endpoint" in honest
    assert "ladder: registration + words" in honest
    assert "text: words without drug names" in honest


def test_a_repeated_name_is_refused():
    with pytest.raises(ValueError, match="duplicate model names"):
        require_unique_names([spec("a"), spec("b"), spec("a")])


def trials_on(days: list[int | None]) -> pl.DataFrame:
    dates = [None if d is None else date(2015, 1, 1 + d) for d in days]
    return pl.DataFrame(
        {"primary_completion_date": dates, "any_arm": [i % 2 == 0 for i in range(len(days))]},
        schema={"primary_completion_date": pl.Date, "any_arm": pl.Boolean},
    )


def test_the_inner_split_never_separates_trials_that_share_the_boundary_date():
    # 14 trials; three quarters is index 10, inside a run of five trials on day 8.
    older, newer = inner_split(trials_on([0, 1, 2, 3, 4, 5, 6, 7, 8, 8, 8, 8, 8, 9]))
    assert newer.height >= 6
    assert older["primary_completion_date"].max() < newer["primary_completion_date"].min()


def test_the_inner_split_leaves_out_trials_with_no_completion_date():
    older, newer = inner_split(trials_on([None, 0, 1, 2, 3, 4, 5, 6, 7, None]))
    assert older.height + newer.height == 8
    assert older["primary_completion_date"].null_count() == 0
    assert newer["primary_completion_date"].null_count() == 0


def test_the_inner_split_refuses_a_table_with_no_dated_trial():
    with pytest.raises(ValueError, match="completion date"):
        inner_split(trials_on([None, None, None]))


def test_the_inner_split_refuses_trials_that_all_share_one_date():
    with pytest.raises(ValueError, match="span enough dates"):
        inner_split(trials_on([3] * 10))


class Scores:
    """A predictor whose scores are set up so one candidate ranks perfectly."""

    def __init__(self, good: bool):
        self.good = good

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        # The outcome of `trials_on` is "even position", which on dates 0, 1, 2... is the day of
        # the month. A candidate works it out from the feature: it never sees the label.
        assert "any_arm" not in frame.columns
        outcomes = (frame["primary_completion_date"].dt.day() % 2 == 1).to_numpy().astype(float)
        return outcomes if self.good else np.zeros(frame.height)


def test_the_candidate_that_ranks_best_on_the_newest_part_is_chosen():
    train = trials_on(list(range(20)))
    pick = best_by_inner_split(train, [False, True], lambda table, good: Scores(good))
    assert pick is True


def test_the_candidates_are_scored_on_a_table_without_the_label_columns():
    train = trials_on(list(range(20))).with_columns(all_arms=pl.lit(True))
    seen = []

    class Looks:
        def predict(self, frame):
            seen.append(set(frame.columns))
            return np.zeros(frame.height)

    best_by_inner_split(train, [1], lambda table, c: Looks())
    assert seen and not ({"any_arm", "all_arms"} & seen[0])


def test_choosing_a_candidate_raises_if_no_score_is_defined():
    single_class = trials_on(list(range(20))).with_columns(pl.lit(True).alias("any_arm"))
    with pytest.raises(ValueError, match="No candidate"):
        best_by_inner_split(single_class, [1, 2], lambda table, c: Scores(True))


def test_slow_families_are_left_out_unless_asked_for():
    from trial_benchmark.models.registry import OPT_IN_FAMILIES

    plain = {s.family for s in all_specs()}
    assert not plain & set(OPT_IN_FAMILIES)
    assert "precedent" in plain and "whathow" not in plain
    assert set(OPT_IN_FAMILIES) <= {s.family for s in all_specs(include_opt_in=True)}
    assert set(OPT_IN_FAMILIES) <= set(FAMILY_MODULES)
