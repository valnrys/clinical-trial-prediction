import numpy as np
import polars as pl
from synthetic import synthetic_trials

from elman_exercise import precedent
from elman_exercise.metrics import auroc
from elman_exercise.models import linear
from elman_exercise.models import precedent as models
from elman_exercise.models.registry import ModelSpec, require_unique_names


def frame_with_flags(
    n: int = 1500, seed: int = 3, signal: float = 0.3, driver: str = "precedent_today"
) -> pl.DataFrame:
    """Synthetic trials with every flag column; success leans on the flag named `driver` only."""
    rng = np.random.default_rng(seed + 1000)  # not the generator `synthetic_trials` uses
    base = synthetic_trials(n, seed=seed)
    flag = rng.random(n) < 0.35
    outcome = base["any_arm"].to_numpy() | (rng.random(n) < signal * flag)
    columns = {
        name: flag if name == driver else rng.random(n) < 0.3 for name in precedent.ALL_FLAGS
    }
    return base.with_columns(
        pl.Series("any_arm", outcome),
        *(pl.Series(name, values) for name, values in columns.items()),
    )


def test_four_specs_with_unique_names_in_the_registry_shape():
    assert len(models.SPECS) == 6
    assert all(isinstance(s, ModelSpec) for s in models.SPECS)
    require_unique_names(models.SPECS)
    require_unique_names([*linear.SPECS, *models.SPECS])  # no clash with the existing models
    assert {s.family for s in models.SPECS} == {"precedent"}
    assert all(s.tables == (precedent.TABLE,) for s in models.SPECS)


def test_names_say_which_links_and_whether_strict():
    names = [s.name for s in models.SPECS]
    assert "logistic: registration + precedent (today's links)" in names
    assert "logistic: registration + precedent (2020 links)" in names
    assert "logistic: registration + precedent (2020 names and links)" in names
    assert sum("experimental arms only" in n for n in names) == 3


def test_models_are_not_marked_honest_because_the_links_postdate_the_trials():
    assert not any(s.honest for s in models.SPECS)  # the flags are labelled "after the start"


def test_each_spec_reads_its_own_flag_and_the_flag_adds_signal():
    train, test = frame_with_flags(seed=1), frame_with_flags(seed=2)
    blind = test.drop("any_arm", "all_arms", "highest_dose", "has_dose_ladder")
    registration = linear.SPECS[0].fit(train).predict(blind)
    main = models.SPECS[0].fit(train).predict(blind)
    outcome = test["any_arm"].to_numpy()
    assert auroc(outcome, main) > auroc(outcome, registration)
    for spec in models.SPECS:
        p = spec.fit(train).predict(blind)
        assert p.shape == (test.height,) and np.all(np.isfinite(p))


def test_a_flag_that_means_nothing_adds_nothing_noticeable():
    train, test = frame_with_flags(seed=4, signal=0.0), frame_with_flags(seed=5, signal=0.0)
    blind = test.drop("any_arm", "all_arms", "highest_dose", "has_dose_ladder")
    gain = auroc(test["any_arm"].to_numpy(), models.SPECS[0].fit(train).predict(blind)) - auroc(
        test["any_arm"].to_numpy(), linear.SPECS[0].fit(train).predict(blind)
    )
    assert abs(gain) < 0.03


def test_every_spec_reads_its_own_flag_column_and_no_other():
    """Success leans on one flag at a time; only the spec for that column gains from it."""
    registration = linear.SPECS[0]
    columns = {
        spec.name: f"precedent_{'exp_' if 'experimental' in spec.name else ''}"
        f"{'2020_names' if 'names' in spec.name else '2020' if '2020' in spec.name else 'today'}"
        for spec in models.SPECS
    }
    assert sorted(columns.values()) == sorted(precedent.ALL_FLAGS)
    for driver in precedent.ALL_FLAGS:
        train = frame_with_flags(seed=11, signal=1.0, driver=driver)
        test = frame_with_flags(seed=12, signal=1.0, driver=driver)
        blind = test.drop("any_arm", "all_arms", "highest_dose", "has_dose_ladder")
        outcome = test["any_arm"].to_numpy()
        base = auroc(outcome, registration.fit(train).predict(blind))
        for spec in models.SPECS:
            gain = auroc(outcome, spec.fit(train).predict(blind)) - base
            if columns[spec.name] == driver:
                assert gain > 0.03, (spec.name, driver)
            else:
                assert gain < 0.03, (spec.name, driver)


def test_the_sensitivity_specs_read_the_phase_2_3_columns():
    names = {s.name for s in models.sensitivity_specs()}
    assert len(names) == 3 and all("with Phase 2/3" in n for n in names)
    assert not names & {s.name for s in models.SPECS}
    assert not any(s.honest for s in models.sensitivity_specs())
