import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise import dating, science
from elman_exercise.metrics import auroc
from elman_exercise.models import biology, ladder
from elman_exercise.models.registry import all_specs

BOOLEANS = {"depmap_common_essential", "expr_disease_tissue_top3"}
TEXT = {"target_class", "target_location", "chem_action_type"}
LEVELS = {
    "target_class": ["Enzyme", "Ion channel", "multiple"],
    "target_location": ["membrane", "nuclear"],
    "chem_action_type": ["inhibitor", "agonist"],
}


def science_frame(n: int, seed: int, mapped_share: float = 0.6, unmapped_failure: float = 0.0):
    """Trials with the registration columns and every science column, as the real table has them.

    Success depends a little on one science column for mapped trials. Unmapped trials have
    every science value null, as in the real table, and `has_target` False. With
    `unmapped_failure`, that share of unmapped trials fail whatever else is true: being unmapped
    then predicts failure, which is the hindsight a leaky model would pick up.
    """
    rng = np.random.default_rng([seed, 99])  # not the stream `synthetic_trials` draws from
    base = synthetic_trials(n, seed=seed)
    mapped = rng.random(n) < mapped_share
    columns = {}
    for name in biology.honest_columns():
        if name in BOOLEANS:
            values = rng.random(n) < 0.3
        elif name in TEXT:
            values = rng.choice(LEVELS[name], n)
        else:
            values = rng.normal(5.0, 2.0, n)
        dtype = pl.Boolean if name in BOOLEANS else pl.String if name in TEXT else pl.Float64
        columns[name] = (
            pl.Series(name, values.tolist(), dtype=dtype)
            .to_frame()
            .with_columns(pl.when(pl.Series(mapped)).then(pl.col(name)).otherwise(None))[name]
        )
    # The label leans on one column, so a model that reads it has something to learn.
    signal = np.nan_to_num(columns["depmap_mean_effect"].to_numpy(), nan=5.0)
    outcome = base["any_arm"].to_numpy() | (rng.random(n) < 0.5 * (signal > 6.5))
    outcome &= mapped | (rng.random(n) >= unmapped_failure)
    return base.with_columns(
        pl.Series("any_arm", outcome),
        pl.Series("has_target", mapped),
        pl.Series("n_target_genes", np.where(mapped, 2, 0)),
        *columns.values(),
    )


def clear_science(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.lit(None, frame.schema[c]).alias(c) for c in biology.honest_columns()
    )


def spec_named(name: str):
    return next(s for s in biology.SPECS if s.name == name)


def unwrap(model):
    """The scikit-learn pipeline inside a chain of predictor wrappers."""
    while not hasattr(model, "pipeline"):
        model = model.model
    return model.pipeline


SPEC_NAMES = [spec.name for spec in biology.SPECS if "ladder" not in spec.name]
SCIENCE_ONLY = [n for n in SPEC_NAMES if "registration" not in n]
WITH_REGISTRATION = [n for n in SPEC_NAMES if "registration" in n]
# Columns a trial can sit exactly at the mean of (a yes/no or a level cannot).
NUMBERS = [c for c in biology.ALL if c not in BOOLEANS and c not in TEXT]


def at_the_mean(test: pl.DataFrame, train: pl.DataFrame) -> pl.DataFrame:
    return test.with_columns(pl.lit(float(train[c].drop_nulls().mean())).alias(c) for c in NUMBERS)


def without_numbers(test: pl.DataFrame) -> pl.DataFrame:
    return test.with_columns(pl.lit(None, pl.Float64).alias(c) for c in NUMBERS)


# ---------------------------------------------------------------- no leak of "mapped"


@pytest.mark.parametrize("name", WITH_REGISTRATION)
def test_a_model_with_registration_columns_cannot_rank_mapped_trials_above_unmapped_ones(name):
    """Being mapped predicts success in this table but must not be readable from the columns."""
    train = science_frame(1500, seed=1, unmapped_failure=0.6)
    test = science_frame(1500, seed=2, unmapped_failure=0.6)
    assert train.filter(~pl.col("has_target"))["any_arm"].mean() < train["any_arm"].mean() - 0.08
    predictions = spec_named(name).fit(train).predict(test)
    mapped = test["has_target"].to_numpy()
    # Unmapped trials fail about 0.1 more often. A model that could tell them apart would
    # predict about that much less for them; a neutral fill leaves the two means within noise.
    assert abs(predictions[mapped].mean() - predictions[~mapped].mean()) < 0.03


@pytest.mark.parametrize("name", SCIENCE_ONLY)
def test_a_science_only_model_never_learns_from_the_outcomes_of_unmapped_trials(name):
    """Trees included: change what happened to every unmapped trial and nothing changes."""
    train = science_frame(900, seed=1, unmapped_failure=0.6)
    outcomes_swapped = train.with_columns(
        pl.when(pl.col("has_target")).then(pl.col("any_arm")).otherwise(True).alias("any_arm")
    )
    test = science_frame(300, seed=2)
    first = spec_named(name).fit(train).predict(test)
    second = spec_named(name).fit(outcomes_swapped).predict(test)
    assert first == pytest.approx(second)


@pytest.mark.parametrize("name", WITH_REGISTRATION)
def test_the_unmapped_trials_of_a_model_with_registration_columns_are_zero_in_every_science_column(
    name,
):
    train = science_frame(900, seed=3)
    centring = biology.fit_centring(train, biology.ALL)
    unmapped = centring.apply(clear_science(science_frame(60, seed=4)))
    encoded = [*centring.numbers, *centring.yes_no, *centring.level_names()]
    assert len(encoded) > len(biology.ALL)  # text levels became columns
    assert not unmapped.select(encoded).to_numpy().any()  # every one exactly zero
    model = spec_named(name).fit(train)
    assert np.isfinite(model.predict(clear_science(science_frame(60, seed=4)))).all()


def test_a_trial_at_the_mean_of_every_number_gets_the_prediction_of_an_unmapped_trial():
    train = science_frame(900, seed=5)
    model = biology.fit_with_registration(train, NUMBERS)
    test = science_frame(50, seed=6)
    assert model.predict(without_numbers(test)) == pytest.approx(
        model.predict(at_the_mean(test, train))
    )


def test_the_ladder_rung_with_science_treats_an_unmapped_trial_the_same_way():
    train = science_frame(900, seed=7)
    model = biology.fit_ladder(train, NUMBERS, tables=())
    test = science_frame(50, seed=8)
    assert model.predict(without_numbers(test)) == pytest.approx(
        model.predict(at_the_mean(test, train))
    )


@pytest.mark.parametrize("name", SCIENCE_ONLY)
def test_science_only_models_give_every_unmapped_trial_exactly_the_prediction_at_the_centring_point(
    name,
):
    train = science_frame(900, seed=9)
    test = science_frame(80, seed=10)
    model = spec_named(name).fit(train)
    predictions = model.predict(test)
    unmapped = ~test["has_target"].to_numpy()
    assert unmapped.any() and (~unmapped).any()
    # The centring point: every encoded column exactly zero.
    centre = model.centring.apply(clear_science(test.head(1)))
    encoded = [*model.centring.numbers, *model.centring.yes_no, *model.centring.level_names()]
    assert not centre.select(encoded).to_numpy().any()
    assert (predictions[unmapped] == model.model.predict(centre)[0]).all()  # exact, not approx
    assert len(set(np.round(predictions[~unmapped], 6))) > 1  # mapped trials get their own values


@pytest.mark.parametrize("name", [*SPEC_NAMES, "biology: ladder, all four tables + science"])
def test_no_fitted_pipeline_has_a_missing_value_indicator_for_a_science_column(name):
    train = science_frame(900, seed=11)
    if "ladder" in name:
        model = biology.fit_ladder(train, biology.ALL, tables=())
    else:
        model = spec_named(name).fit(train)
    names = unwrap(model).named_steps["encode"].get_feature_names_out()
    indicators = [n for n in names if "missingindicator" in n]
    assert not [n for n in indicators if any(c in n for c in biology.ALL)]
    # The registration columns of the synthetic table are complete, so there are none at all.
    assert indicators == []


# ---------------------------------------------------------------- the centring


def test_each_column_is_centred_on_the_mean_of_the_trials_that_have_a_value():
    train = pl.DataFrame({"x": [1.0, 3.0, 8.0, None, None], "b": [True, False, None, None, None]})
    centring = biology.fit_centring(train, ["x", "b"])
    assert centring.numbers == {"x": 4.0} and centring.yes_no == {"b": 0.5}
    out = centring.apply(train)
    assert out["x"].to_list() == [-3.0, -1.0, 4.0, 0.0, 0.0]
    assert out["b"].to_list() == [0.5, -0.5, 0.0, 0.0, 0.0]


def test_the_centred_columns_of_the_training_trials_have_mean_zero():
    train = science_frame(900, seed=12)
    centring = biology.fit_centring(train, biology.ALL)
    centred = centring.apply(train)
    for name in [*centring.numbers, *centring.yes_no, *centring.level_names()]:
        assert centred[name].mean() == pytest.approx(0.0, abs=1e-9)


def test_text_becomes_one_column_per_level_and_rare_levels_are_dropped():
    values = ["common"] * 40 + ["other"] * 40 + ["rare"] * 5 + [None] * 15
    train = pl.DataFrame({"t": values})
    centring = biology.fit_centring(train, ["t"])
    assert sorted(centring.levels["t"]) == ["common", "other"]  # `rare` has 5 trials
    assert centring.levels["t"]["common"] == pytest.approx(40 / 85)
    out = centring.apply(train)
    assert out["t=common"][0] == pytest.approx(1 - 40 / 85)
    assert out["t=common"][-1] == 0.0  # null text is zero in every level column
    assert out["t=other"][-1] == 0.0
    assert out["t=common"][80] == pytest.approx(-40 / 85)  # a rare level counts as "not common"


def test_a_column_with_no_value_in_training_is_left_out():
    train = pl.DataFrame({"x": pl.Series([None, None], dtype=pl.Float64)})
    centring = biology.fit_centring(train, ["x"])
    assert centring.numbers == {} and centring.apply(train)["x"].to_list() == [None, None]


def test_the_means_come_from_the_training_table_only():
    centring = biology.fit_centring(pl.DataFrame({"x": [1.0, 2.0, 3.0]}), ["x"])
    later = pl.DataFrame({"x": [None, 1000.0]})
    assert centring.apply(later)["x"].to_list() == [0.0, 998.0]


def test_integer_columns_are_centred_without_a_type_error():
    centring = biology.fit_centring(pl.DataFrame({"n": [1, 2, 6]}), ["n"])
    out = centring.apply(pl.DataFrame({"n": [None, 5]}, schema={"n": pl.Int64}))
    assert out["n"].to_list() == [0.0, 2.0]


# ---------------------------------------------------------------- what the models read


@pytest.mark.parametrize("name", SPEC_NAMES)
def test_models_do_not_read_has_target_or_n_target_genes(name):
    spec = spec_named(name)
    train = science_frame(900, seed=13)
    test = science_frame(80, seed=14)
    model = spec.fit(train)
    flipped = test.with_columns(~pl.col("has_target"), pl.col("n_target_genes") + 5)
    assert model.predict(flipped) == pytest.approx(model.predict(test))
    # Training on altered target flags gives the same model too.
    altered = spec.fit(train.with_columns(~pl.col("has_target"), pl.col("n_target_genes") + 5))
    assert altered.predict(test) == pytest.approx(model.predict(test))


def test_science_only_models_ignore_the_registration_columns():
    spec = spec_named("biology: logistic, all sources")
    train = science_frame(900, seed=15)
    test = science_frame(50, seed=16)
    scrambled = test.with_columns(pl.col("phase").reverse(), pl.col("sponsor_class").reverse())
    model = spec.fit(train)
    assert model.predict(scrambled) == pytest.approx(model.predict(test))


def test_a_model_that_reads_the_science_signal_beats_chance_on_mapped_trials():
    model = spec_named("biology: logistic, DepMap only").fit(science_frame(3000, seed=17))
    test = science_frame(3000, seed=18).filter(pl.col("has_target"))
    assert auroc(test["any_arm"].to_numpy(), model.predict(test)) > 0.52


def test_boosted_trees_are_fitted_on_trials_with_a_value_only():
    train = science_frame(900, seed=19)
    model = spec_named("biology: boosted trees, all sources").fit(train)
    valued = train.filter(biology.has_value(train, biology.ALL))
    assert 0 < valued.height < train.height
    assert model.model.pipeline.named_steps["fit"].n_iter_ > 0


# ---------------------------------------------------------------- the specs


def test_honesty_is_worked_out_from_the_dating_of_the_columns_read():
    for spec in biology.SPECS:
        assert spec.honest
    assert not biology.is_honest(["depmap_mean_effect", "has_target"])
    read = biology.honest_columns()
    assert "has_target" not in read and "n_target_genes" not in read
    assert all(science.FEATURES[c] in dating.HONEST for c in read)


def test_the_biology_family_is_registered_with_unique_names_and_the_science_table():
    specs = [s for s in all_specs() if s.family == "biology"]
    assert [s.name for s in specs] == [s.name for s in biology.SPECS]
    assert len(specs) == 14
    assert all("science" in s.tables for s in specs)
    assert set(ladder.ALL_FOUR) <= set(next(s for s in specs if "ladder" in s.name).tables)


def test_science_only_specs_say_they_are_defined_on_mapped_trials_only():
    for spec in biology.SPECS:
        science_only = spec.name in SCIENCE_ONLY
        assert ("mapped target" in spec.description) == science_only


def test_each_source_has_its_own_model():
    sources = {
        n for s in biology.SPECS for n in biology.SOURCE_LABELS.values() if f", {n} only" in s.name
    }
    assert sources == set(biology.SOURCE_LABELS.values())
    for source in science.SOURCES:
        assert biology.honest_columns((source,)) == [
            c for c, label in science.SOURCES[source].FEATURES.items() if label in dating.HONEST
        ]
