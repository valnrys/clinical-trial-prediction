import warnings
from datetime import date

import numpy as np
import polars as pl
import pytest
from sklearn.exceptions import ConvergenceWarning
from synthetic import synthetic_trials

from elman_exercise import baselines as bl
from elman_exercise import dataset as ds


def tiny_table(rows):
    """Rows with only the named columns set; the rest are null."""
    base = {name: None for name in ds.SCHEMA}
    base.update(split="dev", primary_completion_date=date(2019, 1, 1))
    return pl.DataFrame([{**base, **row} for row in rows], schema=ds.SCHEMA)


def test_the_constant_model_says_the_training_success_rate_for_every_trial():
    train = tiny_table([{"any_arm": True}] * 3 + [{"any_arm": False}])
    model = bl.fit_constant(train)
    assert model.rate == 0.75
    assert model.predict_probability(tiny_table([{}, {}])).tolist() == [0.75, 0.75]


def test_the_phase_lookup_says_the_success_rate_of_the_trials_phase():
    train = tiny_table(
        [{"phase": "phase2", "any_arm": True}, {"phase": "phase2", "any_arm": False}]
        + [{"phase": "phase3", "any_arm": True}] * 3
        + [{"phase": "phase3", "any_arm": False}]
    )
    model = bl.fit_phase_lookup(train)
    predicted = model.predict_probability(tiny_table([{"phase": "phase2"}, {"phase": "phase3"}]))
    assert predicted.tolist() == [0.5, 0.75]


def test_the_phase_lookup_falls_back_to_the_overall_rate_for_an_unseen_or_missing_phase():
    train = tiny_table(
        [{"phase": "phase2", "any_arm": True}, {"phase": "phase3", "any_arm": False}] * 2
        + [{"phase": "phase3", "any_arm": False}] * 2
    )
    model = bl.fit_phase_lookup(train)
    predicted = model.predict_probability(tiny_table([{"phase": "phase1"}, {"phase": None}]))
    assert predicted.tolist() == pytest.approx([1 / 3, 1 / 3])  # 2 successes in 6 trials


def test_training_refuses_trials_with_no_label():
    with pytest.raises(ValueError, match="no any_arm label"):
        bl.fit_constant(tiny_table([{"any_arm": True}, {"any_arm": None}]))


def test_the_registration_model_ranks_the_group_that_succeeds_more_higher():
    data = synthetic_trials(600, seed=1)
    model = bl.RUNGS[bl.REGISTRATION](data)
    predicted = model.predict_probability(data)
    industry = (data["sponsor_class"] == "INDUSTRY").to_numpy()
    assert predicted[industry].mean() > predicted[~industry].mean() + 0.1


def logistic_inputs(rung):
    """The column names a logistic rung reads, from what it actually passes to scikit-learn."""
    features = bl.FEATURES[rung]
    return set(bl.prepare(synthetic_trials(5), features).columns)


def test_the_registration_model_reads_exactly_the_design_columns_less_the_exclusions():
    expected = set(ds.columns_in_group("design")) - set(bl.REGISTRATION_EXCLUDED)
    assert set(bl.REGISTRATION_COLUMNS) == expected
    assert logistic_inputs(bl.REGISTRATION) == expected


def test_start_year_is_excluded_and_no_rung_receives_it():
    assert bl.REGISTRATION_EXCLUDED == ("start_year",)
    data = synthetic_trials(300, seed=11)
    assert "start_year" in data.columns
    for rung in (bl.REGISTRATION, bl.NO_MODALITY, bl.LEAKY, bl.RESULTS_STRUCTURE):
        model = bl.RUNGS[rung](data)
        assert "start_year" not in logistic_inputs(rung)
        assert not any("start_year" in n for n in model.feature_names())
    with pytest.raises(ValueError, match="Not registration columns"):
        bl.registration_features_without("start_year")


def test_every_logistic_rung_is_built_on_the_one_list_of_registration_columns():
    base = set(bl.REGISTRATION_COLUMNS)
    assert set(bl.FEATURES[bl.REGISTRATION].columns) == base
    assert set(bl.FEATURES[bl.NO_MODALITY].columns) == base - {"modality", "is_vaccine"}
    assert base <= set(bl.FEATURES[bl.LEAKY].columns)
    assert base <= set(bl.FEATURES[bl.RESULTS_STRUCTURE].columns)
    without = bl.registration_features_without("has_dmc")
    assert set(without.columns) == base - {"has_dmc"}
    with pytest.raises(ValueError, match="Not registration columns"):
        bl.registration_features_without("enrollment")


def test_the_registration_model_cannot_see_post_hoc_results_structure_or_label_columns():
    seen = logistic_inputs(bl.REGISTRATION)
    forbidden = {
        *ds.columns_in_group("post_hoc"),
        *ds.columns_in_group("results_structure"),
        *ds.columns_in_group("labels"),
        *ds.columns_in_group("identifier"),
    }
    assert seen.isdisjoint(forbidden)


def test_changing_post_hoc_results_and_label_columns_leaves_registration_predictions_unchanged():
    data = synthetic_trials(300, seed=2)
    model = bl.RUNGS[bl.REGISTRATION](data)
    altered = data.with_columns(
        enrollment=pl.lit(5),
        overall_status=pl.lit("TERMINATED"),
        control_kind=pl.lit("active"),
        n_arms=pl.lit(9),
        any_arm=~pl.col("any_arm"),
    )
    assert np.array_equal(model.predict_probability(data), model.predict_probability(altered))
    # And the fitted coefficients never name them.
    names = " ".join(model.feature_names())
    for leaked in ("enrollment", "overall_status", "control_kind", "n_arms", "any_arm"):
        assert leaked not in names


def test_the_no_modality_model_drops_modality_and_the_vaccine_flag_and_nothing_else():
    registration = logistic_inputs(bl.REGISTRATION)
    assert registration - logistic_inputs(bl.NO_MODALITY) == {"modality", "is_vaccine"}
    assert logistic_inputs(bl.NO_MODALITY) < registration


def test_the_leaky_model_adds_the_post_hoc_group_to_registration():
    extra = logistic_inputs(bl.LEAKY) - logistic_inputs(bl.REGISTRATION)
    assert extra == set(ds.columns_in_group("post_hoc"))


def test_the_results_structure_model_adds_only_that_group_to_registration():
    extra = logistic_inputs(bl.RESULTS_STRUCTURE) - logistic_inputs(bl.REGISTRATION)
    assert extra == set(ds.columns_in_group("results_structure"))


def test_the_leaky_model_picks_up_a_leak_the_registration_model_cannot():
    train = synthetic_trials(1500, seed=3)  # terminated trials rarely succeed
    unseen = synthetic_trials(500, seed=4)
    stopped = (unseen["overall_status"] == "TERMINATED").to_numpy()
    leaky = bl.RUNGS[bl.LEAKY](train).predict_probability(unseen)
    plain = bl.RUNGS[bl.REGISTRATION](train).predict_probability(unseen)
    assert leaky[stopped].mean() < plain[stopped].mean() - 0.05


def test_enrollment_is_modelled_as_log_of_one_plus_the_count():
    table = tiny_table([{"enrollment": 0}, {"enrollment": 99}, {"enrollment": None}])
    prepared = bl.prepare(table, bl.FEATURES[bl.LEAKY])
    assert prepared["enrollment"].to_list()[:2] == pytest.approx([0.0, np.log(100)])
    assert prepared["enrollment"][2] is None


def test_unknown_categories_and_nulls_at_prediction_time_do_not_crash():
    train = synthetic_trials(300, seed=5)
    odd = tiny_table(
        [
            {"phase": "phase9", "sponsor_class": "SPACE AGENCY", "modality": "robot"},
            {},  # everything null
        ]
    )
    for rung in (bl.REGISTRATION, bl.NO_MODALITY, bl.LEAKY, bl.RESULTS_STRUCTURE):
        predicted = bl.RUNGS[rung](train).predict_probability(odd)
        assert predicted.shape == (2,)
        assert np.all((predicted > 0) & (predicted < 1))


def test_nulls_in_training_get_a_missing_category_and_a_missing_indicator():
    train = synthetic_trials(300, seed=6).with_columns(
        sponsor_class=pl.when(pl.int_range(pl.len()) % 7 == 0)
        .then(None)
        .otherwise(pl.col("sponsor_class")),
        n_registered_arms=pl.when(pl.int_range(pl.len()) % 5 == 0)
        .then(None)
        .otherwise(pl.col("n_registered_arms")),
        has_dmc=pl.when(pl.int_range(pl.len()) % 3 == 0).then(None).otherwise(pl.col("has_dmc")),
    )
    names = bl.RUNGS[bl.REGISTRATION](train).feature_names()
    assert "sponsor_class=(missing)" in names
    assert "n_registered_arms is missing" in names
    assert "has_dmc is missing" in names


def test_every_rung_fits_without_a_convergence_warning():
    data = synthetic_trials(800, seed=7)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        models = bl.fit_ladder(data)
    for model in models.values():
        if isinstance(model, bl.LogisticModel):
            assert model.iterations() < bl.MAX_ITERATIONS


def test_fitting_twice_gives_identical_predictions():
    data = synthetic_trials(300, seed=8)
    first = bl.RUNGS[bl.LEAKY](data).predict_probability(data)
    second = bl.RUNGS[bl.LEAKY](data).predict_probability(data)
    assert np.array_equal(first, second)


def test_missing_indicators_are_not_scaled_but_filled_numbers_are():
    train = synthetic_trials(400, seed=9).with_columns(
        n_registered_arms=pl.when(pl.int_range(pl.len()) % 40 == 0)
        .then(None)
        .otherwise(pl.col("n_registered_arms"))
    )
    model = bl.RUNGS[bl.REGISTRATION](train)
    encoded = model.pipeline[:-1].transform(train)
    names = model.pipeline.named_steps["encode"].get_feature_names_out()
    flag = encoded[:, list(names).index("numeric_missing__missingindicator_n_registered_arms")]
    assert set(np.unique(flag)) == {0.0, 1.0}  # a rare flag stays 0/1, not stretched
    year = encoded[:, list(names).index("numeric__n_registered_arms")]
    assert year.mean() == pytest.approx(0, abs=1e-9) and year.std() == pytest.approx(1)


def test_columns_are_sorted_into_kinds_by_their_schema_type():
    features = bl.features_from(("phase", "has_dmc", "start_year", "enrollment"))
    assert features.categorical == ("phase",)
    assert features.boolean == ("has_dmc",)
    assert features.numeric == ("start_year", "enrollment")
    with pytest.raises(ValueError, match="No way to model"):
        bl.features_from(("primary_completion_date",))
