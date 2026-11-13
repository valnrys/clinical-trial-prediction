"""The rules that could be wrong in each model family: shrinkage, windows, routing."""

from datetime import date

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from elman_exercise import baselines as bl
from elman_exercise.models import drift, grouped, lookup, memorisation, pooling
from elman_exercise.models.registry import all_specs
from elman_exercise.sponsor import sponsor_key


def spec_named(name: str):
    (found,) = [s for s in all_specs() if s.name == name]
    return found


# ---------------------------------------------------------------- shrinkage


def test_shrunk_rate_is_the_parent_rate_for_an_empty_cell_and_the_cell_rate_for_a_huge_one():
    assert lookup.shrunk_rate(0, 0, 0.6, 10) == 0.6
    assert lookup.shrunk_rate(900, 1000, 0.1, 10) == pytest.approx(0.9, abs=0.01)


def test_shrunk_rate_adds_strength_pretend_trials_at_the_parent_rate():
    # 3 of 4 succeeded; 10 pretend trials at 0.5 add 5 successes: (3 + 5) / (4 + 10)
    assert lookup.shrunk_rate(3, 4, 0.5, 10) == pytest.approx(8 / 14)


def test_shrunk_rates_counts_each_key_and_keeps_missing_keys_as_a_group():
    keys = pl.Series(["a", "a", "b", None])
    rates = lookup.shrunk_rates(keys, np.array([True, False, True, True]), 0.5, strength=2.0)
    assert rates["a"] == pytest.approx((1 + 1) / (2 + 2))
    assert rates["b"] == pytest.approx((1 + 1) / (1 + 2))
    assert rates[None] == pytest.approx((1 + 1) / (1 + 2))


def test_look_up_uses_the_default_per_row_for_unseen_keys():
    out = lookup.look_up(pl.Series(["a", "z"]), {"a": 0.9}, np.array([0.1, 0.2]))
    assert out.tolist() == [0.9, 0.2]


def test_a_phase_and_sponsor_cell_never_seen_falls_back_to_its_phase_rate():
    train = synthetic_trials(400, seed=3).filter(pl.col("sponsor_class") == "INDUSTRY")
    model = lookup.fit_phase_sponsor(train)
    unseen = synthetic_trials(2, seed=4).with_columns(sponsor_class=pl.lit("NIH"))
    expected = lookup.look_up(unseen["phase"], model.phase_rates, model.overall_rate)
    assert model.predict(unseen).tolist() == expected.tolist()


# ---------------------------------------------------------------- drift


def test_newest_years_keeps_the_last_n_calendar_years_of_the_table():
    train = pl.DataFrame({"primary_completion_date": [date(y, 6, 1) for y in range(2010, 2015)]})
    kept = drift.newest_years(train, 2)["primary_completion_date"].dt.year().to_list()
    assert kept == [2013, 2014]


def test_recency_weights_halve_every_half_life_and_average_one():
    weights = drift.recency_weights(np.array([2014.0, 2010.0, 2006.0]), half_life=4.0)
    assert weights.mean() == pytest.approx(1.0)
    assert (weights[1] / weights[0], weights[2] / weights[1]) == pytest.approx((0.5, 0.5))


def rising_trials(last_year: int, last_month: int = 12) -> pl.DataFrame:
    """10 trials a year from 2000, success rate rising 0.04 a year from 0.2; the last year stops short."""
    rows = []
    for year in range(2000, last_year + 1):
        wins = round(10 * (0.2 + 0.04 * (year - 2000)))
        month = last_month if year == last_year else 12
        rows += [
            {"primary_completion_date": date(year, month, 15), "any_arm": i < wins}
            for i in range(10)
        ]
    return pl.DataFrame(rows)


def test_projected_rate_follows_a_straight_line_trend_forward():
    # Training ends with 2011; two years later, one year past the end, the line gives
    # 0.2 + 0.04 * 12.5 = 0.7 at mid-2012.5 + 0.5.
    assert drift.projected_rate(rising_trials(2011), look_ahead_years=1.0) == pytest.approx(
        0.7, abs=0.03
    )


def test_the_look_ahead_is_a_parameter_and_moves_the_projection():
    train = rising_trials(2011)
    near = drift.projected_rate(train, look_ahead_years=0.0)
    far = drift.projected_rate(train, look_ahead_years=2.0)
    assert far - near == pytest.approx(0.04 * 2, abs=0.01)


def test_a_newest_year_that_stops_short_of_december_is_left_out_of_the_trend():
    # 2011 stops in March and has 10 trials all at the low early-years rate: it must not count
    short = rising_trials(2011, last_month=3)
    whole = rising_trials(2010)
    assert drift.projected_rate(short, 1.0) == pytest.approx(drift.projected_rate(whole, 1.0))


def test_the_shift_makes_the_average_prediction_equal_the_target_exactly():
    predictions = np.random.default_rng(0).uniform(0.05, 0.95, 500)
    for target in (0.15, 0.4, 0.7):
        shift = drift.shift_to_mean(predictions, target)
        shifted = drift.expit(drift.logit(predictions) + shift)
        assert shifted.mean() == pytest.approx(target, abs=1e-9)


def test_the_shift_moves_predictions_on_the_log_odds_scale():
    class Half:
        def predict(self, frame):
            return np.full(frame.height, 0.5)

    shifted = drift.ShiftedPredictor(Half(), shift=np.log(3))
    assert shifted.predict(pl.DataFrame({"a": [1, 2]})).tolist() == pytest.approx([0.75, 0.75])


def test_drift_models_ignore_the_completion_date_of_the_trials_they_predict():
    train = synthetic_trials(300, seed=1)
    later = synthetic_trials(30, seed=2, prefix="NCT1")
    moved = later.with_columns(pl.col("primary_completion_date").dt.offset_by("7y"))
    for name in ("logistic: trend-shifted intercept", "logistic: recency-weighted"):
        model = spec_named(name).fit(train)
        assert model.predict(later).tolist() == model.predict(moved).tolist()


# ---------------------------------------------------------------- memorisation and pooling


def test_sponsor_names_are_matched_without_punctuation_case_or_company_words():
    assert sponsor_key("Pfizer Inc.") == sponsor_key("PFIZER")
    assert sponsor_key("Novartis Pharmaceuticals") == "novartis pharmaceuticals"
    assert sponsor_key("Inc") == "inc"  # never reduced to nothing
    assert sponsor_key(None) is None


def test_memory_of_an_unseen_name_is_the_overall_rate_and_of_a_seen_name_is_shrunk():
    train = pl.DataFrame(
        {"lead_drug": ["d1"] * 10 + [None, "d2"], "any_arm": [True] * 10 + [False, False]}
    )
    model = memorisation.fit_memory(train, "lead_drug")
    overall = 10 / 12
    assert model.overall_rate == pytest.approx(overall)
    known_rate = 10 / 11  # the prior is the rate among trials that have a drug, not all trials
    assert model.rates["d1"] == pytest.approx(lookup.shrunk_rate(10, 10, known_rate, 10))
    assert model.rates["d2"] == pytest.approx(lookup.shrunk_rate(0, 1, known_rate, 10))
    unseen = pl.DataFrame({"lead_drug": ["new", None]})
    assert model.predict(unseen).tolist() == pytest.approx([overall, overall])


def test_memory_is_honest_only_if_the_column_it_reads_is():
    assert spec_named("memory: lead sponsor").honest  # sponsor name: protocol
    assert not spec_named("memory: lead drug").honest  # lead drug: after the start


def sponsor_trials() -> pl.DataFrame:
    """Sponsor S fails in 2000-2004 and succeeds in 2005-2009; sponsor T succeeds half the time."""
    rows = []
    for year in range(2000, 2010):
        for i in range(4):
            rows.append(("s", year, year >= 2005))
            rows.append(("t", year, i % 2 == 0))
    return pl.DataFrame(
        {
            "primary_completion_date": [date(y, 6, 1) for _, y, _ in rows],
            "sponsor_key": [k for k, _, _ in rows],
            "any_arm": [w for _, _, w in rows],
        }
    )


def effect_of_s(train: pl.DataFrame, outcomes: np.ndarray) -> pl.DataFrame:
    table = pooling.add_earlier_only_effect(train, outcomes)
    return table.filter(pl.col("sponsor_key") == "s").select(
        "primary_completion_date", "sponsor_effect"
    )


def test_a_pooled_effect_uses_only_trials_from_earlier_years():
    train = sponsor_trials()
    effect = effect_of_s(train, train["any_arm"].to_numpy())
    by_year = dict(
        effect.group_by(pl.col("primary_completion_date").dt.year(), maintain_order=True)
        .agg(pl.col("sponsor_effect").first())
        .iter_rows()
    )
    assert by_year[2000] == 0  # nothing earlier to learn from
    assert by_year[2005] < 0  # S's earlier trials all failed, whatever came later
    assert by_year[2009] > by_year[2005]  # and recovers as the successes come in


def test_a_pooled_effect_does_not_change_when_later_outcomes_change():
    train = sponsor_trials()
    outcomes = train["any_arm"].to_numpy()
    years = train["primary_completion_date"].dt.year().to_numpy()
    changed = outcomes.copy()
    changed[years >= 2007] = False
    before = pooling.add_earlier_only_effect(train, outcomes)["sponsor_effect"].to_numpy()
    after = pooling.add_earlier_only_effect(train, changed)["sponsor_effect"].to_numpy()
    np.testing.assert_array_equal(before[years <= 2007], after[years <= 2007])


# ---------------------------------------------------------------- grouped


def test_each_trial_is_routed_to_its_group_and_an_unknown_group_to_the_fallback():
    class Says:
        def __init__(self, value):
            self.value = value

        def predict(self, frame):
            return np.full(frame.height, self.value)

    model = grouped.GroupedPredictor(
        grouped.phase_group, {"phase2": Says(0.2), "phase3": Says(0.3)}, Says(0.9)
    )
    frame = pl.DataFrame({"phase": ["phase3", "phase2", None, "phase4"]})
    assert model.predict(frame).tolist() == [0.3, 0.2, 0.9, 0.9]


def test_a_group_too_small_to_fit_uses_the_overall_model():
    train = synthetic_trials(300, seed=6).with_columns(
        sponsor_class=pl.when(pl.int_range(pl.len()) < 20)
        .then(pl.lit("OTHER"))
        .otherwise(pl.lit("INDUSTRY"))
    )
    model = grouped.fit_grouped(train, grouped.company_or_other)
    assert set(model.by_group) == {"company"}


def test_company_or_other_names_the_two_groups_and_leaves_unknown_sponsors_null():
    frame = pl.DataFrame({"sponsor_class": ["INDUSTRY", "NIH", None]})
    assert grouped.company_or_other(frame).to_list() == ["company", "other", None]


# ---------------------------------------------------------------- the pre-registered baseline


def test_logistic_registration_matches_the_baselines_module_to_numerical_precision():
    train = synthetic_trials(400, seed=1)
    target = synthetic_trials(100, seed=2, prefix="NCT1")
    ours = spec_named("logistic: registration").fit(train).predict(target)
    theirs = bl.RUNGS[bl.REGISTRATION](train).predict_probability(target)
    np.testing.assert_allclose(ours, theirs, rtol=1e-12, atol=1e-14)
