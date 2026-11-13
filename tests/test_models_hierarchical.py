"""The hierarchical model on made-up trials: it finds a planted effect, shrinks thin groups, and
never lets a scored trial's outcome change its own prediction."""

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

pytest.importorskip("numpyro")

from trial_benchmark.models import hierarchical as hm

SAMPLER = hm.Sampler(warmup=250, samples=250, chains=2, seed=3)
BIG, SMALL, TINY = "big sponsor", "small sponsor", "tiny sponsor"


def planted_frame(seed: int = 0) -> pl.DataFrame:
    """1,000 trials. Sponsors differ in true effect; one sponsor has only two trials."""
    n = 1000
    rng = np.random.default_rng(seed)
    frame = synthetic_trials(n, seed=seed).drop("any_arm", "all_arms")
    sponsors = rng.choice([f"sponsor {i}" for i in range(20)], n).tolist()
    sponsors[0], sponsors[1] = TINY, TINY
    sponsors[2:122] = [BIG] * 120
    sponsors[122:182] = [SMALL] * 60
    truth = {BIG: 1.8, SMALL: -1.8, TINY: 1.8}
    logit = np.array([-0.3 + truth.get(s, 0.0) for s in sponsors])
    success = rng.random(n) < 1 / (1 + np.exp(-logit))
    success[:2] = True
    return frame.with_columns(
        pl.Series("sponsor_key", sponsors),
        pl.Series("lead_drug", [f"drug {i % 300}" for i in range(n)]),
        pl.Series("condition", rng.choice(["c1", "c2", "c3"], n).tolist()),
        pl.Series("target_ids", [["T1"] if i % 2 else [] for i in range(n)]),
        pl.Series("any_arm", success),
        pl.Series("all_arms", success),
    )


@pytest.fixture(scope="module")
def fitted():
    train = planted_frame()
    return train, hm.fit(train, hm.VARIANT["sponsor"], SAMPLER)


def sponsor_effect(model: hm.FittedModel, name: str) -> float:
    """Posterior mean log-odds effect of one sponsor."""
    level = model.design.levels["sponsor"][name]
    z = model.posterior["z_sponsor"][:, level]
    return float((model.posterior["s_sponsor"] * z).mean())


def test_it_recovers_a_planted_group_effect(fitted):
    _, model = fitted
    assert sponsor_effect(model, BIG) == pytest.approx(1.8, abs=0.5)
    assert sponsor_effect(model, SMALL) == pytest.approx(-1.8, abs=0.5)
    assert model.diagnostics["divergences"] < 25
    assert model.diagnostics["max_rhat"] < 1.1


def test_a_group_with_two_trials_is_pulled_towards_the_average(fitted):
    _, model = fitted
    tiny, big = sponsor_effect(model, TINY), sponsor_effect(model, BIG)
    assert 0 < tiny < 0.5 * big  # two wins, same true effect as BIG, far less of it believed


def test_a_level_never_seen_in_training_gets_the_population_prediction(fitted):
    _train, model = fitted
    scored = planted_frame(seed=1).head(60)
    novel = scored.with_columns(pl.lit("never seen").alias("sponsor_key"))
    blank = scored.with_columns(pl.lit(None, pl.String).alias("sponsor_key"))
    assert "never seen" not in model.design.levels["sponsor"]
    assert model.predict(novel).mean() == pytest.approx(model.predict(blank).mean(), abs=0.03)
    # ... and that is the population prediction, not the prediction of any real sponsor
    big = scored.with_columns(pl.lit(BIG).alias("sponsor_key"))
    small = scored.with_columns(pl.lit(SMALL).alias("sponsor_key"))
    assert model.predict(big).mean() - model.predict(novel).mean() > 0.2
    assert model.predict(novel).mean() - model.predict(small).mean() > 0.2


def test_a_scored_trials_outcome_cannot_change_its_own_prediction(fitted):
    """Fit on the training trials only; flip every scored outcome; predictions must not move."""
    _train, model = fitted
    scored = planted_frame(seed=2).head(80)
    flipped = scored.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    assert np.array_equal(model.predict(scored), model.predict(flipped))
    assert np.array_equal(model.predict(scored), model.predict(scored.drop("any_arm", "all_arms")))


def test_levels_and_encoding_come_from_the_training_trials_only(fitted):
    train, model = fitted
    scored = (
        planted_frame(seed=4).head(50).with_columns(pl.lit("only in scored").alias("sponsor_key"))
    )
    assert "only in scored" not in model.design.levels["sponsor"]
    assert set(model.design.levels["sponsor"]) == set(train["sponsor_key"])
    assert encode_new(model, scored)


def encode_new(model, frame) -> bool:
    data = hm.encode(frame, model.design)
    return bool(data["sponsor_new"].all())


def test_a_fit_is_stored_and_reused_by_variant_and_training_trials(tmp_path):
    train = planted_frame().head(300)
    spec = hm.VARIANT["registration"]
    one = hm.fit_cached(train, spec, "today", cache_dir=tmp_path, sampler=SAMPLER)
    two = hm.fit_cached(train, spec, "today", cache_dir=tmp_path, sampler=SAMPLER, allow_fit=False)
    assert np.array_equal(one.posterior["a0"], two.posterior["a0"])
    assert len(list(tmp_path.glob("*.pkl"))) == 1
    with pytest.raises(FileNotFoundError):  # other training trials, other fit
        hm.fit_cached(train.head(200), spec, "today", cache_dir=tmp_path, allow_fit=False)
    with pytest.raises(FileNotFoundError):  # other link version, other fit
        hm.fit_cached(train, spec, "2020", cache_dir=tmp_path, allow_fit=False)


def test_the_registry_specs_name_every_variant_once_and_are_honest():
    names = [s.name for s in hm.SPECS]
    assert len(names) == len(set(names)) == len(hm.VARIANTS)
    assert all(s.family == "hierarchical" for s in hm.SPECS)
    honest = {s.name for s in hm.SPECS if s.honest}
    assert honest == {
        f"hierarchical: {t}" for t in ("registration only", "+ condition", "+ sponsor")
    }
    assert not {s.name for s in hm.SPECS} & {s.name for s in hm.make_specs("2020")}


def test_a_spec_fits_and_predicts_through_the_registry_interface(tmp_path, monkeypatch):
    monkeypatch.setattr(hm, "CACHE_DIR", tmp_path)
    frame = planted_frame().head(400)
    specs = hm.make_specs("today", provider=lambda f: f, source="test", sampler=SAMPLER)
    spec = next(s for s in specs if s.name == "hierarchical: registration only")
    p = spec.fit(frame.head(300)).predict(frame.tail(100).drop("any_arm", "all_arms"))
    assert p.shape == (100,) and np.all((p > 0) & (p < 1))


def test_the_precedent_and_genetics_variant_reads_those_columns(tmp_path):
    frame = (
        planted_frame()
        .head(400)
        .with_columns(
            pl.Series("precedent", np.arange(400) % 3 == 0),
            pl.Series("genetic_score_at_start", np.linspace(0, 1, 400)),
            pl.Series("supported_at_start", np.arange(400) % 5 == 0),
        )
    )
    model = hm.fit(frame, hm.VARIANT["full_pg"], hm.Sampler(warmup=100, samples=100, chains=2))
    assert model.design.coef_names == ("genetic_score_at_start", "supported_at_start", "precedent")
    assert model.posterior["g"].shape[1] == 3
    scored = frame.tail(50).drop("any_arm", "all_arms")
    p = model.predict(scored)
    assert p.shape == (50,) and np.all((p > 0) & (p < 1))
    flipped = scored.with_columns(~pl.col("precedent"))
    assert not np.array_equal(p, model.predict(flipped))  # the flag is read at prediction time


def test_covariate_effects_follow_a_planted_precedent_effect():
    from trial_benchmark import hierarchy_report as report

    frame = planted_frame().head(600)
    flag = np.arange(600) % 3 == 0
    rng = np.random.default_rng(9)
    success = rng.random(600) < np.where(flag, 0.85, 0.25)
    frame = frame.with_columns(
        pl.Series("precedent", flag),
        pl.Series("genetic_score_at_start", rng.random(600)),
        pl.Series("supported_at_start", rng.random(600) < 0.3),
        pl.Series("any_arm", success),
        pl.Series("all_arms", success),
    )
    model = hm.fit(frame, hm.VARIANT["full_pg"], hm.Sampler(warmup=150, samples=150, chains=2))
    effects = {e["name"]: e for e in report.covariate_effects(model, frame)}
    assert set(effects) == {"genetic_score_at_start", "supported_at_start", "precedent"}
    assert effects["precedent"]["unit"] == "yes vs no"
    assert effects["precedent"]["per100"].mean() > 1.0
    assert effects["precedent"]["odds_ratio"].mean() > 1.0


def test_the_validation_path_gives_the_same_predictions_whatever_the_scored_outcomes_are(
    tmp_path, monkeypatch
):
    """The repo's own `predict_all` hands a spec the training trials, then the scored trials
    with labels removed. Change every scored outcome: nothing a model returns may move."""
    from trial_benchmark import validate

    monkeypatch.setattr(hm, "CACHE_DIR", tmp_path)
    frame = planted_frame().head(500)
    train, scored = frame.head(400), frame.tail(100)
    specs = hm.make_specs("today", provider=lambda f: f, source="test", sampler=SAMPLER)
    chosen = [s for s in specs if s.name.endswith(("registration only", "+ sponsor"))]
    flipped = scored.with_columns(~pl.col("any_arm"), ~pl.col("all_arms"))
    one = validate.predict_all(chosen, train, scored)
    two = validate.predict_all(chosen, train, flipped)
    assert all(np.array_equal(one[k], two[k]) for k in one)
    assert len(list(tmp_path.glob("*.pkl"))) == 2  # one fit per variant, reused for both


def test_a_cached_fit_is_keyed_by_every_input_the_fit_reads():
    spec = hm.VARIANT["full_pg"]
    train = (
        planted_frame()
        .head(300)
        .with_columns(
            pl.Series("precedent", np.arange(300) % 3 == 0),
            pl.Series("genetic_score_at_start", np.linspace(0, 1, 300)),
            pl.Series("supported_at_start", np.arange(300) % 5 == 0),
        )
    )
    key = hm.cache_key(spec, SAMPLER, train, "today", "repo")
    assert key == hm.cache_key(spec, SAMPLER, train.reverse(), "today", "repo")
    # other trials, same number of them
    other_ids = train.with_columns(pl.col("nct_id") + "x")
    assert key != hm.cache_key(spec, SAMPLER, other_ids, "today", "repo")
    # same trials, one value changed
    flip_label = train.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(~pl.col("any_arm"))
        .otherwise(pl.col("any_arm"))
        .alias("any_arm")
    )
    flip_precedent = train.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(~pl.col("precedent"))
        .otherwise(pl.col("precedent"))
        .alias("precedent")
    )
    other_targets = train.with_columns(
        pl.Series(
            "target_ids",
            [["T2"] if i == 5 else t for i, t in enumerate(train["target_ids"].to_list())],
        )
    )
    other_genetics = train.with_columns(
        pl.when(pl.int_range(pl.len()) == 5)
        .then(0.123)
        .otherwise(pl.col("genetic_score_at_start"))
        .alias("genetic_score_at_start")
    )
    for changed in (flip_label, flip_precedent, other_targets, other_genetics):
        assert key != hm.cache_key(spec, SAMPLER, changed, "today", "repo")
    # a column this variant does not read does not change its key
    registration = hm.VARIANT["registration"]
    assert hm.cache_key(registration, SAMPLER, train, "today", "repo") == hm.cache_key(
        registration, SAMPLER, flip_precedent, "today", "repo"
    )


def test_a_trial_with_one_known_and_one_unknown_target_gets_half_weight_on_each():
    train = planted_frame().head(300)
    model = hm.fit(train, hm.VARIANT["target"], SAMPLER)
    mixed = (
        planted_frame(seed=5).head(10).with_columns(pl.Series("target_ids", [["T1", "NEVER"]] * 10))
    )
    data = hm.encode(mixed, model.design)
    known = model.design.levels["tgt"]["T1"]
    assert (data["tgt_index"][:, 0] == known).all()
    assert (data["tgt_weight"][:, :2] == 0.5).all()
    assert data["tgt_new"][:, 1].all() and not data["tgt_new"][:, 0].any()
    p = model.predict(mixed)
    assert p.shape == (10,) and np.all((p > 0) & (p < 1))


def test_one_sd_in_plain_terms():
    from trial_benchmark import hierarchy_report as report

    assert report.per_100_between_one_sd(0.0, 0.3) == pytest.approx(0.0)
    # at a 50% rate, an SD of 1 in log-odds spans 73.1% down to 26.9%
    assert report.per_100_between_one_sd(1.0, 0.5) == pytest.approx(46.2, abs=0.1)


def test_a_trials_target_effect_is_the_average_of_its_targets_so_it_varies_less():
    from trial_benchmark import hierarchy_report as report

    ids = [[], ["A"], ["A", "B", "C", "D"], ["A", "B"]]
    scale, none = report.target_scale(pl.DataFrame({"target_ids": ids}))
    assert none == 0.25  # one trial in four has no target
    assert scale == pytest.approx(np.sqrt(np.mean([1.0, 1 / 4, 1 / 2])))
    assert scale < 1.0
