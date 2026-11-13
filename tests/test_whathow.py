from datetime import date, timedelta

import numpy as np
import polars as pl
import pytest
from synthetic import synthetic_trials

from trial_benchmark import whathow as wh
from trial_benchmark import whathow_report as report
from trial_benchmark.models import whathow as models

N_TARGETS, N_DISEASES = 7, 5


def what_trials(
    n: int = 300, seed: int = 0, prefix: str = "NCT0"
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Synthetic trials with the WHAT table columns, and their (nct_id, kind, key) table.

    Every trial has a start date in 2005-2018, results posted three to seven years later, one
    of 30 lead drugs, one target and one disease. Success leans on the target (target 0 is good).
    """
    rng = np.random.default_rng(seed)
    base = synthetic_trials(n, seed=seed, prefix=prefix)
    target = rng.integers(0, N_TARGETS, n)
    disease = rng.integers(0, N_DISEASES, n)
    start = [date(2005, 1, 1) + timedelta(days=int(d)) for d in rng.integers(0, 365 * 13, n)]
    posted = [
        s + timedelta(days=int(d)) for s, d in zip(start, rng.integers(1100, 2600, n), strict=True)
    ]
    success = rng.random(n) < 0.3 + 0.4 * (target == 0)
    frame = base.with_columns(
        pl.Series("any_arm", success),
        pl.Series("lead_drug", [f"D{i % 30}" for i in range(n)]),
        pl.Series(wh.column("start_first"), start),
        pl.Series(wh.column("posted_last"), posted),
        pl.Series(wh.column("precedent"), rng.random(n) < 0.3),
        pl.Series(wh.column("supported_at_start"), rng.random(n) < 0.1),
        pl.Series(wh.column("has_target"), np.ones(n, dtype=bool)),
        pl.Series(wh.column("has_disease"), np.ones(n, dtype=bool)),
        *(pl.Series(wh.column(c), rng.normal(size=n)) for c in wh.SCIENCE_NUMERIC),
        *(
            pl.Series(wh.column(c), rng.choice(["a", "b"], n).tolist())
            for c in wh.SCIENCE_CATEGORICAL
        ),
    )
    ids = frame["nct_id"].to_list()
    keys = pl.DataFrame(
        {
            "nct_id": ids * 3,
            "kind": ["target"] * n + ["disease"] * n + ["pair"] * n,
            "key": [f"g{t}" for t in target] + [f"d{d}" for d in disease]
            + [f"g{t}|d{d}" for t, d in zip(target, disease, strict=True)],
        }
    )  # fmt: skip
    return frame, keys


def pool_of(frame: pl.DataFrame, labels: np.ndarray) -> pl.DataFrame:
    return frame.select("nct_id", wh.column("posted_last"), "lead_drug").with_columns(
        pl.Series("y", labels.astype(float))
    )


def rates(frame, keys, labels, exclude=None) -> pl.DataFrame:
    return wh.group_rates(frame, pool_of(frame, labels), keys, exclude).sort("nct_id")


RATE_COLUMNS = [f"{p}_{k}" for p in ("rate", "n", "logit", "logn") for k in wh.KINDS]


def test_rate_of_a_hand_made_case_is_shrunk_toward_one_half():
    frame = pl.DataFrame(
        {
            "nct_id": ["T", "A", "B", "C"],
            "lead_drug": ["x", "y", "z", "w"],
            wh.column("start_first"): [date(2015, 1, 1), date(2005, 1, 1), date(2005, 1, 1), date(2005, 1, 1)],
            wh.column("posted_last"): [date(2020, 1, 1), date(2010, 1, 1), date(2012, 1, 1), date(2016, 1, 1)],
        }
    )  # fmt: skip
    keys = pl.DataFrame({"nct_id": ["T", "A", "B", "C"], "kind": ["target"] * 4, "key": ["g"] * 4})
    pool = pool_of(frame, np.array([0, 1, 0, 1]))
    got = wh.group_rates(frame.filter(pl.col("nct_id") == "T"), pool, keys)
    # A and B were posted before T started, C was posted a year after: only A (1) and B (0) count
    assert got["n_target"][0] == 2
    assert np.isclose(got["rate_target"][0], (1 + 0.5 * wh.K) / (2 + wh.K))
    assert got["n_disease"][0] == 0 and got["rate_disease"][0] == wh.BASE  # nothing known


def test_flipping_labels_posted_after_a_trial_starts_changes_none_of_its_features():
    frame, keys = what_trials()
    labels = frame["any_arm"].to_numpy()
    cut = date(2012, 1, 1)
    later = (frame[wh.column("posted_last")] >= cut).to_numpy()
    flipped = np.where(later, ~labels, labels)
    before, after = rates(frame, keys, labels), rates(frame, keys, flipped)
    early = (frame.sort("nct_id")[wh.column("start_first")] <= cut).to_numpy()
    assert early.sum() > 50
    for name in RATE_COLUMNS:
        assert np.array_equal(before[name].to_numpy()[early], after[name].to_numpy()[early]), name
    # the check has power: trials that started after the cut do see the flip
    assert not np.allclose(
        before["rate_target"].to_numpy()[~early], after["rate_target"].to_numpy()[~early]
    )


def test_the_same_flip_would_change_features_if_the_posting_rule_were_dropped():
    """Red check: let every pool trial count regardless of when it was posted."""
    frame, keys = what_trials()
    labels = frame["any_arm"].to_numpy()
    cut = date(2012, 1, 1)
    flipped = np.where((frame[wh.column("posted_last")] >= cut).to_numpy(), ~labels, labels)
    loose = frame.with_columns(pl.lit(date(1900, 1, 1)).alias(wh.column("posted_last")))
    before, after = rates(loose, keys, labels), rates(loose, keys, flipped)
    early = (frame.sort("nct_id")[wh.column("start_first")] <= cut).to_numpy()
    assert not np.allclose(
        before["rate_target"].to_numpy()[early], after["rate_target"].to_numpy()[early]
    )


def test_a_trial_never_contributes_to_its_own_rate():
    frame, keys = what_trials(n=120)
    # make every trial eligible for every other, so only "never itself" can keep a label out
    frame = frame.with_columns(pl.lit(date(1900, 1, 1)).alias(wh.column("posted_last")))
    labels = frame["any_arm"].to_numpy()
    for exclude in (None, "lead"):
        before = rates(frame, keys, labels, exclude)
        for i in (0, 17, 60):
            flipped = labels.copy()
            flipped[i] = ~flipped[i]
            after = rates(frame, keys, flipped, exclude)
            for name in RATE_COLUMNS:
                assert before[name][i] == after[name][i], (exclude, name)
            assert not np.allclose(
                before["rate_target"].to_numpy(), after["rate_target"].to_numpy()
            )


def test_excluding_the_lead_drug_drops_its_other_trials_from_every_rate():
    frame, keys = what_trials(n=150)
    frame = frame.with_columns(pl.lit(date(1900, 1, 1)).alias(wh.column("posted_last")))
    labels = frame["any_arm"].to_numpy()
    drug = frame["lead_drug"].to_numpy()
    base = rates(frame, keys, labels, "lead")
    i = 5
    same_drug = np.flatnonzero(drug == drug[i])
    assert len(same_drug) > 2
    flipped = labels.copy()
    flipped[same_drug[same_drug != i]] ^= True  # every other trial of trial i's drug
    after = rates(frame, keys, flipped, "lead")
    for name in RATE_COLUMNS:
        assert base[name][i] == after[name][i]
    plain = rates(frame, keys, labels), rates(frame, keys, flipped)
    assert plain[0]["rate_target"][i] != plain[1]["rate_target"][i]


def test_a_trial_with_no_start_or_no_posting_date_counts_for_and_from_nobody():
    frame, keys = what_trials(n=80)
    frame = frame.with_columns(
        pl.when(pl.int_range(pl.len()) < 10).then(None).otherwise(pl.col(wh.column("start_first")))
        .alias(wh.column("start_first")),
        pl.when(pl.int_range(pl.len()) >= 70).then(None).otherwise(pl.col(wh.column("posted_last")))
        .alias(wh.column("posted_last")),
    )  # fmt: skip
    got = rates(frame, keys, frame["any_arm"].to_numpy())
    assert (got["n_target"].to_numpy()[:10] == 0).all()
    assert (got["rate_target"].to_numpy()[:10] == wh.BASE).all()
    labels = frame["any_arm"].to_numpy()
    changed = labels.copy()
    changed[70:] ^= True  # their results have no posting date: no one may count them
    assert np.array_equal(
        got["n_target"].to_numpy(), rates(frame, keys, changed)["n_target"].to_numpy()
    )


def test_how_has_no_what_column_and_what_has_no_how_column():
    how, what = wh.how_columns(), wh.what_columns()
    assert not how & what
    assert "therapeutic_area" not in how and "therapeutic_area" in what
    assert not any(c.startswith((wh.PREFIX, "rate_", "logit_", "logn_")) for c in how)
    assert "phase" in how and "phase" not in what
    for result_column in ("control_kind", "n_arms", "overall_status", "results_lag_months"):
        assert result_column not in how | what
    text, yes_no, numbers = models.column_sets("WHAT")
    assert not {*text, *yes_no, *numbers} & how
    assert set(models.column_sets("WHAT + phase")[0]) - set(text) == {"phase"}
    both = {c for part in models.column_sets("HOW + WHAT") for c in part}
    assert both == how | what


def test_four_specs_per_version_in_the_registry_shape_with_unique_names():
    from trial_benchmark.models.registry import ModelSpec, require_unique_names
    from trial_benchmark.science import links

    assert len(models.SPECS) == 4 * len(links.VERSIONS)
    assert all(isinstance(s, ModelSpec) for s in models.SPECS)
    require_unique_names(models.SPECS)
    require_unique_names([*models.SPECS, *models.make_specs("today", exclude="lead")])
    assert {s.family for s in models.SPECS} == {"whathow"}
    assert [s.honest for s in models.make_specs("today")] == [True, False, False, False]
    assert all(wh.table_name("2020") in s.tables for s in models.make_specs("2020"))


def test_models_fit_and_predict_without_labels_and_what_adds_to_how():
    train, keys = what_trials(n=700, seed=1)
    test, test_keys = what_trials(n=400, seed=2, prefix="NCT9")
    # start after every training result is posted, so the whole pool counts toward the test rates
    test = test.with_columns(pl.lit(date(2030, 1, 1)).alias(wh.column("start_first")))
    all_keys = pl.concat([keys, test_keys])
    blind = test.drop("any_arm", "all_arms", "highest_dose", "has_dose_ladder")
    outcome = test["any_arm"].to_numpy()
    from trial_benchmark.metrics import auroc

    scores = {}
    for block in models.BLOCKS:
        p = models.fit_what(train, block, "today", keys=all_keys).predict(blind)
        assert p.shape == (test.height,) and np.all(np.isfinite(p))
        scores[block] = auroc(outcome, p)
    assert scores["WHAT"] > 0.55  # the target rate carries the signal built into the data
    assert scores["HOW + WHAT"] > scores["HOW"]


def test_predictions_do_not_depend_on_the_scored_trials_labels():
    train, keys = what_trials(n=500, seed=3)
    test, _ = what_trials(n=500, seed=3)
    model = models.fit_what(train.head(350), "HOW + WHAT", "today", keys=keys)
    scored = test.tail(150)
    other = scored.with_columns(~pl.col("any_arm"))
    assert np.array_equal(model.predict(scored), model.predict(other))


# ---------------------------------------------------------------- the report's arithmetic


def test_thirds_split_evenly_and_in_order():
    values = np.random.default_rng(0).normal(size=300)
    third = wh.thirds(values)
    assert np.bincount(third).tolist() == [100, 100, 100]
    assert (
        values[third == 0].max()
        < values[third == 1].min()
        < values[third == 1].max()
        < values[third == 2].min()
    )


def test_survival_share_is_high_when_what_is_new_and_low_when_what_copies_how():
    rng = np.random.default_rng(1)
    n = 6000
    parts = rng.integers(0, 3, n)
    phase = rng.choice(["phase2", "phase3"], n)
    sponsor = rng.choice(["INDUSTRY", "OTHER"], n)
    how = rng.normal(size=n) + (phase == "phase3")
    new = rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-(0.8 * new + 0.5 * how - 0.3)))
    y_how_only = rng.random(n) < 1 / (1 + np.exp(-(0.8 * how - 0.3)))
    as_probability = lambda x: 1 / (1 + np.exp(-x))
    independent = report.survival(
        y, as_probability(how), as_probability(new), parts, phase, sponsor
    )
    copy = report.survival(
        y_how_only,
        as_probability(how),
        as_probability(how + 0.7 * rng.normal(size=n)),
        parts,
        phase,
        sponsor,
    )
    assert independent.share > 0.8
    assert copy.share < 0.3
    assert independent.top_minus_bottom > 0


def test_correlation_of_a_model_with_itself_is_one_and_with_noise_is_near_zero():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.1, 0.9, 2000)
    parts = rng.integers(0, 3, 2000)
    assert report.correlation(p, p, parts) == (1.0, 1.0) or np.allclose(
        report.correlation(p, p, parts), 1.0
    )
    assert abs(report.correlation(p, rng.uniform(0.1, 0.9, 2000), parts)[0]) < 0.1


def test_success_by_thirds_compares_trials_of_the_same_phase_only():
    rng = np.random.default_rng(3)
    n = 3000
    phase = np.where(rng.random(n) < 0.5, "phase2", "phase3")
    score = rng.normal(size=n) + 3 * (phase == "phase3")  # phase moves the score a lot
    y = rng.random(n) < 0.3 + 0.1 * (phase == "phase3") + 0.2 * (
        score - 3 * (phase == "phase3") > 0
    )
    result = report.success_by_thirds(y, score, np.zeros(n, dtype=int), phase)
    for name in report.PHASES:
        rates, counts = result[name]
        assert abs(counts[0] - counts[2]) <= 1  # thirds are cut inside the phase
        assert rates[2] > rates[0]


def test_clustered_intervals_are_wider_than_the_trial_level_ones_when_a_drug_repeats_its_fate():
    rng = np.random.default_rng(4)
    n_drugs, per = 40, 20
    drug = np.repeat(np.arange(n_drugs), per)
    parts = np.tile(np.arange(2), n_drugs * per // 2)
    drug_effect = np.repeat(rng.normal(size=n_drugs), per)
    y = rng.random(len(drug)) < 1 / (1 + np.exp(-3 * drug_effect))
    p = 1 / (1 + np.exp(-(drug_effect + 0.5 * rng.normal(size=len(drug)))))
    q = rng.random(len(drug))
    predictions = {"a": p, "b": q}
    clustered, differences = report.clustered_scores(y, predictions, parts, drug, [("a", "b")], 300)
    plain, _ = report.clustered_scores(
        y, predictions, parts, np.arange(len(drug)), [("a", "b")], 300
    )
    assert clustered["a"][2] - clustered["a"][1] > plain["a"][2] - plain["a"][1]
    value, low, high = differences[("a", "b")]
    assert low < value < high and value > 0
    same, _ = report.clustered_scores(y, {"a": p, "b": p}, parts, drug, [], 50)
    assert same["a"][0] == same["b"][0]


def test_report_text_runs_on_made_up_arrays():
    rng = np.random.default_rng(5)
    n = 900
    frame_parts = rng.integers(0, 3, n)
    y = rng.random(n) < 0.5
    names = (
        {models.spec_name(b, "today"): rng.uniform(0.2, 0.8, n) for b in models.BLOCKS}
        | {
            models.spec_name(b, "today", "lead"): rng.uniform(0.2, 0.8, n)
            for b in models.BLOCKS[1:]
        }
        | {report.REGISTRATION: rng.uniform(0.2, 0.8, n)}
        | {report.strict_precedent_name("today"): rng.uniform(0.2, 0.8, n)}
    )
    scored = report.Scored(
        y, names, frame_parts, rng.choice(["phase2", "phase3"], n),
        rng.choice(["INDUSTRY", "OTHER"], n), rng.choice(["onc", "neuro", "other"], n),
        rng.integers(0, 200, n).astype(str),
    )  # fmt: skip
    import dataclasses

    folds = dataclasses.replace(scored, unit="fold")
    text = report.report({"today": {"rounds": scored, "folds": folds}}, {"2020": "no table"})
    assert "HOW + WHAT minus HOW" in text and "survives" in text and "Not run: no table" in text
    assert "registration baseline + strict precedent flag alone" in text
    assert "therapeutic area and HOW third" in text and "strata had trials in both tails" in text
    assert "Mean within-round AUROC" in text and "Mean within-fold AUROC" in text


# ---------------------------------------------------------------- round 12 additions


def test_a_trial_posted_on_the_scored_trials_start_day_does_not_count():
    frame = pl.DataFrame(
        {
            "nct_id": ["T", "A", "B"],
            "lead_drug": ["x", "y", "z"],
            wh.column("start_first"): [date(2015, 1, 1), date(2005, 1, 1), date(2005, 1, 1)],
            wh.column("posted_last"): [date(2020, 1, 1), date(2015, 1, 1), date(2014, 12, 31)],
        }
    )
    keys = pl.DataFrame({"nct_id": ["T", "A", "B"], "kind": ["target"] * 3, "key": ["g"] * 3})
    got = wh.group_rates(
        frame.filter(pl.col("nct_id") == "T"), pool_of(frame, np.array([0, 1, 1])), keys
    )
    assert got["n_target"][0] == 1  # B (the day before) counts, A (the same day) does not


def test_a_pool_trial_sharing_two_targets_counts_once_and_pairs_need_both_halves():
    frame = pl.DataFrame(
        {
            "nct_id": ["T", "U1", "U2"],
            "lead_drug": ["x", "y", "z"],
            wh.column("start_first"): [date(2015, 1, 1), date(2005, 1, 1), date(2005, 1, 1)],
            wh.column("posted_last"): [date(2020, 1, 1), date(2010, 1, 1), date(2010, 1, 1)],
        }
    )
    rows = [
        ("T", "target", "g1"), ("T", "target", "g2"),
        ("T", "disease", "d1"), ("T", "disease", "d2"),
        ("T", "pair", "g1|d1"), ("T", "pair", "g1|d2"), ("T", "pair", "g2|d1"), ("T", "pair", "g2|d2"),
        ("U1", "target", "g1"), ("U1", "target", "g2"), ("U1", "disease", "d1"), ("U1", "pair", "g1|d1"),
        ("U2", "target", "g2"), ("U2", "disease", "d9"),
    ]  # fmt: skip
    keys = pl.DataFrame(rows, schema=["nct_id", "kind", "key"], orient="row")
    got = wh.group_rates(
        frame.filter(pl.col("nct_id") == "T"), pool_of(frame, np.array([0, 1, 1])), keys
    )
    assert got["n_target"][0] == 2  # U1 shares two targets and still counts once
    assert got["n_disease"][0] == 1 and got["n_pair"][0] == 1  # only U1 shares d1 and g1|d1
    assert np.isclose(got["rate_pair"][0], (1 + 0.5 * wh.K) / (1 + wh.K))  # U1 succeeded


def test_an_unknown_lead_drug_on_either_side_excludes_nobody_even_when_excluding():
    frame = pl.DataFrame(
        {
            "nct_id": ["T", "A", "B"],
            "lead_drug": [None, "y", None],
            wh.column("start_first"): [date(2015, 1, 1), date(2005, 1, 1), date(2005, 1, 1)],
            wh.column("posted_last"): [date(2020, 1, 1), date(2010, 1, 1), date(2010, 1, 1)],
        }
    )
    keys = pl.DataFrame({"nct_id": ["T", "A", "B"], "kind": ["target"] * 3, "key": ["g"] * 3})
    got = wh.group_rates(
        frame.filter(pl.col("nct_id") == "T"), pool_of(frame, np.array([0, 1, 1])), keys, "lead"
    )
    assert got["n_target"][0] == 2


def test_a_scored_frame_with_no_lead_drug_column_is_refused_not_filled_with_nulls():
    import pytest

    frame, keys = what_trials(n=60)
    pool = pool_of(frame, frame["any_arm"].to_numpy())
    with pytest.raises(ValueError, match="lead_drug"):
        wh.what_frame(frame.drop("lead_drug"), pool, keys)


def test_thirds_break_ties_by_average_rank_so_equal_values_share_a_third():
    assert wh.thirds(np.zeros(9)).tolist() == [1] * 9  # all tied: one middle group, not split
    values = np.array([1.0] * 4 + [2.0] * 4 + [3.0] * 4)
    assert wh.thirds(values).tolist() == [0] * 4 + [1] * 4 + [2] * 4
    shuffled = np.array([3.0, 1.0, 2.0, 1.0, 3.0, 2.0, 3.0, 2.0, 1.0, 1.0, 2.0, 3.0])
    assert wh.thirds(shuffled).tolist() == [2 if v == 3 else 1 if v == 2 else 0 for v in shuffled]


def test_cluster_weights_keep_every_trial_of_a_drug_together_across_rounds():
    drug = np.array(["a", "a", "b", "b", "c", "c", "c"])
    weights = report.cluster_weights(drug, 200, seed=1)
    assert weights.shape == (200, 7)
    for rows in ([0, 1], [2, 3], [4, 5, 6]):
        assert (weights[:, rows] == weights[:, [rows[0]]]).all()
    assert (weights.sum(axis=1) > 0).all() and weights.max() > 1  # drawn with replacement


def test_the_shared_clustered_bootstrap_draws_a_drug_once_for_all_rounds(monkeypatch):
    from trial_benchmark import validate

    n = 12
    drug = np.array(list("aabbccddeeff"))
    parts = np.tile([0, 1], n // 2)  # every drug has one trial in each round
    y = np.arange(n) % 3 == 0
    p = np.arange(n, dtype=float)  # the score names the trial, so a call shows who was drawn
    seen = []

    def spy(outcomes, scores):
        seen.append(np.sort(scores).astype(int))
        return 0.5

    monkeypatch.setattr(validate.metrics, "auroc", spy)
    validate.mean_auroc_bootstrap(y, {"m": p}, parts, [], n_resamples=20, clusters=drug)
    resamples = [(seen[2 * i], seen[2 * i + 1]) for i in range(20)]
    for first, second in resamples:
        drugs_first = sorted(drug[first].tolist())
        drugs_second = sorted(drug[second].tolist())
        assert drugs_first == drugs_second  # the same drugs, as often, in both rounds


def test_survival_uses_thirds_and_strata_within_each_part_and_counts_strata_with_both_tails():
    rng = np.random.default_rng(7)
    n = 4000
    parts = np.repeat([0, 1], n // 2)
    how = rng.normal(size=n) + 5 * parts  # round 1 is calibrated very differently
    what = rng.normal(size=n) + 5 * parts
    y = rng.random(n) < 1 / (1 + np.exp(-(0.8 * (what - 5 * parts))))
    phase = rng.choice(["phase2", "phase3"], n)
    sponsor = rng.choice(["INDUSTRY", "OTHER"], n)
    s = report.survival(y, how, what, parts, phase, sponsor)
    assert s.n_strata <= s.n_strata_total and s.n_strata > 0
    assert 0.5 < s.share < 1.5  # pooled thirds would be all-top in round 1 and break this
    finer = report.survival(y, how, what, parts, phase, sponsor, rng.choice(list("abcdef"), n))
    assert finer.n_strata_total > s.n_strata_total


def test_survival_with_weights_of_one_is_the_plain_answer_and_zero_weight_drops_a_trial():
    rng = np.random.default_rng(8)
    n = 1500
    parts = rng.integers(0, 3, n)
    how, what = rng.normal(size=n), rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-what))
    phase, sponsor = rng.choice(["phase2", "phase3"], n), rng.choice(["I", "O"], n)
    plain = report.survival(y, how, what, parts, phase, sponsor)
    ones = report.survival(y, how, what, parts, phase, sponsor, weights=np.ones(n))
    assert plain == ones


def test_survival_and_correlation_intervals_contain_the_point_and_are_wider_with_clusters():
    rng = np.random.default_rng(9)
    n_drugs, per = 60, 15
    drug = np.repeat(np.arange(n_drugs), per).astype(str)
    n = len(drug)
    parts = rng.integers(0, 3, n)
    effect = np.repeat(rng.normal(size=n_drugs), per)
    how = rng.normal(size=n)
    what = effect + 0.5 * rng.normal(size=n)
    y = rng.random(n) < 1 / (1 + np.exp(-2 * effect))
    phase, sponsor = rng.choice(["phase2", "phase3"], n), rng.choice(["I", "O"], n)
    point = report.survival(y, how, what, parts, phase, sponsor)
    low, high = report.survival_interval(y, how, what, parts, drug, phase, sponsor, n_resamples=200)
    assert low < point.share < high
    plain = report.survival_interval(
        y, how, what, parts, np.arange(n).astype(str), phase, sponsor, n_resamples=200
    )
    assert high - low > plain[1] - plain[0]
    value, c_low, c_high = report.correlation_interval(how, what, parts, drug, n_resamples=200)
    assert c_low < value < c_high
    cells = report.thirds_intervals(y, what, parts, phase, drug, n_resamples=200)
    for name in report.PHASES:
        rates, intervals, gap = cells[name]
        assert all(lo <= r <= hi for r, (lo, hi) in zip(rates, intervals, strict=True))
        assert gap[0] <= rates[2] - rates[0] <= gap[1]


# ---------------------------------------------------------------- the real tables (need the data)


def _needs_data():
    from trial_benchmark.science import links

    return not (wh.INTERIM / "precedent.parquet").exists() or not links.TRIAL_DRUGS.exists()


@pytest.mark.skipif(_needs_data(), reason="needs the built data tables")
@pytest.mark.parametrize("version", ["today", "2020_names"])
def test_build_tables_smoke_uses_the_strict_flag_and_leaves_chemistry_empty_for_names(
    version, tmp_path, monkeypatch
):
    for name in ("dev", "test_features", "precedent", "genetics"):
        (tmp_path / f"{name}.parquet").symlink_to(wh.INTERIM / f"{name}.parquet")
    monkeypatch.setattr(wh, "INTERIM", tmp_path)
    table, keys = wh.build_tables(version)
    assert table.columns == ["nct_id", *(wh.column(c) for c in wh.TABLE_COLUMNS)]
    assert table["nct_id"].is_unique().all() and keys.columns == ["nct_id", "kind", "key"]
    flag = pl.read_parquet(tmp_path / "precedent.parquet")
    strict = flag.sort("nct_id")[f"precedent_exp_{version}"]
    assert table.sort("nct_id")[wh.column("precedent")].to_list() == strict.to_list()
    chemistry = [wh.column(c) for c in wh.SCIENCE_NUMERIC if c.startswith("chem_")]
    chemistry.append(wh.column("chem_action_type"))
    if version == "2020_names":
        assert all(table[c].null_count() == table.height for c in chemistry)
    else:
        assert any(table[c].null_count() < table.height for c in chemistry)


@pytest.mark.skipif(_needs_data(), reason="needs the built data tables")
def test_the_built_tables_on_disk_carry_the_strict_flag_and_the_same_columns():
    for version in ("today", "2020", "2020_names"):
        table = pl.read_parquet(wh.INTERIM / f"{wh.table_name(version)}.parquet").sort("nct_id")
        flag = pl.read_parquet(wh.INTERIM / "precedent.parquet").sort("nct_id")
        assert table[wh.column("precedent")].to_list() == flag[f"precedent_exp_{version}"].to_list()
