import numpy as np
import polars as pl
import pytest

from elman_exercise import biology_score as bs
from elman_exercise.opentargets import INTERIM

DECLARED_INPUTS = [
    "depmap_mean_effect",
    "depmap_fraction_dependent",
    "depmap_lineage_effect",
    "depmap_lineage_minus_other",
    "expr_tau",
    "expr_n_tissues",
    "expr_vital_max",
    "expr_disease_tissue",
    "net_viability_hit_fraction",
    "net_other_hit_fraction",
    "net_partners_per_paper",
    "constraint",
    "mouse_knockout",
    "chem_mol_weight",
    "chem_logp",
    "chem_polar_surface_area",
    "chem_qed",
    "chem_rule_of_five_violations",
    "depmap_common_essential",
    "expr_disease_tissue_top3",
    "target_class",
    "target_location",
    "chem_action_type",
]


def features(n: int, seed: int = 0) -> pl.DataFrame:
    """Made-up biology columns with the real names; no outcome."""
    rng = np.random.default_rng(seed)
    data = {"nct_id": [f"NCT{i:07d}" for i in range(n)]}
    for c in bs.NUMERIC:
        data[c] = rng.normal(size=n).tolist()
    data["expr_n_tissues"] = rng.integers(1, 52, n).astype(float).tolist()
    data["chem_qed"] = rng.uniform(0.1, 0.95, n).tolist()
    data["expr_vital_max"] = rng.uniform(0, 10, n).tolist()
    for c in bs.BOOLEAN:
        data[c] = (rng.random(n) < 0.2).tolist()
    data["target_class"] = rng.choice(["Enzyme", "Ion channel", "Other"], n).tolist()
    data["target_location"] = rng.choice(["membrane", "intracellular", "secreted"], n).tolist()
    data["chem_action_type"] = rng.choice(["inhibitor", "activator", "agonist"], n).tolist()
    data["lead_drug"] = [f"D{i % 400}" for i in range(n)]
    return pl.DataFrame(data)


def row(**values) -> pl.DataFrame:
    """One trial with every flag input missing except the ones named."""
    base = {c: None for c in bs.INPUT_COLUMNS}
    base["nct_id"] = "NCT0000001"
    base.update(values)
    schema = {c: pl.Float64 for c in bs.NUMERIC} | {
        **{c: pl.Boolean for c in bs.BOOLEAN},
        **{c: pl.String for c in bs.CATEGORICAL},
    }
    return pl.DataFrame([base], schema={"nct_id": pl.String, **schema})


def flags_of(**values) -> dict:
    return bs.flag_frame(row(**values)).row(0, named=True)


# ---------------------------------------------------------------- the flags


def test_the_flags_are_these_six_in_this_order():
    assert bs.FLAGS == (
        "f_broad",
        "f_vital",
        "f_essential",
        "f_intracellular",
        "f_activator",
        "f_lowqed",
    )
    out = bs.flag_frame(features(5))
    assert out.columns == ["nct_id", *bs.FLAGS, "n_flags"]


@pytest.mark.parametrize(
    "column, value, flag, expected",
    [
        ("expr_n_tissues", 49.0, "f_broad", False),  # more than 49, not 49 or more
        ("expr_n_tissues", 50.0, "f_broad", True),
        ("expr_vital_max", bs.VITAL_CUT, "f_vital", False),  # strictly above the constant
        ("expr_vital_max", float(np.nextafter(bs.VITAL_CUT, 10)), "f_vital", True),
        ("expr_vital_max", 5.79, "f_vital", True),  # the rounded figure in the plan is above it
        ("chem_qed", bs.QED_CUT, "f_lowqed", True),  # at or below
        ("chem_qed", float(np.nextafter(bs.QED_CUT, 1)), "f_lowqed", False),
        ("depmap_common_essential", True, "f_essential", True),
        ("depmap_common_essential", False, "f_essential", False),
        ("target_location", "intracellular", "f_intracellular", True),
        ("target_location", "membrane", "f_intracellular", False),
        ("chem_action_type", "activator", "f_activator", True),
        ("chem_action_type", "inhibitor", "f_activator", False),
    ],
)
def test_each_flag_turns_on_at_its_declared_threshold(column, value, flag, expected):
    out = flags_of(**{column: value})
    assert out[flag] is expected
    assert out["n_flags"] == int(expected)


def test_a_missing_value_leaves_its_flag_off():
    out = flags_of()  # every input missing
    assert out["n_flags"] == 0
    assert not any(out[f] for f in bs.FLAGS)


def test_n_flags_counts_flags_without_weights():
    out = flags_of(
        expr_n_tissues=51.0,
        expr_vital_max=9.0,
        depmap_common_essential=True,
        target_location="intracellular",
        chem_action_type="activator",
        chem_qed=0.2,
    )
    assert out["n_flags"] == 6


def test_the_groups_are_zero_one_and_two_or_more():
    assert bs.flag_groups(np.array([0, 1, 2, 3, 6])).tolist() == ["0", "1", "2+", "2+", "2+"]


def test_recomputed_thresholds_are_the_declared_quantiles_of_the_early_trials_only():
    """Early values 0..299 (301 would shift the rank): 2/3 and 1/3 nearest-rank quantiles are exact."""
    n = 300
    values = [float(i) for i in range(n)]
    f = pl.DataFrame(
        {
            "nct_id": [f"E{i}" for i in range(n)] + ["LATE1", "LATE2"],
            "expr_n_tissues": [*values, 51.0, 51.0],
            "expr_vital_max": [*values, 1000.0, 1000.0],
            "chem_qed": [*values, -5.0, -5.0],
        }
    )
    got = bs.recompute_thresholds(f, [f"E{i}" for i in range(n)])  # the late trials are ignored
    expected_top = float(np.sort(values)[round((n - 1) * 2 / 3)])  # nearest rank
    expected_bottom = float(np.sort(values)[round((n - 1) / 3)])
    assert got == {
        "N_TISSUES_CUT": expected_top,
        "VITAL_CUT": expected_top,
        "QED_CUT": expected_bottom,
    }


@pytest.mark.skipif(
    not (INTERIM / "whathow_today.parquet").exists(), reason="interim science tables not built"
)
def test_the_declared_constants_match_the_real_pre_2015_distribution():
    ff = bs.feature_frame("today")
    got = bs.recompute_thresholds(ff, bs.pre_2015_ids())
    assert got["N_TISSUES_CUT"] == bs.N_TISSUES_CUT
    assert got["VITAL_CUT"] == bs.VITAL_CUT  # exact, not approximate
    assert got["QED_CUT"] == bs.QED_CUT


@pytest.mark.skipif(
    not (INTERIM / "whathow_today.parquet").exists(), reason="interim science tables not built"
)
def test_the_feature_frame_has_no_outcome_columns_and_only_mapped_targets():
    ff = bs.feature_frame("today")
    assert set(ff.columns) == {"nct_id", *DECLARED_INPUTS, *bs.EXTRA_COLUMNS}
    assert ff["nct_id"].n_unique() == ff.height
    assert not {"any_arm", "all_arms", "y"} & set(ff.columns)


# ---------------------------------------------------------------- the logistic model


def planted(n: int, seed: int, effect: float = 1.5):
    """Features plus outcomes where higher expr_tau means more success (a planted effect)."""
    f = features(n, seed)
    rng = np.random.default_rng(seed + 1)
    p = 1 / (1 + np.exp(-(0.2 + effect * f["expr_tau"].to_numpy())))
    return f, rng.random(n) < p


def test_the_logistic_reads_only_the_declared_columns():
    assert list(bs.INPUT_COLUMNS) == DECLARED_INPUTS
    f, y = planted(600, 1)
    fitted = bs.fit_logistic(f, y)
    assert list(fitted.input_columns) == DECLARED_INPUTS
    # A column outside the list changes nothing, whether or not it is present.
    extra = f.with_columns(
        phase=pl.Series(np.random.default_rng(5).choice(["phase2", "phase3"], f.height)),
        any_arm=pl.Series(y),
        sponsor_class=pl.lit("INDUSTRY"),
    )
    refit = bs.fit_logistic(extra, y)
    assert np.allclose(refit.score(extra), fitted.score(f))
    assert np.allclose(fitted.score(extra), fitted.score(f))


def test_the_model_settings_are_the_declared_ones():
    f, y = planted(300, 2)
    model = bs.fit_logistic(f, y).model
    assert model.C == 1.0
    assert model.l1_ratio in (None, 0, 0.0)  # pure L2 penalty
    assert model.fit_intercept


def with_holes(f: pl.DataFrame, seed: int = 9) -> pl.DataFrame:
    """Blank about a fifth of every numeric column (including the flag inputs)."""
    rng = np.random.default_rng(seed)
    return f.with_columns(
        pl.when(pl.Series(rng.random(f.height) < 0.2)).then(None).otherwise(pl.col(c)).alias(c)
        for c in bs.NUMERIC
    )


def test_missing_values_take_the_training_mean_with_no_indicator():
    f, y = planted(800, 3)
    holes = with_holes(f)
    assert holes["expr_n_tissues"].null_count() > 100  # missing values are really in training
    fitted = bs.fit_logistic(holes, y)
    levels = sum(len(v) for v in fitted.levels.values())
    width = len(bs.FLAGS) + len(bs.NUMERIC) + len(bs.BOOLEAN) + levels
    assert fitted.model.coef_.shape[1] == width  # no extra missing-indicator columns
    probe = f.head(40)
    for column in bs.NUMERIC:  # the flag inputs included
        blank = probe.with_columns(pl.lit(None, pl.Float64).alias(column))
        filled = probe.with_columns(pl.lit(fitted.stats[column][0]).alias(column))
        if column in ("expr_n_tissues", "expr_vital_max", "chem_qed"):
            continue  # these also feed a flag, which a filled-in mean can switch on
        assert np.allclose(fitted.score(blank), fitted.score(filled)), column
    # the training mean is the mean of the NON-missing training values
    assert fitted.stats["expr_tau"][0] == pytest.approx(holes["expr_tau"].drop_nulls().mean())


def test_a_column_empty_in_training_contributes_exactly_nothing():
    f, y = planted(400, 4)
    empty = f.with_columns(pl.lit(None, pl.Float64).alias("chem_logp"))
    fitted = bs.fit_logistic(empty, y)  # the "2020 names" case: no chemistry
    rng = np.random.default_rng(2)
    scored = f.with_columns(pl.Series("chem_logp", rng.normal(5, 3, f.height)))
    blank = f.with_columns(pl.lit(None, pl.Float64).alias("chem_logp"))
    assert np.array_equal(fitted.score(scored), fitted.score(blank))
    assert np.isfinite(fitted.score(scored)).all()


def test_scores_do_not_change_when_the_frames_outcomes_are_flipped():
    f, y = planted(800, 6)
    train, other = f.head(500), f.tail(300)
    fitted = bs.fit_logistic(train, y[:500])
    other_y = y[500:]
    labelled = other.with_columns(y=pl.Series(other_y))
    flipped = other.with_columns(y=pl.Series(~other_y))
    assert np.array_equal(fitted.score(labelled), fitted.score(flipped))
    assert np.array_equal(fitted.score(labelled), fitted.score(other))
    # and the training outcomes DO matter for the fit (the guard is not vacuous)
    assert not np.allclose(bs.fit_logistic(train, ~y[:500]).score(other), fitted.score(other))


def test_flags_ignore_any_outcome_column_that_is_present():
    f = features(60)
    noisy = f.with_columns(
        any_arm=pl.Series(np.random.default_rng(1).random(60) < 0.5),
        y=pl.Series(np.arange(60)),
    )
    assert bs.flag_frame(noisy).equals(bs.flag_frame(f))
    assert set(bs.flag_frame(noisy).columns) == {"nct_id", *bs.FLAGS, "n_flags"}


def test_tertile_cuts_come_from_the_training_scores_and_split_them_in_thirds():
    f, y = planted(900, 7)
    fitted = bs.fit_logistic(f, y)
    cuts = fitted.tertile_cuts
    assert cuts[0] < cuts[1]
    groups = bs.thirds(fitted.score(f), cuts)
    for g in bs.THIRDS:
        assert abs((groups == g).sum() - 300) <= 3
    assert np.allclose(cuts, np.quantile(fitted.score(f), [1 / 3, 2 / 3]))


def test_outcomes_must_line_up_with_the_training_rows():
    with pytest.raises(ValueError):
        bs.fit_logistic(features(10), np.ones(9))


# ---------------------------------------------------------------- the statistics


def test_a_planted_flag_effect_is_recovered_with_the_right_sign():
    rng = np.random.default_rng(11)
    n = 4000
    n_flags = rng.choice([0, 1, 2, 3], n, p=[0.4, 0.2, 0.25, 0.15])
    phase = rng.choice(["phase2", "phase3"], n)
    base = np.where(phase == "phase3", 0.7, 0.45)
    y = rng.random(n) < base - 0.12 * (n_flags >= 2)
    lead = [f"D{i % 1500}" for i in range(n)]
    c = bs.flag_contrast(y, n_flags, phase, lead)
    assert c.estimate < 0
    assert c.estimate == pytest.approx(-12, abs=4)
    assert c.high < 0  # the interval excludes zero
    assert c.n_exposed == int((n_flags >= 2).sum())
    assert c.n_reference == int((n_flags == 0).sum())


def test_phase_is_pooled_not_confused():
    """Flags are commoner in phase 2, which succeeds less: pooling over phase removes that."""
    rng = np.random.default_rng(12)
    n = 4000
    phase = rng.choice(["phase2", "phase3"], n)
    flagged = rng.random(n) < np.where(phase == "phase2", 0.7, 0.2)
    y = rng.random(n) < np.where(phase == "phase3", 0.8, 0.3)  # no flag effect at all
    n_flags = np.where(flagged, 2, 0)
    c = bs.flag_contrast(y, n_flags, phase, [f"D{i}" for i in range(n)])
    assert abs(c.estimate) < 4
    crude = 100 * (y[n_flags == 2].mean() - y[n_flags == 0].mean())
    assert crude < -10


def test_a_planted_logistic_effect_is_recovered_out_of_sample():
    f, y = planted(3000, 20)
    fitted = bs.fit_logistic(f.head(1500), y[:1500])
    test, test_y = f.tail(1500), y[1500:]
    rng = np.random.default_rng(1)
    phase = rng.choice(["phase2", "phase3"], 1500)
    c = bs.thirds_contrast(
        test_y, fitted.score(test), fitted.tertile_cuts, phase, test["lead_drug"].to_list()
    )
    assert c.estimate > 20
    assert c.low > 0


def clustered(n_per: int, n_drugs: int, seed: int):
    """Trials grouped by drug: every drug has its own success chance, so outcomes cluster."""
    rng = np.random.default_rng(seed)
    drug = np.repeat(np.arange(n_drugs), n_per)
    chance = rng.beta(0.6, 0.6, n_drugs)[drug]
    y = rng.random(len(drug)) < chance
    n_flags = rng.choice([0, 2], n_drugs)[drug]  # the flag belongs to the drug
    phase = rng.choice(["phase2", "phase3"], len(drug))
    return y, n_flags, phase, drug


def test_trials_of_one_drug_share_a_cluster_so_intervals_are_wider():
    y, n_flags, phase, drug = clustered(10, 60, 5)
    shared = bs.flag_contrast(y, n_flags, phase, [f"D{d}" for d in drug])
    unique = bs.flag_contrast(y, n_flags, phase, [f"U{i}" for i in range(len(drug))])
    assert shared.estimate == pytest.approx(unique.estimate)
    assert (shared.high - shared.low) > 1.5 * (unique.high - unique.low)
    r_shared = bs.rate(y, [f"D{d}" for d in drug])
    r_unique = bs.rate(y, [f"U{i}" for i in range(len(drug))])
    assert (r_shared[2] - r_shared[1]) > 1.5 * (r_unique[2] - r_unique[1])


@pytest.mark.parametrize("missing", [None, float("nan"), np.nan])
def test_a_missing_lead_drug_is_its_own_cluster_whatever_its_spelling(missing):
    y, n_flags, phase, drug = clustered(8, 40, 6)
    unique = [f"U{i}" for i in range(len(drug))]
    holes = [missing] * len(drug)
    assert bs.rate(y, holes) == bs.rate(y, unique)
    assert bs.flag_contrast(y, n_flags, phase, holes) == bs.flag_contrast(y, n_flags, phase, unique)
    table = bs.by_group(y, n_flags.astype(str), phase, holes, n_resamples=100)
    ref = bs.by_group(y, n_flags.astype(str), phase, unique, n_resamples=100)
    assert table.select("phase", "group", "n", "per_100").equals(
        ref.select("phase", "group", "n", "per_100")
    )


def test_nan_and_none_mixed_do_not_merge_into_one_cluster():
    y, _, _, drug = clustered(8, 40, 7)
    mixed = [None if i % 2 else float("nan") for i in range(len(drug))]
    unique = [f"U{i}" for i in range(len(drug))]
    assert bs.rate(y, mixed) == bs.rate(y, unique)


def test_a_contrast_with_too_few_trials_is_not_a_number():
    c = bs.flag_contrast([True] * 10, [0] * 5 + [2] * 5, ["phase2"] * 10, [None] * 10)
    assert np.isnan(c.estimate)
    assert c.text() == "n/a"


def test_the_statistics_take_whatever_set_the_caller_supplies():
    """Any set of labels can be scored; the same arrays give the same answer twice."""
    rng = np.random.default_rng(13)
    n = 600
    args = (
        rng.random(n) < 0.5,
        rng.integers(0, 4, n),
        rng.choice(["phase2", "phase3"], n),
        [f"D{i % 200}" for i in range(n)],
    )
    a, b = bs.flag_contrast(*args), bs.flag_contrast(*args)
    assert a == b


def test_group_rates_by_phase():
    y = [True] * 3 + [False] * 1 + [True] * 1 + [False] * 3 + [True] * 2
    group = ["0"] * 4 + ["2+"] * 4 + ["0"] * 2
    phase = ["phase2"] * 8 + ["phase3"] * 2
    table = bs.by_group(y, group, phase, [f"D{i}" for i in range(10)], n_resamples=200)
    assert table.columns == ["phase", "group", "n", "per_100", "low", "high"]
    p2_zero = table.filter((pl.col("phase") == "phase2") & (pl.col("group") == "0")).row(
        0, named=True
    )
    assert p2_zero["n"] == 4
    assert p2_zero["per_100"] == pytest.approx(75.0)
    assert p2_zero["low"] <= 75.0 <= p2_zero["high"]
    assert table.filter(pl.col("phase") == "phase3")["per_100"].to_list() == [100.0]


# ---------------------------------------------------------------- unseen targets


def test_unseen_targets_are_scored_trials_with_no_target_in_any_training_trial(monkeypatch):
    links = pl.DataFrame(
        {
            "nct_id": ["T1", "T2", "S1", "S2", "S3", "S4"],
            "drug_id": ["d"] * 6,
            "target_id": ["G1", "G2", "G1", "G3", "G2", "G9"],
        }
    )
    # S2 has G3 (unseen) AND G2 (seen) -> seen; S4 only G9 -> unseen; S5 has no target at all.
    links = pl.concat(
        [links, pl.DataFrame({"nct_id": ["S2"], "drug_id": ["d"], "target_id": ["G2"]})]
    )
    monkeypatch.setattr(bs, "target_table", lambda version: links)
    got = bs.unseen_targets(["S1", "S2", "S3", "S4", "S5"], ["T1", "T2"], "today")
    assert got == {"S4"}


def test_a_scored_trial_with_no_target_is_neither_unseen_nor_counted_as_seen(monkeypatch):
    links = pl.DataFrame({"nct_id": ["T1", "S1"], "drug_id": ["d", "d"], "target_id": ["G1", "G1"]})
    monkeypatch.setattr(bs, "target_table", lambda version: links)
    assert bs.unseen_targets(["S1", "NOTARGET"], ["T1"], "today") == set()


def test_a_target_seen_only_in_a_scored_trial_is_still_unseen(monkeypatch):
    links = pl.DataFrame({"nct_id": ["T1", "S1"], "drug_id": ["d", "d"], "target_id": ["G1", "G7"]})
    monkeypatch.setattr(bs, "target_table", lambda version: links)
    assert bs.unseen_targets(["S1"], ["T1"], "today") == {"S1"}
