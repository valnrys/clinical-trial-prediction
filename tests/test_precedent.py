from datetime import date, timedelta

import polars as pl

from trial_benchmark import precedent as pr

LINKS = pl.DataFrame(
    {
        "drug_id": ["A", "B", "BSALT", "C"],
        "target_id": ["ENSG1", "ENSG1", "ENSG1", "ENSG2"],
        "action_type": [None] * 4,
    }
)


def registry(rows):
    """rows of (nct id, phases, start, last day of start, first submitted[, last day of the
    first-submitted date[, withdrawn]])."""
    out = []
    for n, phases, start, last, submitted, *more in rows:
        out.append(
            {
                "nct_id": n,
                "phases": phases,
                "start_first": start,
                "start_last": last,
                "submit_first": submitted,
                "submit_last": more[0] if more and more[0] else submitted,
                "withdrawn": bool(more[1]) if len(more) > 1 else False,
            }
        )
    return pl.DataFrame(
        out,
        schema={
            "nct_id": pl.String,
            "phases": pl.List(pl.String),
            "start_first": pl.Date,
            "start_last": pl.Date,
            "submit_first": pl.Date,
            "submit_last": pl.Date,
            "withdrawn": pl.Boolean,
        },
    )


def drugs(rows):
    """rows of (nct id, drug id, canonical molecule, matches an experimental arm)."""
    table = pl.DataFrame(
        rows, schema=["nct_id", "drug_id", "canon", "is_exp"], orient="row"
    ).with_columns(pl.col("is_exp").cast(pl.Boolean))
    return table.with_columns(
        (pl.col("is_exp") | ~pl.col("is_exp").any().over("nct_id")).alias("is_main")
    )


def diseases(rows):
    return pl.DataFrame(rows, schema=["nct_id", "disease"], orient="row")


D = date
T_START = D(2020, 6, 1)


def base(
    u_start=date(2018, 1, 1),
    u_submit=date(2017, 12, 1),
    u_phases=("PHASE3",),
    u_drug="B",
    u_canon="B",
    u_disease="MONDO1",
    u_exp=True,
    u_submit_last=None,
    u_withdrawn=False,
):
    """Scored trial T (drug A, target ENSG1, disease MONDO1) and one earlier trial U."""
    return pr.Inputs(
        registry=registry(
            [
                ("T", ["PHASE2"], T_START, D(2020, 6, 30), D(2020, 5, 1)),
                ("U", list(u_phases), u_start, u_start, u_submit, u_submit_last, u_withdrawn),
            ]
        ),
        drugs=drugs([("T", "A", "A", True), ("U", u_drug, u_canon, u_exp)]),
        diseases=diseases([("T", "MONDO1"), ("U", u_disease)]),
        scored=("T",),
    )


def flag(inputs, strict=False, links=LINKS, include_combined=False):
    pairs = pr.precedent_pairs(inputs, links, strict, include_combined)
    return pr.flag_from_pairs(pairs, inputs.scored, "f")["f"].to_list()


def test_an_earlier_phase3_of_another_molecule_on_the_same_target_and_disease_sets_the_flag():
    assert flag(base()) == [True]


def test_leakage_a_phase3_that_starts_after_the_scored_trial_does_not_set_the_flag():
    assert flag(base(u_start=D(2020, 7, 1), u_submit=D(2020, 7, 1))) == [False]
    assert flag(base(u_start=D(2021, 1, 1), u_submit=D(2017, 1, 1))) == [
        False
    ]  # even if registered early


def test_leakage_a_trial_registered_after_the_scored_one_started_does_not_set_the_flag():
    assert flag(base(u_start=D(2018, 1, 1), u_submit=D(2020, 8, 1))) == [False]  # backdated start


def test_a_start_in_the_same_month_is_not_known_to_be_earlier():
    inputs = base(u_start=D(2020, 6, 30), u_submit=D(2019, 1, 1))  # latest day of "2020-06"
    assert flag(inputs) == [False]


def test_the_flag_ignores_the_earlier_trials_outcome_columns():
    inputs = base()
    extra = inputs.registry.with_columns(
        pl.lit(True).alias("any_arm"), pl.lit("COMPLETED").alias("status")
    )
    assert flag(pr.Inputs(extra, inputs.drugs, inputs.diseases, inputs.scored)) == flag(inputs)


def test_other_requirements_each_switch_the_flag_off():
    assert flag(base(u_phases=("PHASE2",))) == [False]  # not Phase 3
    assert flag(base(u_disease="MONDO2")) == [False]  # other disease
    assert flag(base(u_drug="C", u_canon="C")) == [False]  # other target
    assert flag(base(u_drug="A", u_canon="A")) == [False]  # the same drug
    assert flag(base(u_drug="BSALT", u_canon="A")) == [False]  # a salt of the trial's own molecule


def test_a_combined_phase_2_3_trial_is_not_a_phase_3_precedent_unless_asked():
    combined = base(u_phases=("PHASE2", "PHASE3"))
    assert flag(combined) == [False]
    assert flag(combined, include_combined=True) == [True]  # the sensitivity version
    assert flag(base(), include_combined=True) == [True]


def test_a_withdrawn_trial_never_started_so_it_is_not_a_precedent():
    assert flag(base(u_withdrawn=True)) == [False]
    assert flag(base(u_withdrawn=True), strict=True) == [False]


def test_boundary_an_earlier_trial_that_starts_or_registers_on_the_scored_start_day_does_not_count():
    day_before = T_START - timedelta(days=1)
    assert flag(base(u_start=day_before, u_submit=day_before)) == [True]
    assert flag(base(u_start=T_START, u_submit=day_before)) == [False]  # starts the same day
    assert flag(base(u_start=day_before, u_submit=T_START)) == [False]  # posted the same day


def test_a_month_only_submission_date_must_be_before_the_start_on_its_latest_day():
    early_month = {"u_submit": D(2020, 5, 1), "u_submit_last": D(2020, 5, 31)}
    spanning = {
        "u_submit": D(2020, 5, 1),
        "u_submit_last": D(2020, 6, 30),
    }  # "2020-06" is not "May"
    assert flag(base(**early_month)) == [True]
    assert flag(base(**spanning)) == [False]
    # the same rule through a real month-only registry date
    from trial_benchmark.science.links import date_columns

    raw = pl.DataFrame({"nct_id": ["U"], "start": ["2018-01-05"], "first_submit": ["2020-06"]})
    cols = date_columns(raw)
    assert cols["submit_first"][0] == D(2020, 6, 1) and cols["submit_last"][0] == D(2020, 6, 30)


def test_the_scored_trial_never_counts_as_its_own_precedent():
    inputs = pr.Inputs(
        registry([("T", ["PHASE3"], T_START, D(2020, 6, 30), D(2020, 5, 1))]),
        drugs([("T", "A", "A", True), ("T", "B", "B", False)]),
        diseases([("T", "MONDO1")]),
        ("T",),
    )
    assert flag(inputs) == [False]


def test_a_trial_without_a_start_date_gets_no_flag():
    inputs = base()
    registry_ = inputs.registry.with_columns(
        pl.when(pl.col("nct_id") == "T")
        .then(None)
        .otherwise(pl.col("start_first"))
        .alias("start_first")
    )
    assert flag(pr.Inputs(registry_, inputs.drugs, inputs.diseases, inputs.scored)) == [False]


def test_strict_flag_needs_experimental_drugs_on_both_sides_and_has_no_fallback():
    assert flag(base(u_exp=False), strict=True) == [False]  # earlier drug was only a comparator
    assert flag(base(u_exp=False)) == [True]  # the main flag takes any listed drug there
    only_comparator = base()
    drugs_ = only_comparator.drugs.with_columns(
        pl.when(pl.col("nct_id") == "T").then(False).otherwise(pl.col("is_exp")).alias("is_exp")
    )  # T has no experimental match: main flag falls back to its listed drugs, strict does not
    fallback = pr.Inputs(only_comparator.registry, drugs_, only_comparator.diseases, ("T",))
    fallback = pr.Inputs(
        fallback.registry,
        fallback.drugs.with_columns(
            (pl.col("is_exp") | (pl.col("nct_id") == "T")).alias("is_main")
        ),
        fallback.diseases,
        ("T",),
    )
    assert flag(fallback) == [True]
    assert flag(fallback, strict=True) == [False]


def test_flags_for_two_link_versions_differ_when_the_links_differ():
    inputs = base()
    none = LINKS.filter(pl.col("drug_id") != "B")  # in this release B has no known target
    table = pr.precedent_flags(inputs, {"today": LINKS, "2020": none})
    assert table.columns == ["nct_id", *pr.FLAGS, *pr.SENSITIVITY_FLAGS[:2]]
    assert table.row(0)[:5] == ("T", True, False, True, False)
    assert table.row(0)[5:] == (True, False)


def test_known_answer_check_accepts_real_pairs_and_catches_a_wrong_one():
    inputs = base()
    pairs = pr.precedent_pairs(inputs, LINKS)
    assert pr.check_pairs(inputs, LINKS, pairs) == []
    bad = pairs.with_columns(pl.lit("ENSG2").alias("target_id"))
    assert pr.check_pairs(inputs, LINKS, bad)


def test_funnel_counts_never_rise_after_the_drug_steps():
    inputs = base()
    steps = dict(pr.funnel(inputs, LINKS))
    counts = list(steps.values())
    assert counts[0] == 1 and counts[-1] == 1
    assert counts == sorted(counts, reverse=True)
    late = pr.funnel(base(u_start=D(2021, 1, 1), u_submit=D(2021, 1, 1)), LINKS)
    assert late[-2][1] == 0 and late[-3][1] == 1  # shared target and disease, but it started later


def test_first_in_disease_is_false_when_the_molecule_was_tried_earlier_in_the_disease():
    reports = pl.DataFrame(
        {
            "id": ["T", "E"],
            "drugs": [[{"drugId": "A"}], [{"drugId": "A"}]],
            "trialStartDate": [D(2020, 6, 1), D(2015, 1, 1)],
        },
        schema_overrides={"trialStartDate": pl.Date},
    )
    inputs = pr.Inputs(
        registry([("T", ["PHASE2"], T_START, D(2020, 6, 30), D(2020, 5, 1))]),
        drugs([("T", "A", "A", True), ("E", "A", "A", True)]),
        diseases([("T", "MONDO1"), ("E", "MONDO1")]),
        ("T",),
    )
    assert pr.first_in_disease(inputs, reports)["first_in_disease"].to_list() == [False]
    other_disease = pr.Inputs(
        inputs.registry, inputs.drugs, diseases([("T", "MONDO1"), ("E", "MONDO2")]), ("T",)
    )
    assert pr.first_in_disease(other_disease, reports)["first_in_disease"].to_list() == [True]
    nothing = pr.Inputs(
        inputs.registry, inputs.drugs.filter(pl.col("nct_id") == "E"), inputs.diseases, ("T",)
    )
    assert pr.first_in_disease(nothing, reports)["first_in_disease"].to_list() == [None]


def test_the_dated_name_version_uses_its_own_drug_table_and_its_own_leakage_rules():
    inputs = base()
    dated = precedent_names(inputs, u_drug="B")
    links_2006 = LINKS.filter(pl.col("drug_id") != "BSALT")
    table = pr.precedent_flags(dated, {"2020_names": links_2006})
    assert table.columns == [
        "nct_id",
        "precedent_2020_names",
        "precedent_exp_2020_names",
        "precedent_exp_2020_names_p23",
    ]
    assert table.row(0) == ("T", True, True, True)
    # leakage: the same pair with the earlier Phase 3 starting after the scored trial
    late = base(u_start=D(2020, 7, 1), u_submit=D(2020, 7, 1))
    late_dated = precedent_names(late, u_drug="B")
    assert pr.precedent_flags(late_dated, {"2020_names": links_2006}).row(0) == (
        "T",
        False,
        False,
        False,
    )
    # a drug the 20.06 names do not know is simply absent from that table: no flag
    unknown = precedent_names(inputs, u_drug=None)
    assert pr.precedent_flags(unknown, {"2020_names": links_2006}).row(0) == (
        "T",
        False,
        False,
        False,
    )


def test_version_without_a_name_table_is_refused_not_silently_run_on_todays_drugs():
    try:
        pr.inputs_for_version(base(), "2020_names")
    except ValueError:
        return
    raise AssertionError("expected a ValueError")


def precedent_names(inputs, u_drug):
    matched = [("T", "A", True)] + ([("U", u_drug, True)] if u_drug else [])
    names = pr.names_listed_drugs(
        pl.DataFrame(matched, schema=["nct_id", "drug_id", "is_exp"], orient="row")
    )
    return pr.Inputs(inputs.registry, inputs.drugs, inputs.diseases, inputs.scored, names)


def test_the_sensitivity_flag_adds_combined_phase_2_3_trials_and_nothing_else():
    inputs = base(u_phases=("PHASE2", "PHASE3"))
    table = pr.precedent_flags(inputs, {"today": LINKS})
    assert table.columns == [
        "nct_id",
        "precedent_today",
        "precedent_exp_today",
        "precedent_exp_today_p23",
    ]
    assert table.row(0) == ("T", False, False, True)


def test_strict_flag_through_the_real_listing_ignores_a_comparator_drug():
    """Build the drug table from Open Targets-shaped reports, not by hand, then flag it."""
    drug = pl.Struct({"drugFromSource": pl.String, "drugId": pl.String})
    reports = pl.DataFrame(
        {
            "id": ["T", "U"],
            "drugs": [
                [
                    {"drugFromSource": "alphamab", "drugId": "A"},
                    {"drugFromSource": "standard pill", "drugId": "C"},
                ],
                [{"drugFromSource": "standard pill", "drugId": "B"}],  # a comparator only
            ],
        },
        schema_overrides={"drugs": pl.List(drug)},
    )
    molecules = pl.DataFrame(
        {
            "id": ["A", "B", "C"],
            "name": ["alphamab", "standard pill", "standard pill"],
            "drugType": ["x"] * 3,
        }
    )
    reg = registry(
        [
            ("T", ["PHASE2"], T_START, D(2020, 6, 30), D(2020, 5, 1)),
            ("U", ["PHASE3"], D(2018, 1, 1), D(2018, 1, 1), D(2017, 12, 1)),
        ]
    ).with_columns(
        pl.Series("exp_names", [["alphamab"], ["other thing"]], dtype=pl.List(pl.String))
    )
    listed = pr.listed_drugs(reports, molecules, reg, {})
    row = {
        (n, d): (e, m)
        for n, d, e, m in listed.select("nct_id", "drug_id", "is_exp", "is_main").rows()
    }
    assert row[("T", "A")] == (True, True) and row[("T", "C")] == (False, False)
    assert row[("U", "B")] == (False, True)  # no experimental match: the fallback takes it
    links = pl.DataFrame(
        {"drug_id": ["A", "B", "C"], "target_id": ["G"] * 3, "action_type": [None] * 3}
    )
    inputs = pr.Inputs(reg, listed, diseases([("T", "M"), ("U", "M")]), ("T",))
    assert flag(inputs, links=links) == [True]  # through the fallback
    assert flag(inputs, strict=True, links=links) == [False]  # a comparator never supplies it
    names = pr.names_listed_drugs(
        pl.DataFrame(
            [("T", "A", True), ("U", "B", False)],
            schema=["nct_id", "drug_id", "is_exp"],
            orient="row",
        )
    )
    assert names.filter(pl.col("nct_id") == "U")["is_main"].to_list() == [True]
    assert names.filter(pl.col("nct_id") == "U")["is_exp"].to_list() == [False]


def test_first_in_disease_ignores_an_earlier_trial_registered_after_the_scored_one_started():
    reports = pl.DataFrame(
        {
            "id": ["T", "E"],
            "drugs": [[{"drugId": "A"}], [{"drugId": "A"}]],
            "trialStartDate": [D(2020, 6, 1), D(2015, 1, 1)],
        },
        schema_overrides={"trialStartDate": pl.Date},
    )
    reg = registry(
        [
            ("T", ["PHASE2"], T_START, D(2020, 6, 30), D(2020, 5, 1)),
            ("E", ["PHASE2"], D(2015, 1, 1), D(2015, 1, 1), D(2020, 8, 1)),  # backdated start
        ]
    )
    inputs = pr.Inputs(
        reg,
        drugs([("T", "A", "A", True), ("E", "A", "A", True)]),
        diseases([("T", "M"), ("E", "M")]),
        ("T",),
    )
    assert pr.first_in_disease(inputs, reports)["first_in_disease"].to_list() == [True]
    submitted = reg.with_columns(
        pl.when(pl.col("nct_id") == "E")
        .then(D(2014, 12, 1))
        .otherwise(pl.col("submit_last"))
        .alias("submit_last")
    )
    inputs = pr.Inputs(submitted, inputs.drugs, inputs.diseases, ("T",))
    assert pr.first_in_disease(inputs, reports)["first_in_disease"].to_list() == [False]
    withdrawn = submitted.with_columns((pl.col("nct_id") == "E").alias("withdrawn"))
    inputs = pr.Inputs(withdrawn, inputs.drugs, inputs.diseases, ("T",))
    assert pr.first_in_disease(inputs, reports)["first_in_disease"].to_list() == [True]


def test_first_in_disease_reads_a_month_level_open_targets_date_as_its_last_day():
    """Open Targets gives a month-only start as the month's last day; so a trial only it knows,
    dated 2020-06-30, is not earlier than a trial that started on 2020-06-15."""
    reports = pl.DataFrame(
        {
            "id": ["T", "E"],
            "drugs": [[{"drugId": "A"}], [{"drugId": "A"}]],
            "trialStartDate": [D(2020, 6, 15), D(2020, 6, 30)],
        },
        schema_overrides={"trialStartDate": pl.Date},
    )
    reg = registry([("T", ["PHASE2"], D(2020, 6, 15), D(2020, 6, 15), D(2020, 5, 1))])
    inputs = pr.Inputs(
        reg,
        drugs([("T", "A", "A", True), ("E", "A", "A", True)]),
        diseases([("T", "M"), ("E", "M")]),
        ("T",),
    )
    assert pr.first_in_disease(inputs, reports)["first_in_disease"].to_list() == [True]
    older = reports.with_columns(pl.Series("trialStartDate", [D(2020, 6, 15), D(2020, 5, 31)]))
    assert pr.first_in_disease(inputs, older)["first_in_disease"].to_list() == [False]


def test_first_in_disease_by_20_06_names_uses_only_the_name_matched_registry_trials():
    reg = registry(
        [
            ("T", ["PHASE2"], T_START, D(2020, 6, 30), D(2020, 5, 1)),
            ("E", ["PHASE3"], D(2015, 1, 1), D(2015, 1, 1), D(2014, 12, 1)),
        ]
    )
    names = pr.names_listed_drugs(
        pl.DataFrame(
            [("T", "N1", True), ("E", "N1", True)],
            schema=["nct_id", "drug_id", "is_exp"],
            orient="row",
        )
    )
    inputs = pr.Inputs(
        reg, drugs([("T", "A", "A", True)]), diseases([("T", "M"), ("E", "M")]), ("T",), names
    )
    got = pr.first_in_disease_names(inputs)
    assert got.columns == ["nct_id", "first_in_disease_2020_names"]
    assert got["first_in_disease_2020_names"].to_list() == [False]


def test_replace_dates_keeps_the_name_table():
    inputs = pr.Inputs(base().registry, base().drugs, base().diseases, ("T",), base().drugs)
    assert pr.replace_dates(inputs, far_past=True).names_drugs is not None
    assert pr.replace_dates(inputs, submit_free=True).names_drugs is not None


def test_check_pairs_recomputes_dates_from_the_raw_strings_when_the_registry_has_them():
    inputs = base()
    pairs = pr.precedent_pairs(inputs, LINKS)
    raw = inputs.registry.with_columns(
        pl.when(pl.col("nct_id") == "U")
        .then(pl.lit("2018-01-01"))
        .otherwise(pl.lit("2020-06-01"))
        .alias("start"),
        pl.when(pl.col("nct_id") == "U")
        .then(pl.lit("2017-12-01"))
        .otherwise(pl.lit("2020-05-01"))
        .alias("first_submit"),
    )
    good = pr.Inputs(raw, inputs.drugs, inputs.diseases, inputs.scored)
    assert pr.check_pairs(good, LINKS, pairs) == []
    # the derived columns agree with the pair but the raw start string says "later": caught
    wrong = raw.with_columns(
        pl.when(pl.col("nct_id") == "U")
        .then(pl.lit("2020-06"))
        .otherwise(pl.col("start"))
        .alias("start")
    )
    bad = pr.Inputs(wrong, inputs.drugs, inputs.diseases, inputs.scored)
    assert pr.check_pairs(bad, LINKS, pairs)
