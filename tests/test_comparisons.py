import pytest

from elman_exercise.comparisons import (
    MANY_GROUPS,
    NO_CONTROL,
    NON_INFERIORITY,
    REFERENCE_ARM,
    SINGLE_GROUP,
    classify_analyses,
    dose_of,
    find_control,
    first_eligible_outcome,
    highest_dose_arm,
)


def arm(label, kind):
    return {"label": label, "type": kind}


# --- the control by a placebo title ----------------------------------------------


@pytest.mark.parametrize(
    ("titles", "control"),
    [
        (("Drug X 10 mg", "Placebo"), 1),
        (("Vehicle Cream", "Drug X Cream"), 0),
        (("Sham Stimulation", "Active Stimulation"), 0),
        (("Drug X", "Placebo (0 mg)"), 1),
        (("Drug X", "Placebos"), 1),
    ],
)
def test_a_placebo_title_marks_the_control(titles, control):
    assert find_control(titles, []) == (control, "placebo", "placebo title")


def test_a_placebo_title_beats_a_mistyped_arm():
    # The drug arm is registered as an active comparator, a common registry error.
    arms = [arm("Drug X", "ACTIVE_COMPARATOR"), arm("Placebo", "PLACEBO_COMPARATOR")]
    assert find_control(("Drug X", "Placebo"), arms) == (1, "placebo", "placebo title")


def test_placebo_with_shared_background_is_the_control():
    titles = ("Drug X + Methotrexate", "Placebo + Methotrexate")
    assert find_control(titles, []) == (1, "placebo", "placebo title")
    titles = ("Drug X and Standard Care", "Placebo and Standard Care")
    assert find_control(titles, []) == (1, "placebo", "placebo title")


def test_a_drug_given_with_a_dummy_placebo_is_not_the_control():
    # Double-dummy head-to-head where only one title names the dummy.
    assert find_control(("Drug Y + Placebo for X", "Drug X"), []) is None


def test_when_both_titles_name_a_placebo_only_a_pure_placebo_arm_is_the_control():
    assert find_control(("Drug X + Placebo", "Placebo"), []) == (1, "placebo", "placebo title")
    titles = ("Amitriptyline 4% Cream, Placebo Capsules", "Placebo Cream and Capsules")
    assert find_control(titles, []) == (1, "placebo", "placebo title")


def test_double_dummy_head_to_head_has_no_control():
    titles = ("Drug X + Placebo to Drug Y", "Drug Y + Placebo to Drug X")
    assert find_control(titles, []) is None


def test_an_unsettled_placebo_title_can_still_be_settled_by_arm_types():
    # Double-dummy trial against an active comparator the registry marks.
    arms = [arm("Drug Y + Placebo for X", "EXPERIMENTAL"), arm("Drug X", "ACTIVE_COMPARATOR")]
    assert find_control(("Drug Y + Placebo for X", "Drug X"), arms) == (1, "active", "arm type")


def test_an_unsettled_placebo_title_is_never_settled_by_a_control_phrase():
    # Without this, "standard of care" would make the drug arm the control.
    assert find_control(("Placebo + SOC", "Drug X plus standard of care"), []) is None


@pytest.mark.parametrize(
    "titles",
    [
        ("Arm B: Placebo + MTX", "Arm A: Drug X + MTX"),
        ("Placebo, BID", "Drug X 10 mg BID"),
        ("2) Placebo and Methotrexate", "1) Drug X and Methotrexate"),
        ("Placebo Patch, 0 mg", "Drug Patch, 5 mg"),
    ],
)
def test_arm_labels_doses_and_schedules_do_not_hide_a_placebo_control(titles):
    assert find_control(titles, []) == (0, "placebo", "placebo title")


def test_a_placebo_arm_with_a_dose_is_still_pure_when_both_titles_name_placebo():
    titles = ("Drug X 10 mg + Placebo", "Placebo (0 mg) BID")
    assert find_control(titles, []) == (1, "placebo", "placebo title")


@pytest.mark.parametrize("title", ["Ketoconazole Shampoo", "Vehicles Study Drug X"])
def test_placebo_words_are_whole_words(title):
    assert find_control((title, "Selenium Lotion"), []) is None


# --- the control by registered arm type -------------------------------------------


def test_arm_types_mark_an_active_comparator():
    arms = [arm("New Drug", "EXPERIMENTAL"), arm("Old Drug", "ACTIVE_COMPARATOR")]
    assert find_control(("New Drug", "Old Drug"), arms) == (1, "active", "arm type")


def test_arm_titles_are_matched_ignoring_case_spacing_and_punctuation():
    arms = [arm("New-Drug (20mg)", "EXPERIMENTAL"), arm("Old Drug", "ACTIVE_COMPARATOR")]
    assert find_control(("new drug 20 mg", "OLD DRUG"), arms) == (1, "active", "arm type")


def test_a_placebo_arm_is_the_control_against_an_active_comparator_arm():
    # The drug arm is mistyped and the placebo is called something else.
    arms = [arm("Drug X", "ACTIVE_COMPARATOR"), arm("Sugar pill", "PLACEBO_COMPARATOR")]
    assert find_control(("Drug X", "Sugar Pill"), arms) == (1, "placebo", "arm type")


def test_two_active_comparators_have_no_control():
    arms = [arm("Drug A", "ACTIVE_COMPARATOR"), arm("Drug B", "ACTIVE_COMPARATOR")]
    assert find_control(("Drug A", "Drug B"), arms) is None


def test_two_experimental_arms_have_no_control():
    arms = [arm("Drug A 10 mg", "EXPERIMENTAL"), arm("Drug A 20 mg", "EXPERIMENTAL")]
    assert find_control(("Drug A 10 mg", "Drug A 20 mg"), arms) is None


def test_a_comparator_against_an_arm_of_unclear_type_has_no_control():
    arms = [arm("Drug X Group", "ACTIVE_COMPARATOR"), arm("Iron Group", "OTHER")]
    assert find_control(("Drug X Group", "Iron Group"), arms) is None


def test_an_unregistered_title_is_the_treatment_when_the_trial_has_one_comparator():
    arms = [arm("New Drug", "EXPERIMENTAL"), arm("Old Drug", "ACTIVE_COMPARATOR")]
    assert find_control(("New Drug 10 mg (pooled)", "Old Drug"), arms) == (1, "active", "arm type")


def test_an_unregistered_title_is_not_trusted_when_the_trial_has_several_comparators():
    # The unmatched group could be the other comparator under a different name.
    arms = [
        arm("Drug X", "EXPERIMENTAL"),
        arm("Active Control", "ACTIVE_COMPARATOR"),
        arm("Untreated Control", "NO_INTERVENTION"),
    ]
    assert find_control(("Active Control", "Untreated"), arms) is None


def test_an_unregistered_title_that_reads_as_a_control_is_not_the_treatment():
    arms = [arm("Pravastatin", "ACTIVE_COMPARATOR"), arm("Something Else", "EXPERIMENTAL")]
    assert find_control(("Pravastatin", "Control Group"), arms) is None


def test_arms_with_colliding_titles_are_not_matched():
    arms = [arm("Arm 1", "EXPERIMENTAL"), arm("Arm-1", "ACTIVE_COMPARATOR")]
    assert find_control(("Arm 1", "Drug B"), arms) is None


# --- the control by a phrase in the title ------------------------------------------


@pytest.mark.parametrize(
    ("titles", "control"),
    [
        (("Drug X", "Standard of Care"), 1),
        (("Control", "Drug X"), 0),
        (("Drug X", "Observation"), 1),
        (("No Treatment", "Drug X"), 0),
        (("Active Surveillance", "Drug X"), 0),
    ],
)
def test_a_control_phrase_is_the_last_resort(titles, control):
    assert find_control(titles, []) == (control, "unspecified", "control word")


@pytest.mark.parametrize(
    "titles",
    [
        ("Drug X Standard Dose", "Drug X High Dose"),
        ("Drug X Alone", "Drug X + Drug Y"),
        ("Controlled-release Drug X", "Drug X"),
        ("Erlotinib", "Gefitinib"),
        ("Standard of Care", "Usual Care"),
    ],
)
def test_titles_that_do_not_name_one_control(titles):
    assert find_control(titles, []) is None


# --- sorting a trial's analyses ----------------------------------------------------


def study(outcomes, arms=()):
    return {
        "protocolSection": {"armsInterventionsModule": {"armGroups": list(arms)}},
        "resultsSection": {"outcomeMeasuresModule": {"outcomeMeasures": outcomes}},
    }


def outcome(groups, analyses, kind="PRIMARY"):
    return {
        "type": kind,
        "groups": [{"id": group_id, "title": title} for group_id, title in groups.items()],
        "analyses": analyses,
    }


def analysis(*group_ids, p="0.03", test_type="SUPERIORITY"):
    record = {"groupIds": list(group_ids), "nonInferiorityType": test_type}
    if p is not None:
        record["pValue"] = p
    return record


GROUPS = {"OG000": "Placebo", "OG001": "Drug X 10 mg", "OG002": "Drug X 20 mg"}


def test_each_dose_against_placebo_is_a_comparison():
    trial = study([outcome(GROUPS, [analysis("OG000", "OG001"), analysis("OG000", "OG002")])])
    comparisons, rejected = classify_analyses(trial)
    assert rejected == []
    assert [c.treatment_id for c in comparisons] == ["OG001", "OG002"]
    assert {c.control_id for c in comparisons} == {"OG000"}
    assert comparisons[0].treatment_title == "Drug X 10 mg"


def test_every_analysis_with_a_p_value_is_accounted_for():
    analyses = [
        analysis("OG001"),
        analysis("OG001", "OG001"),
        analysis("OG000", "OG001", "OG002"),
        analysis("OG000", "OG001", test_type="NON_INFERIORITY"),
        analysis("OG001", "OG002"),
        analysis("OG000", "OG009"),
        analysis("OG000", "OG002"),
    ]
    comparisons, rejected = classify_analyses(study([outcome(GROUPS, analyses)]))
    assert rejected == [
        SINGLE_GROUP,
        SINGLE_GROUP,
        MANY_GROUPS,
        NON_INFERIORITY,
        NO_CONTROL,
        NO_CONTROL,
    ]
    assert len(comparisons) == 1
    assert len(comparisons) + len(rejected) == len(analyses)


@pytest.mark.parametrize("p", [None, "", " "])
def test_an_analysis_without_a_p_value_is_ignored(p):
    trial = study([outcome(GROUPS, [analysis("OG000", "OG001", p=p)])])
    assert classify_analyses(trial) == ([], [])


@pytest.mark.parametrize("p", ["NS", "NA", "1.7"])
def test_an_analysis_with_an_unusable_p_value_is_still_a_comparison(p):
    # The test was run, so the arm must not silently vanish from the trial.
    trial = study([outcome(GROUPS, [analysis("OG000", "OG001", p=p)])])
    comparisons, rejected = classify_analyses(trial)
    assert len(comparisons) == 1 and rejected == []


def test_a_missing_test_type_is_treated_as_superiority():
    trial = study([outcome(GROUPS, [{"groupIds": ["OG000", "OG001"], "pValue": "<0.001"}])])
    comparisons, rejected = classify_analyses(trial)
    assert len(comparisons) == 1 and rejected == []


REFERENCE_GROUPS = {"OG000": "Placebo", "OG001": "New Drug", "OG002": "Duloxetine"}


def test_a_reference_drug_against_placebo_is_not_a_treatment_comparison():
    arms = [
        arm("New Drug", "EXPERIMENTAL"),
        arm("Duloxetine", "ACTIVE_COMPARATOR"),
        arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    analyses = [analysis("OG000", "OG001"), analysis("OG000", "OG002")]
    comparisons, rejected = classify_analyses(study([outcome(REFERENCE_GROUPS, analyses)], arms))
    assert [c.treatment_title for c in comparisons] == ["New Drug"]
    assert rejected == [REFERENCE_ARM]


def test_a_mistyped_drug_arm_is_kept_when_it_is_the_only_treatment_compared():
    # The trial has an experimental arm (an open-label extension), but the only
    # comparison is the mistyped drug arm against placebo. Dropping it would lose the trial.
    arms = [
        arm("Drug X", "ACTIVE_COMPARATOR"),
        arm("Placebo", "PLACEBO_COMPARATOR"),
        arm("Open-label Extension", "EXPERIMENTAL"),
    ]
    groups = {"OG000": "Placebo", "OG001": "Drug X"}
    comparisons, rejected = classify_analyses(
        study([outcome(groups, [analysis("OG000", "OG001")])], arms)
    )
    assert [c.treatment_title for c in comparisons] == ["Drug X"]
    assert rejected == []


def test_a_drug_arm_mistyped_as_a_placebo_comparator_is_not_a_reference():
    arms = [
        arm("Drug X 75 mg", "EXPERIMENTAL"),
        arm("Drug X 150 mg", "PLACEBO_COMPARATOR"),
        arm("Placebo", "PLACEBO_COMPARATOR"),
    ]
    groups = {"OG000": "Placebo", "OG001": "Drug X 75 mg", "OG002": "Drug X 150 mg"}
    analyses = [analysis("OG000", "OG001"), analysis("OG000", "OG002")]
    comparisons, rejected = classify_analyses(study([outcome(groups, analyses)], arms))
    assert len(comparisons) == 2 and rejected == []


def test_secondary_outcomes_are_ignored():
    trial = study([outcome(GROUPS, [analysis("OG000", "OG001")], kind="SECONDARY")])
    assert classify_analyses(trial) == ([], [])


@pytest.mark.parametrize("record", [{}, {"protocolSection": {}}])
def test_a_trial_without_results_has_no_comparisons(record):
    assert classify_analyses(record) == ([], [])


def test_the_first_eligible_outcome_skips_one_without_a_control_comparison():
    trial = study(
        [
            outcome(GROUPS, [analysis("OG001")]),
            outcome(GROUPS, [analysis("OG000", "OG001")]),
            outcome(GROUPS, [analysis("OG000", "OG002")]),
        ]
    )
    comparisons, _ = classify_analyses(trial)
    chosen = first_eligible_outcome(comparisons)
    assert [c.outcome_index for c in chosen] == [1]
    assert first_eligible_outcome([]) == []


# --- dose ladders -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("title", "expected"),
    [
        ("Apremilast 30 mg", (30.0, "mg")),
        ("Drug 0,5 mg/kg", (0.5, "mg/kg")),
        ("Vitamin D 1,000 IU", (1000.0, "iu")),
        ("Insulin 10 Units", (10.0, "unit")),
        ("Insulin 1 unit", (1.0, "unit")),
        ("Drug X 0.05% Cream", (0.05, "%")),
        ("Drug X 50 µg", (50.0, "mcg")),
        ("Day 1 Drug X 10 mg", (10.0, "mg")),
        ("Drug X 10 mg/day", None),
        ("Drug X 10 to 20 mg", None),
        ("Drug X 10 – 20 mg", None),
        ("Drug X 2x10 mg", None),
        ("Drug X", None),
        ("Drug 10 mg then 20 mg", None),
        ("Drug X 10-20 mg", None),
        ("Drug X 2 x 10 mg", None),
    ],
)
def test_dose_of(title, expected):
    assert dose_of(title) == expected


def test_the_highest_dose_arm_of_a_clean_ladder():
    arms = {"OG001": "Drug 10 mg", "OG002": "Drug 40 mg", "OG003": "Drug 20 mg"}
    assert highest_dose_arm(arms) == "OG002"
    assert highest_dose_arm({"A": "Vitamin D 400 IU", "B": "Vitamin D 1,000 IU"}) == "B"


@pytest.mark.parametrize(
    "arms",
    [
        {"A": "Drug 10 mg"},
        {"A": "Drug 10 mg", "B": "Drug high dose"},
        {"A": "Drug 10 mg", "B": "Drug 2 mg/kg"},
        {"A": "Drug X 10 mg", "B": "Drug X 10 mg"},
        {"A": "Drug X 10 mg BID", "B": "Drug X 15 mg QD"},
        {"A": "Drug X 10 mg", "B": "Drug Y 20 mg"},
        {"A": "Drug X 10 mg + Drug Y", "B": "Drug X + Drug Y 20 mg"},
    ],
)
def test_no_highest_dose_without_a_clean_ladder(arms):
    assert highest_dose_arm(arms) is None
