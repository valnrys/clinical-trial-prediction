import pytest

from trial_benchmark.labels import PValue, is_significant, label_trial, parse_p_value


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.03", PValue(0.03, "=")),
        ("=0.040", PValue(0.04, "=")),
        (" .5 ", PValue(0.5, "=")),
        ("<0.001", PValue(0.001, "<")),
        ("< 0.0001", PValue(0.0001, "<")),
        ("<=0.05", PValue(0.05, "≤")),
        ("≤0.05", PValue(0.05, "≤")),
        (">0.999", PValue(0.999, ">")),
        (">= 0.05", PValue(0.05, "≥")),
        ("0.0000", PValue(0.0, "<")),
        ("1", PValue(1.0, "=")),
    ],
)
def test_parse_p_value(text, expected):
    assert parse_p_value(text) == expected


@pytest.mark.parametrize(
    "text", [None, "", "NS", "NA", "-1.75", "<-0.001", "0. 9818", "1.5", "23", "0,03", "0.05*"]
)
def test_unusable_p_values_are_not_parsed(text):
    assert parse_p_value(text) is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0.049", True),
        ("0.05", False),
        ("0.5", False),
        ("0.000", True),
        ("<0.001", True),
        ("<0.05", True),
        ("<0.2", None),
        ("<=0.01", True),
        ("<=0.05", None),
        ("≤0.2", None),
        (">0.05", False),
        (">=0.05", False),
        (">0.999", False),
        (">0.01", None),
        ("NS", None),
    ],
)
def test_is_significant(text, expected):
    assert is_significant(parse_p_value(text)) is expected


DOSES = {"OG000": "Placebo", "OG001": "Drug X 10 mg", "OG002": "Drug X 20 mg"}


def trial(analyses, groups=DOSES, outcomes_before=(), arms=()):
    """A trial whose primary outcome has the given (control, arm, p-value) analyses."""
    outcome = {
        "type": "PRIMARY",
        "groups": [{"id": group_id, "title": title} for group_id, title in groups.items()],
        "analyses": [
            {"groupIds": [control, arm], "pValue": p, "nonInferiorityType": "SUPERIORITY"}
            for control, arm, p in analyses
        ],
    }
    return {
        "protocolSection": {"armsInterventionsModule": {"armGroups": list(arms)}},
        "resultsSection": {
            "outcomeMeasuresModule": {"outcomeMeasures": [*outcomes_before, outcome]}
        },
    }


def vs_placebo(*arm_p_values):
    return [("OG000", arm, p) for arm, p in arm_p_values]


def test_one_arm_that_beats_control():
    label = label_trial(trial(vs_placebo(("OG001", "0.01"))))
    assert (label.any_arm, label.all_arms, label.n_arms) == (True, True, 1)
    assert label.highest_dose is None and not label.has_dose_ladder
    assert label.control_kind == "placebo" and label.recognised_by == "placebo title"


def test_one_arm_that_does_not():
    label = label_trial(trial(vs_placebo(("OG001", "0.40"))))
    assert (label.any_arm, label.all_arms) == (False, False)


def test_only_the_low_dose_beats_control():
    label = label_trial(trial(vs_placebo(("OG001", "0.01"), ("OG002", "0.30"))))
    assert (label.any_arm, label.all_arms, label.highest_dose) == (True, False, False)
    assert label.has_dose_ladder and label.n_arms == 2


def test_only_the_high_dose_beats_control():
    label = label_trial(trial(vs_placebo(("OG001", "0.30"), ("OG002", "<0.001"))))
    assert (label.any_arm, label.all_arms, label.highest_dose) == (True, False, True)


def test_only_an_arms_first_listed_comparison_counts():
    # A failed primary analysis followed by a significant sensitivity analysis.
    label = label_trial(trial(vs_placebo(("OG001", "0.30"), ("OG001", "0.02"))))
    assert (label.any_arm, label.all_arms, label.n_arms) == (False, False, 1)
    # And a later unusable value does not unsettle a result already known.
    label = label_trial(trial(vs_placebo(("OG001", "0.30"), ("OG001", "<0.2"))))
    assert (label.any_arm, label.all_arms) == (False, False)


@pytest.mark.parametrize("p", ["<0.2", "<=0.05", "NS", "1.7"])
def test_a_p_value_that_does_not_settle_it_leaves_the_label_undetermined(p):
    label = label_trial(trial(vs_placebo(("OG001", p))))
    assert (label.any_arm, label.all_arms) == (None, None)


def test_a_known_result_can_settle_the_label_despite_an_unsettled_one():
    won = label_trial(trial(vs_placebo(("OG001", "0.01"), ("OG002", "<0.2"))))
    assert (won.any_arm, won.all_arms, won.highest_dose) == (True, None, None)
    lost = label_trial(trial(vs_placebo(("OG001", "0.60"), ("OG002", "NS"))))
    assert (lost.any_arm, lost.all_arms, lost.n_arms) == (None, False, 2)


def test_a_trial_with_placebo_and_an_active_comparator_is_labelled_against_placebo():
    groups = {"OG000": "Placebo", "OG001": "New Drug", "OG002": "Old Drug"}
    arms = [
        {"label": "New Drug", "type": "EXPERIMENTAL"},
        {"label": "Old Drug", "type": "ACTIVE_COMPARATOR"},
    ]
    # Loses to the old drug (listed first), beats placebo.
    analyses = [("OG002", "OG001", "0.50"), ("OG000", "OG001", "0.01")]
    label = label_trial(trial(analyses, groups=groups, arms=arms))
    assert (label.any_arm, label.control_kind, label.n_arms) == (True, "placebo", 1)


def test_with_two_non_placebo_controls_the_first_listed_is_used():
    groups = {"OG000": "Standard of Care", "OG001": "New Drug", "OG002": "Observation"}
    analyses = [("OG000", "OG001", "0.50"), ("OG002", "OG001", "0.01")]
    label = label_trial(trial(analyses, groups=groups))
    assert (label.any_arm, label.control_kind) == (False, "unspecified")


def test_no_highest_dose_when_a_higher_dose_was_reported_but_not_compared():
    groups = {**DOSES, "OG003": "Drug X 40 mg"}
    label = label_trial(trial(vs_placebo(("OG001", "0.01"), ("OG002", "0.01")), groups=groups))
    assert (label.any_arm, label.highest_dose, label.has_dose_ladder) == (True, None, False)


def test_an_untested_lower_dose_or_another_drug_does_not_block_the_highest_dose():
    groups = {**DOSES, "OG003": "Drug X 5 mg", "OG004": "Drug Y 80 mg"}
    label = label_trial(trial(vs_placebo(("OG001", "0.30"), ("OG002", "0.01")), groups=groups))
    assert (label.highest_dose, label.has_dose_ladder) == (True, True)


def test_the_label_uses_the_first_outcome_that_has_a_control_comparison():
    baseline_only = {
        "type": "PRIMARY",
        "groups": [{"id": "OG001", "title": "Drug X 10 mg"}],
        "analyses": [{"groupIds": ["OG001"], "pValue": "0.001"}],
    }
    label = label_trial(trial(vs_placebo(("OG001", "0.40")), outcomes_before=[baseline_only]))
    assert (label.outcome_index, label.any_arm) == (1, False)


def test_a_trial_with_no_control_comparison_has_no_label():
    groups = {"OG000": "Drug X 10 mg", "OG001": "Drug X 20 mg"}
    assert label_trial(trial([("OG000", "OG001", "0.01")], groups=groups)) is None
    assert label_trial({"protocolSection": {}}) is None
