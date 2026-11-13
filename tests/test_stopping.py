import pytest

from trial_benchmark import stopping as st


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Slow accrual", st.RECRUITMENT),
        ("Study was not continued due to lack of enrollment.", st.RECRUITMENT),
        ("Not enough patients could be found", st.RECRUITMENT),
        ("Sponsor decision based on slow recruitment", st.RECRUITMENT),
        ("Futility at the interim analysis", st.EFFICACY),
        ("No additional efficacy seen over the standard", st.EFFICACY),
        ("The study drug was ineffective", st.EFFICACY),
        ("DSMB found a higher than expected risk, safety concerns", st.SAFETY),
        ("Terminated due to chemoradiotherapy toxicity", st.SAFETY),
        ("Lack of funding", st.BUSINESS),
        ("Sponsor decision based on portfolio prioritization", st.BUSINESS),
        ("The PI left the institution", st.OTHER),
        ("Study drug out of specification", st.OTHER),
    ],
)
def test_each_stop_reason_category_is_found(text, expected):
    assert st.classify_stop_reason(text) == expected


@pytest.mark.parametrize("text", [None, "", "   \n"])
def test_a_missing_or_blank_reason_is_no_reason_given(text):
    assert st.classify_stop_reason(text) == st.NO_REASON


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("No patients accrued.", st.RECRUITMENT),
        ("lack of response to study therapy", st.EFFICACY),
        ("CRx-102-006 study results, negative", st.EFFICACY),
        (
            "Overall response rate was 3% (only 1 out of 33 participants had confirmed PR).",
            st.EFFICACY,
        ),
    ],
)
def test_patterns_the_first_hand_check_found_missed_are_still_caught(text, expected):
    assert st.classify_stop_reason(text) == expected


def test_providing_safety_data_is_not_a_safety_stop():
    text = "This study served its purpose in providing considerable safety data."
    assert st.classify_stop_reason(text) == st.OTHER


def test_a_real_safety_stop_that_mentions_data_still_counts():
    assert st.classify_stop_reason("Stopped: the safety data showed liver injury") == st.SAFETY


def test_recruitment_wins_over_a_business_word_in_the_same_reason():
    text = "Sponsor decision based on slow recruitment and new emerging drug combinations"
    assert st.classify_stop_reason(text) == st.RECRUITMENT


# ---- negated safety and efficacy phrases are business decisions, not safety or efficacy stops


@pytest.mark.parametrize(
    "text",
    [
        "Company decision to discontinue the study, not due to any safety or efficacy concerns",
        "Business Decision; No Safety Or Efficacy Concerns. (See below)",
        "Sponsor's decision, no safety concerns.",
        "Discontinuation not based on any safety concerns. Strategic decision.",
        "Strategic decision; not because of toxicity.",
    ],
)
def test_a_stop_that_says_what_it_was_not_about_is_a_business_stop(text):
    assert st.classify_stop_reason(text) == st.BUSINESS


def test_a_stop_unrelated_to_safety_or_efficacy_is_not_either():
    text = "Terminated for reasons unrelated to safety or efficacy."
    assert st.classify_stop_reason(text) == st.OTHER


def test_the_real_reason_survives_when_a_negated_one_sits_beside_it():
    text = "Terminated not based on safety concerns, but due to insufficient efficacy."
    assert st.classify_stop_reason(text) == st.EFFICACY


def test_no_efficacy_seen_is_still_an_efficacy_stop():
    assert (
        st.classify_stop_reason("No efficacy seen at the interim; no safety concern") == st.EFFICACY
    )


def test_not_continued_due_to_lack_of_efficacy_is_not_treated_as_negated():
    assert st.classify_stop_reason("Study was not continued due to lack of efficacy") == st.EFFICACY


# ---- recruitment needs a failure word, not just the word "enrolment"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Enrollment stopped due to toxicity", st.SAFETY),
        ("Study enrollment was terminated due to a corporate strategic decision", st.BUSINESS),
        ("No participants were enrolled in the study. PI decided not to proceed.", st.OTHER),
        ("Once enough data had accrued to decide on development", st.OTHER),
        ("Recruitment halted due to problems with study drug supply", st.OTHER),
        ("Study stopped because of slow disease progression", st.OTHER),
    ],
)
def test_mentioning_enrolment_is_not_a_recruitment_stop(text, expected):
    assert st.classify_stop_reason(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Poor recruitment",
        "Enrollment was slower than anticipated",
        "Unable to recruit large enough sample",
        "Insufficient enrollment; no safety concerns",
        "Only 3 patients enrolled in 2 years",
        "Lack of eligible patients",
        "Did not meet target accrual",
        "No enrollment.",
        "Accrual",
        "Low inclusion rate",
    ],
)
def test_a_failure_word_beside_the_recruitment_word_is_a_recruitment_stop(text):
    assert st.classify_stop_reason(text) == st.RECRUITMENT


# ---- "interim analysis" alone is not a reason; stops for benefit are their own category


def test_an_interim_analysis_with_no_failure_word_is_not_efficacy():
    assert st.classify_stop_reason("An interim analysis was performed.") == st.OTHER


def test_an_interim_analysis_with_a_failure_word_is_efficacy():
    text = "Interim analysis showed no significant difference between the arms."
    assert st.classify_stop_reason(text) == st.EFFICACY


@pytest.mark.parametrize(
    "text",
    [
        "Stopped early for overwhelming efficacy at interim analysis",
        "After a positive interim analysis, the decision was made to terminate the study",
        "Met primary endpoint",
        "Study was terminated due to positive results seen in statistical analysis.",
    ],
)
def test_a_stop_for_demonstrated_benefit_is_its_own_category(text):
    assert st.classify_stop_reason(text) == st.BENEFIT


@pytest.mark.parametrize(
    "text",
    [
        "Interim analysis suggested no statistical superiority over historic controls",
        "Lack of evidence of superiority",
        "Failed to meet the primary objective of demonstrating superiority over placebo",
        "Low probability of positive outcome",
    ],
)
def test_a_denied_benefit_is_not_a_stop_for_benefit(text):
    assert st.classify_stop_reason(text) != st.BENEFIT


# ---- short stems are word-bound


def test_fundamental_is_not_funding():
    assert st.classify_stop_reason("Fundamental change in standard of care") == st.OTHER


def test_company_is_a_word_not_a_stem():
    assert st.classify_stop_reason("Company decision") == st.BUSINESS
    assert st.classify_stop_reason("Accompanying study closed") == st.OTHER


def test_slow_needs_a_recruitment_word_beside_it():
    assert st.classify_stop_reason("Slow enrolment") == st.RECRUITMENT
    assert st.classify_stop_reason("Slowly progressing disease made the endpoint unreachable") == (
        st.OTHER
    )


# ---- the three efficacy patterns the first hand read did not cover


@pytest.mark.parametrize(
    "text",
    [
        "Non-inferiority could not be shown at interim",
        "Efficacy not sufficient to continue",
        "Benefit/risk does not support continued development",
    ],
)
def test_wider_efficacy_phrasings_are_caught(text):
    assert st.classify_stop_reason(text) == st.EFFICACY


def test_the_category_list_has_every_label_the_classifier_can_return():
    seen = {
        st.classify_stop_reason(t)
        for t in (
            "slow accrual",
            "futility",
            "toxicity",
            "funding",
            "?",
            None,
            "stopped early for overwhelming efficacy",
        )
    }
    assert seen == set(st.REASONS)
