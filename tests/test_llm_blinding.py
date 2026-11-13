import hashlib
import json

import pytest
from test_llm_predictor import FORBIDDEN, study

from elman_exercise import llm_blinding as lb
from elman_exercise import llm_predictor as lp

ARMS = {
    "armGroups": [
        {
            "label": "Zorbitrex 10 mg",
            "type": "EXPERIMENTAL",
            "description": "ABT-494 10 mg once daily, plus zorbimab every 2 weeks (Brandex).",
            "interventionNames": [
                "Drug: Zorbitrex hydrochloride 10 mg tablets",
                "Biological: Zorbimab",
            ],
        },
        {
            "label": "Placebo",
            "type": "PLACEBO_COMPARATOR",
            "description": "Matching placebo for Zorbitrex, standard of care.",
            "interventionNames": ["Drug: Placebo for Zorbitrex", "Other: Standard of care"],
        },
    ],
    "interventions": [
        {
            "type": "DRUG",
            "name": "Zorbitrex hydrochloride 10 mg tablets",
            "otherNames": ["ABT-494", "Brandex"],
        },
        {"type": "BIOLOGICAL", "name": "Zorbimab", "otherNames": ["MK-0941"]},
        {"type": "DRUG", "name": "Whizzoline sodium"},
        {"type": "DRUG", "name": "Placebo for Zorbitrex"},
        {"type": "OTHER", "name": "Standard of care"},
    ],
}
OUTCOMES = {
    "primaryOutcomes": [
        {
            "measure": "Change in FEV1 on ZORBITREX versus MK-0941 at week 12",
            "timeFrame": "12 weeks",
        }
    ]
}


def record() -> dict:
    return lp.extract_record(study(armsInterventionsModule=ARMS, outcomesModule=OUTCOMES))


def blinded_text() -> str:
    return lp.render_trial(record(), "blinded", 1)


def test_each_intervention_gets_one_placeholder_by_type_and_the_same_one_everywhere():
    names = lb.placeholders(record()["intervention_details"])
    assert names["Zorbitrex hydrochloride 10 mg tablets"] == "Drug A"
    assert names["Zorbimab"] == "Biological A"  # its own series
    assert names["Whizzoline sodium"] == "Drug B"
    text = blinded_text()
    assert "Interventions: drug: Drug A; biological: Biological A; drug: Drug B;" in text
    assert (
        "- Drug A 10 mg (experimental): Drug A 10 mg once daily, plus Biological A every 2" in text
    )
    assert "[Drug A; Biological A]" in text  # the arm's own list uses the same placeholders


def test_company_codes_other_names_salt_forms_and_case_variants_are_replaced():
    text = blinded_text()
    for gone in (
        "ABT-494",
        "Brandex",
        "MK-0941",
        "Zorbitrex",
        "zorbimab",
        "ZORBITREX",
        "hydrochloride",
    ):
        assert gone not in text and gone.lower() not in text.lower()
    assert "Change in FEV1 on Drug A versus Biological A at week 12" in text  # outcome text too
    assert "(Drug A)." in text  # the brand name in brackets


def test_placebo_and_control_words_doses_and_schedules_stay():
    text = blinded_text()
    assert "Matching placebo for Drug A, standard of care." in text
    assert "placebo comparator" in text and "other: Standard of care" in text
    assert "10 mg once daily" in text and "every 2 weeks" in text


def test_a_name_is_replaced_as_a_whole_token_only():
    details = [{"type": "drug", "name": "Brandex", "other_names": []}]
    replace = lb.replacer(lb.alias_map(details))
    assert (
        replace("Brandex and brandex, Brandexton, XBrandex")
        == "Drug A and Drug A, Brandexton, XBrandex"
    )


def test_a_combination_name_is_taken_apart_and_a_dose_is_not_an_alias():
    variants = lb.name_variants("Imipenem/Cilastatin 500 mg/500 mg")
    assert {"Imipenem", "Cilastatin"} <= variants
    assert not any(v in {"500 mg", "mg"} for v in variants)
    assert lb.name_variants("Placebo") == set() and lb.name_variants("Standard of care") == set()


def test_the_start_date_is_dropped_and_only_in_the_blinded_prompt():
    assert "Start date" not in blinded_text() and "2019" not in blinded_text()
    assert "Start date: 2019-03" in lp.render_trial(record(), "masked", 1)


@pytest.mark.parametrize("variant", lp.VARIANTS)
def test_nothing_forbidden_comes_back_in_any_variant(variant):
    text = lp.render_prompt([record()], variant)
    for marker in FORBIDDEN.values():
        assert marker not in text


def test_blinded_hides_id_title_and_sponsor_name_as_masked_does():
    text = blinded_text()
    for hidden in ("NCT01234567", "Official Title", "Acme"):
        assert hidden not in text
    assert "industry" in text


def test_the_variants_differ_only_in_the_fields_they_are_meant_to():
    identified, masked, blinded = (
        lp.render_trial(record(), v, 1).splitlines() for v in lp.VARIANTS
    )
    # Identified to masked: only the id, title and the sponsor's name.
    dropped = [x.split(":")[0] for x in identified if x not in masked]
    assert dropped == ["Registry id", "Title", "Lead sponsor"]
    # Masked to blinded: the start date goes and the lines that name an intervention change.
    # Every other line (phase, sponsor class, design, conditions, eligibility, time frames) is
    # the same.
    changed = [x for x in masked if x not in blinded]
    assert changed[0] == "Start date: 2019-03"
    assert all(
        x.startswith(("Interventions:", "- Zorbitrex", "- Placebo", "- Change in FEV1"))
        for x in changed[1:]
    )
    same = [x for x in blinded if x in masked]
    for kept in ("Phase: phase2", "Lead sponsor: industry", "Conditions: Asthma"):
        assert kept in same
    assert any(x.startswith(("Design:", "Eligibility:")) for x in same)
    assert [x for x in blinded if "time frame" in x] and "(time frame: 12 weeks)" in blinded[-1]


def test_leaked_words_are_the_trials_own_name_words_that_survive():
    rec = record()
    assert lb.leaked_tokens(blinded_text(), rec) == []
    leaky = blinded_text() + "\nPrimary outcomes: whizzoline levels"
    assert lb.leaked_tokens(leaky, rec) == ["whizzoline"]
    # Doses, salts, control words and the drug's type words are not counted.
    assert lb.leaked_tokens("tablets 10 mg hydrochloride placebo standard of care drug", rec) == []


def test_a_drug_name_inside_a_condition_is_replaced_too():
    rec = lp.extract_record(
        study(
            conditionsModule={"conditions": ["Lenalidomide-Refractory Myeloma"]},
            armsInterventionsModule={
                "armGroups": [],
                "interventions": [{"type": "DRUG", "name": "Lenalidomide"}],
            },
        )
    )
    text = lp.render_trial(rec, "blinded", 1)
    assert "Conditions: Drug A-Refractory Myeloma" in text and lb.leaked_tokens(text, rec) == []
    assert lp.leak_counts([rec]) == {"trials": 1, "any": 0}


def test_a_name_that_is_only_a_code_is_replaced_even_inside_a_placebo_name():
    assert lb.name_variants("K-877") == {"K-877", "k-877", "k 877", "k877"}
    rec = lp.extract_record(
        study(
            armsInterventionsModule={
                "armGroups": [{"label": "K-877 or placebo", "interventionNames": ["Drug: K-877"]}],
                "interventions": [
                    {"type": "DRUG", "name": "K-877"},
                    {"type": "DRUG", "name": "Placebo (for K-877)"},
                ],
            }
        )
    )
    text = lp.render_trial(rec, "blinded", 1)
    assert "K-877" not in text
    assert "Interventions: drug: Drug A; drug: Placebo (for Drug A)" in text
    assert "- Drug A or placebo" in text


# ---------------------------------------------------------------- the defects found in review


def trial(interventions, arms=(), **extra):
    """A record from interventions given as (type, name, other names) and arms as (label, type, [names])."""
    module = {
        "interventions": [{"type": t, "name": n, "otherNames": o} for t, n, o in interventions],
        "armGroups": [
            {"label": label, "type": kind, "interventionNames": names}
            for label, kind, names in arms
        ],
    }
    return lp.extract_record(study(armsInterventionsModule=module, **extra))


def blind(rec) -> str:
    return lp.render_trial(rec, "blinded", 1)


def test_a_drug_with_standard_therapy_in_its_name_is_blinded_and_the_control_words_stay():
    rec = trial(
        [
            ("BIOLOGICAL", "Placebo plus standard therapy", []),
            ("BIOLOGICAL", "Belimumab 10 mg/kg plus standard therapy", ["BENLYSTA™"]),
            ("DRUG", "Standard therapy", []),
        ],
        [
            (
                "Placebo plus standard therapy",
                "PLACEBO_COMPARATOR",
                ["Biological: Placebo plus standard therapy"],
            ),
            (
                "Belimumab 10 mg/kg plus standard therapy",
                "EXPERIMENTAL",
                ["Biological: Belimumab 10 mg/kg plus standard therapy"],
            ),
        ],
    )
    text = blind(rec)
    assert "Belimumab" not in text and "BENLYSTA" not in text
    assert "biological: Biological A" in text and "Placebo plus standard therapy" in text
    assert "Biological A 10 mg/kg plus standard therapy (experimental)" in text
    assert lb.active_flags(rec["intervention_details"]) == [False, True, False]


def test_a_vehicle_of_a_listed_drug_is_a_control_and_the_company_code_in_it_is_replaced():
    rec = trial(
        [("DRUG", "IDP 121 Lotion", []), ("DRUG", "IDP-121 Vehicle Lotion", [])],
        [("IDP-121 Vehicle Lotion", "PLACEBO_COMPARATOR", ["Drug: IDP-121 Vehicle Lotion"])],
    )
    text = blind(rec)
    assert "IDP" not in text and "idp" not in text.lower()
    assert "Interventions: drug: Drug A; drug: Drug A Vehicle Lotion" in text
    assert lb.leaked_tokens(text, rec) == []


def test_a_comparator_is_blinded_and_not_given_the_test_drugs_placeholder():
    rec = trial(
        [
            ("DRUG", "Indacaterol acetate/Mometasone furoate", ["QMF149"]),
            ("DRUG", "Mometasone furoate", []),
        ],
        [
            ("QMF149", "EXPERIMENTAL", ["Drug: Indacaterol acetate/Mometasone furoate"]),
            ("Mometasone furoate 400 µg", "ACTIVE_COMPARATOR", ["Drug: Mometasone furoate"]),
        ],
    )
    text = blind(rec)
    assert "- Drug A (experimental)" in text
    assert "- Drug B 400 µg (active comparator): Drug B 400 µg" not in text  # no description here
    assert "- Drug B 400 µg (active comparator)" in text and "[Drug B]" in text
    assert "ometasone" not in text and "QMF149" not in text


def test_an_alias_claimed_by_two_different_drugs_is_dropped_not_given_to_the_first():
    details = [
        {"type": "drug", "name": "Alphadrug (XYZ)", "other_names": []},
        {"type": "drug", "name": "Betadrug (XYZ)", "other_names": []},
    ]
    found = lb.alias_map(details)
    assert "xyz" not in found
    assert found["alphadrug"] == "Drug A" and found["betadrug"] == "Drug B"


def test_a_name_repeated_in_the_list_gets_one_placeholder():
    rec = trial([("DRUG", "BMS-986036", []), ("DRUG", "BMS-986036", []), ("DRUG", "Placebo", [])])
    text = blind(rec)
    assert "Interventions: drug: Drug A; drug: Placebo" in text
    assert lb.placeholders(rec["intervention_details"]) == {"BMS-986036": "Drug A"}


def test_two_doses_of_one_drug_share_a_placeholder_and_a_different_drug_does_not():
    rec = trial(
        [
            ("DRUG", "Ustekinumab approximately 6 mg/kg (IV)", ["STELARA"]),
            ("DRUG", "Ustekinumab 90 mg (SC) Group 1", ["STELARA"]),
            ("DRUG", "Placebo (SC)", []),
            ("DRUG", "Vedolizumab", []),
        ],
        [("Group 1", "EXPERIMENTAL", ["Drug: Ustekinumab 90 mg (SC) Group 1"])],
        outcomesModule={
            "primaryOutcomes": [
                {"measure": "Ustekinumab versus vedolizumab", "timeFrame": "Week 8"}
            ]
        },
    )
    text = blind(rec)
    assert "Ustekinumab" not in text and "STELARA" not in text
    assert "Drug A versus Drug B" in text and lb.leaked_tokens(text, rec) == []


def test_a_single_drug_name_inside_a_longer_listed_name_is_replaced_by_itself():
    rec = trial(
        [("DRUG", "Pembrolizumab 200 mg every 3 weeks", [])],
        [("Arm", "EXPERIMENTAL", ["Drug: Pembrolizumab 200 mg every 3 weeks"])],
        outcomesModule={
            "primaryOutcomes": [{"measure": "OS on pembrolizumab", "timeFrame": "5 years"}]
        },
    )
    text = blind(rec)
    assert "embrolizumab" not in text and "OS on Drug A" in text
    assert "every 3 weeks" not in lb.name_variants("Pembrolizumab 200 mg every 3 weeks")


@pytest.mark.parametrize("name", ["Drug X (low dose)", "Drug X, once daily", "Drug X high dose"])
def test_schedule_and_dose_words_never_become_aliases(name):
    variants = lb.name_variants(name.replace("Drug X", "Zorbitrex"))
    assert not {"low dose", "once daily", "high dose", "dose"} & variants
    rec = trial(
        [("DRUG", name.replace("Drug X", "Zorbitrex"), []), ("DRUG", "Placebo", [])],
        [("Low dose placebo once daily", "PLACEBO_COMPARATOR", ["Drug: Placebo"])],
    )
    assert "Low dose placebo once daily" in blind(rec)


@pytest.mark.parametrize(
    "name, written",
    [
        ("K-877 extended release", ["K-877", "K 877", "K877", "k-877"]),
        ("AZD-1222", ["AZD1222", "AZD 1222", "azd-1222"]),
        ("BI 655066", ["BI-655066", "BI655066", "bi 655066"]),
        ("BNT162b2", ["BNT-162b2", "bnt162b2"]),
        ("mRNA-1273", ["mRNA 1273", "MRNA1273"]),
    ],
)
def test_short_company_codes_are_replaced_in_every_written_form_and_the_leak_check_sees_them(
    name, written
):
    rec = trial([("DRUG", name, [])])
    for form in written:
        assert lb.leaked_tokens(f"given {form} daily", rec), form
        assert "Drug A" in lb.replacer(lb.alias_map(rec["intervention_details"]))(f"{form} 5 mg")
    assert lb.leaked_tokens("Dose 100 and Cohort 101", trial([("DRUG", "Dose 100", [])])) == []


def test_a_dose_is_not_a_company_code():
    assert lb.company_codes("Vitamin D 1000 IU") == set()
    assert lb.company_codes("Cohort 101 and Part 200") == set()
    assert lb.company_codes("ABT-494 and GSP 301") == {("abt", "494"), ("gsp", "301")}


def test_a_combination_does_not_take_a_code_from_the_drug_listed_on_its_own():
    rec = trial(
        [("DRUG", "REGN2810", ["cemiplimab"]), ("DRUG", "REGN2810/chemo/ipi", ["cemiplimab"])],
        outcomesModule={"primaryOutcomes": [{"measure": "REGN2810 alone", "timeFrame": "1 year"}]},
    )
    assert "REGN2810 alone" not in blind(rec) and "Drug A alone" in blind(rec)


def test_the_acronym_study_id_and_short_sponsor_names_are_masked_on_whole_words_only():
    rec = lp.extract_record(
        study(
            identificationModule={
                "nctId": "NCT01234567",
                "officialTitle": "T",
                "acronym": "KEYSTONE-24",
                "orgStudyIdInfo": {"id": "CAIN457A2302"},
            },
            sponsorCollaboratorsModule={
                "leadSponsor": {"name": "Eli Lilly and Company", "class": "INDUSTRY"}
            },
            armsInterventionsModule={
                "armGroups": [
                    {
                        "label": "Arm",
                        "description": "In KEYSTONE-24 (CAIN457A2302) Lilly supplies; a combination of keystone-24ish",
                    }
                ],
                "interventions": [],
            },
        )
    )
    for variant in ("masked", "blinded"):
        text = lp.render_trial(rec, variant, 1)
        assert "KEYSTONE-24 " not in text and "CAIN457A2302" not in text and "Lilly" not in text
        assert "combination of keystone-24ish" in text  # not inside another word
        assert "In [study name removed] ([id removed]) the sponsor supplies" in text
    assert "Lilly" in lp.render_trial(rec, "identified", 1)
    short = {
        "sponsor_name": "BI",
        "sponsor_class": "industry",
        "collaborators": [],
        "acronym": None,
        "own_ids": [],
    }
    assert lp.mask("BI paid for the combination", short) == "the sponsor paid for the combination"


def test_text_is_cut_after_replacing_so_a_name_at_the_cut_does_not_leak():
    long_text = "x" * 290 + " pembrolizumab again"
    rec = trial(
        [("DRUG", "Pembrolizumab", [])],
        [("Arm", "EXPERIMENTAL", ["Drug: Pembrolizumab"])],
        outcomesModule={"primaryOutcomes": [{"measure": long_text, "timeFrame": "1 year"}]},
    )
    assert "pembroli" in lp.render_trial(rec, "masked", 1)  # a name cut in half would show here
    text = blind(rec)
    assert "pembro" not in text.lower() and "Drug A" in text
    assert lp._cut(long_text).endswith("…") and len(lp._cut(long_text)) <= lp.MAX_TEXT


def test_the_blinded_prompt_cuts_post_hoc_phrases_and_keeps_the_rest_of_the_time_frame():
    rec = trial(
        [("DRUG", "Zorbitrex", [])],
        outcomesModule={
            "primaryOutcomes": [
                {
                    "measure": "PFS",
                    "timeFrame": "Up to 30 months (through database cut-off date of 26 Oct 2017)",
                },
                {
                    "measure": "OS",
                    "timeFrame": "5 years; the median duration of follow-up was 21.8 months",
                },
            ]
        },
    )
    assert rec["post_hoc"] is True
    masked, blinded = lp.render_trial(rec, "masked", 1), blind(rec)
    assert "cut-off date of 26 Oct 2017" in masked and "median duration" in masked
    assert "- PFS (time frame: Up to 30 months)" in blinded
    assert "- OS (time frame: 5 years)" in blinded
    assert "cut-off" not in blinded and "median" not in blinded


# ---------------------------------------------------------------- did the model name the drug?


def named_trial():
    return trial(
        [
            ("DRUG", "Zorbitrex 10 mg tablets", ["Brandex", "ABT-494"]),
            ("DRUG", "Whizzoline", []),
            ("DRUG", "Placebo", []),
        ],
        [
            ("A", "EXPERIMENTAL", ["Drug: Zorbitrex 10 mg tablets"]),
            ("B", "ACTIVE_COMPARATOR", ["Drug: Whizzoline"]),
        ],
    )


@pytest.mark.parametrize(
    "guess, expected",
    [
        ("Zorbitrex", True),
        ("zorbitrex 10 mg", True),
        ("Brandex (an oral drug)", True),
        ("brandex", True),
        ("ABT 494", True),
        ("abt494", True),
        ("The main drug is ZORBITREX tablets", True),
        ("Whizzoline", False),  # the comparator, not the tested drug
        ("a monoclonal antibody", False),
        ("Zorbi", False),  # part of a word is not the name
        ("drug", False),
        ("", False),
        (None, False),
    ],
)
def test_a_drug_guess_matches_any_alias_of_the_tested_intervention(guess, expected):
    assert lb.guess_matches(guess, named_trial()) is expected


def test_with_no_experimental_arm_the_active_interventions_are_the_tested_ones():
    rec = trial(
        [("DRUG", "Zorbitrex", []), ("DRUG", "Placebo", [])], [("A", "OTHER", ["Drug: Zorbitrex"])]
    )
    assert lb.guess_matches("Zorbitrex", rec) and not lb.guess_matches("Placebo", rec)


def background_trial():
    """Zorbitrex is tested; metformin is a background drug both arms take."""
    return trial(
        [
            ("DRUG", "Zorbitrex", []),
            ("DRUG", "Metformin", []),
            ("DRUG", "Placebo", []),
            ("BIOLOGICAL", "Recombinant human monoclonal antibody Zorbimab", []),
        ],
        [
            ("A", "EXPERIMENTAL", ["Drug: Zorbitrex", "Drug: Metformin"]),
            ("B", "PLACEBO_COMPARATOR", ["Drug: Placebo", "Drug: Metformin"]),
        ],
    )


@pytest.mark.parametrize(
    "guess, expected",
    [
        ("Zorbitrex", True),
        ("metformin", False),  # the background drug both arms take is not the tested one
        ("monoclonal antibody", False),  # a class, not a name
        ("a recombinant human antibody", False),
        ("Zorbitrex plus metformin", True),  # it names the tested drug as well
    ],
)
def test_a_background_drug_or_a_class_word_is_not_the_tested_drug(guess, expected):
    assert lb.guess_matches(guess, background_trial()) is expected


def test_a_class_guess_does_not_match_a_long_name_that_holds_those_words():
    rec = trial(
        [("BIOLOGICAL", "Recombinant human monoclonal antibody Zorbimab", [])],
        [("A", "EXPERIMENTAL", ["Biological: Recombinant human monoclonal antibody Zorbimab"])],
    )
    assert not lb.guess_matches("monoclonal antibody", rec)
    assert not lb.guess_matches("human antibody", rec)
    assert lb.guess_matches("Zorbimab", rec)


def test_an_intervention_in_every_arm_is_still_tested_if_nothing_else_is_left():
    rec = trial(
        [("DRUG", "Zorbitrex", [])],
        [
            ("A", "EXPERIMENTAL", ["Drug: Zorbitrex"]),
            ("B", "ACTIVE_COMPARATOR", ["Drug: Zorbitrex"]),
        ],
    )
    assert lb.guess_matches("Zorbitrex", rec)


def test_text_names_the_drug_by_any_whole_alias_and_not_by_a_class_word_or_a_part_of_a_word():
    rec = named_trial()
    assert lb.text_names_drug("This is the ABT 494 programme", rec)
    assert lb.text_names_drug("brandex trials usually succeed", rec)
    assert not lb.text_names_drug("Zorbi is not a name", rec)
    assert not lb.text_names_drug("Whizzoline is the comparator", rec)
    assert not lb.text_names_drug(None, rec)
    assert not lb.text_names_drug("a monoclonal antibody", background_trial())


def test_the_known_gaps_are_listed_in_the_module_docstring_and_as_data():
    assert "Known gaps, unfixed on purpose" in lb.__doc__
    assert len(lb.KNOWN_GAPS) == 7
    assert all(gap.split(" (")[0][:20].strip() for gap in lb.KNOWN_GAPS)


# ---------------------------------------------------------------- the stored prompts still reproduce


def stored_batches(variant: str, picks: tuple[int, ...] = (0, -1)) -> list[dict]:
    """The first and last stored batch of one variant (the prompts are the same for both models)."""
    paths = sorted(lp.RAW_DIR.glob(f"haiku_{variant}_run0_b*.json"))
    return [json.loads(paths[i].read_text()) for i in picks]


@pytest.mark.skipif(
    not lp.RAW_DIR.exists() or not (lp.ROOT / "data" / "raw" / "ctgov").exists(),
    reason="needs the stored predictor calls and the raw download (both git-ignored)",
)
def test_stored_prompts_still_reproduce_exactly_from_the_current_rules():
    """If this fails, a blinding or masking rule changed after the predictions were stored.

    The stored predictions answer the stored prompts. A rule that changes a rendered prompt makes
    them answer one nobody can render again; see "Known gaps, unfixed on purpose" in `llm_blinding`.
    """
    batches = [b for variant in lp.VARIANTS for b in stored_batches(variant)]
    records = lp.load_records({i for b in batches for i in b["nct_ids"]})
    for batch in batches:
        prompt = lp.render_prompt([records[i] for i in batch["nct_ids"]], batch["variant"])
        assert hashlib.sha256(prompt.encode()).hexdigest() == batch["prompt_sha256"], (
            f"{batch['variant']} batch {batch['batch']} no longer renders to its stored prompt"
        )
