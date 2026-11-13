"""Blinding a trial record: take out what a language model could recognise a trial by.

The "masked" prompt of `llm_predictor` hides the registry id, title, sponsor name, trial acronym
and the organisation's study id. A drug name is enough to recognise a trial, so the "blinded"
prompt goes further. It is the masked prompt with these changes, and nothing else:

- Every active drug, whether it is the test drug or a comparator, is replaced by a placeholder that
  keeps only its type and a letter: "Drug A", "Drug B", "Biological A". Only placebo-like
  controls keep their word (placebo, sham, vehicle, saline, "standard of care", "no treatment").
  The same drug gets the same placeholder throughout one trial, also when it is listed twice (two
  doses of one drug).
- The same replacement is made wherever the name turns up: in a condition, an arm label, an arm
  description, an arm's list of interventions, a primary outcome or a time frame. A name is found
  by its own text, its other names in the protocol (brand names, company codes), the name without
  its dose, schedule, salt or dosage-form words, each part of a combination ("A plus B"), each
  part in brackets, and any company code such as ABT-494, K-877, GSP 301, BNT162b2 or mRNA-1273
  inside a name, written with a hyphen, a space or neither. Matching ignores case and works on
  whole tokens only.
- An intervention whose name is a control plus a drug ("Belimumab plus standard therapy") is a
  drug: the drug is replaced, the control words stay. A name that is a placebo of a drug listed
  elsewhere ("Placebo for Zorbitrex") is a control, and the drug inside it is replaced.
- When two different drugs of one trial claim the same short text ("MF" in two names), the text is
  replaced by neither: a wrong placeholder would be worse than none. A part of a combination
  ("A + B") never overrides a drug listed under that name on its own.
- Doses and schedules are kept: they say how the trial was designed, not which trial it was.
- The start date is dropped altogether: the year helps place a drug in its development programme.
- Phrases a sponsor writes after the results are in ("through database cut-off date of 26 Oct
  2017", "the median duration of follow-up was 21.8 months") are cut out; see `llm_posthoc`.

Conditions, phase, design, eligibility, primary outcome measures and time frames stay. They are
the design the model is meant to reason from. A condition that holds a drug name ("Lenalidomide-
refractory myeloma") has the name replaced like any other text.

Blinding is imperfect. A drug-class word in an outcome ("PD-1 receptor occupancy"), an
abbreviation, a brand or device name that is not listed, an unusual endpoint or a named scale can
still point to a programme. `leaked_tokens` measures what is left: the words and company codes of
the trial's own intervention names that survive in the blinded text. It undercounts, because it
cannot see a name that was never listed as an intervention, nor a programme recognised from its
design. Name removal cannot blind a model to a published trial.

Known gaps, unfixed on purpose
------------------------------
The stored predictions were made from prompts rendered by the rules above, and each stored prompt
carries a hash (`tests/test_llm_blinding.py` re-renders a sample and compares). A rule that changes
a rendered prompt would make the stored predictions answer a prompt nobody can reproduce, so these
gaps are listed, not fixed. Counts are over the 300 sampled development trials (2017 to 2020), by
simple pattern searches on their blinded text, made when the predictions were stored:

- A drug name followed by a colon or a dash suffix can be left unblinded: 9 of 300 prompts show a
  word of a tested drug's name beside a colon or dash.
- A spaced company code whose letters are an ignored word is not read as a code: 0 of 300.
- "Placebo for X" is a control, but in an add-on design X is also given: 3 of 300 trials list
  a "Placebo for ..." intervention.
- A date written with a comma ("March 3, 2019") can survive the post-hoc clean-up: 0 of 300.
- The short name of an industry collaborator can survive: not counted.
- An abbreviation glossary in a description that names the drug can survive: not counted.
- Doubled placeholders such as "Drug B Drug B" appear: 3 of 300.
"""

from __future__ import annotations

import re
import string

from elman_exercise.comparisons import CONTROL_WORDS, DOSE, PLACEBO_FILLER, PLACEBO_WORDS
from elman_exercise.history import _WORDS_DROPPED as HISTORY_WORDS

KNOWN_GAPS = (
    (
        "a drug name followed by a colon or a dash suffix can be left unblinded (9 of 300 "
        "development prompts show a word of a tested drug's name beside a colon or dash)"
    ),
    "a spaced company code whose letters are an ignored word is not read as a code (0 of 300)",
    (
        '"Placebo for X" is treated as a control, but in an add-on design X is also given '
        '(3 of 300 trials list a "Placebo for ..." intervention)'
    ),
    'a date written with a comma ("March 3, 2019") can survive the post-hoc clean-up (0 of 300)',
    "the short name of an industry collaborator can survive (not counted)",
    "an abbreviation glossary in a description that names the drug can survive (not counted)",
    'doubled placeholders such as "Drug B Drug B" appear (3 of 300)',
)

TYPE_LABELS = {
    "drug": "Drug",
    "biological": "Biological",
    "device": "Device",
    "procedure": "Procedure",
    "radiation": "Radiation",
    "behavioral": "Behavioral",
    "genetic": "Genetic",
    "dietary supplement": "Supplement",
    "combination product": "Combination product",
    "diagnostic test": "Diagnostic test",
}
OTHER_LABEL = "Other"

# A company code is one to five letters and three or more digits: K-877, ABT-494, MK-0941,
# BNT162b2, mRNA-1273. A space in place of the hyphen is accepted only after two to five capital
# letters (GSP 301, BI 655066), so "Dose 100" is not a code. Only looked for inside intervention
# names, where a code is meant.
HYPHENATED_CODE = re.compile(
    r"(?<![A-Za-z0-9])([A-Za-z]{1,5})-?(\d{3,8}[A-Za-z0-9]{0,2})(?![A-Za-z0-9-])"
)
SPACED_CODE = re.compile(r"(?<![A-Za-z0-9])([A-Z]{2,5}) (\d{3,8}[A-Za-z0-9]{0,2})(?![A-Za-z0-9-])")
NOT_A_CODE = frozenset(
    """
    phase day days week weeks month months year cohort part arm group dose study type stage grade
    level visit cycle tab cap
    """.split()  # noqa: SIM905
)
# `comparisons.DOSE` skips a unit followed by "/" ("500 mg/500 mg"); a name needs the dose gone.
LOOSE_DOSE = re.compile(
    r"\d+(?:[.,]\d+)?(?:\s*[/-]\s*\d+(?:[.,]\d+)?)*\s*"
    r"(?:mg|mcg|µg|μg|ug|g|kg|ml|iu|units?|%|milligrams?|micrograms?|grams?)"
    r"(?:/(?:kg|m2|m²|ml|l|day|dose|h|hr))?(?![a-z])",
    re.IGNORECASE,
)
# A name that is one code, such as K-877 or 5FU: it has a letter and a digit and no spaces.
CODE_NAME = re.compile(r"(?=.*[A-Za-z])(?=.*\d)[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*")
SPLIT_NAME = re.compile(r"[()\[\],;+/&，、；]|\b(?:plus|and|or|with)\b", re.IGNORECASE)
NAME_TYPE_PREFIX = re.compile(r"^[A-Za-z ]+:\s*(?=\S)")  # "Drug: Zorbitrex" -> "Zorbitrex"
CONTROL_NAME = re.compile(
    r"\b(saline|dummy|no intervention|soc|bsc|treatment as usual)\b", re.IGNORECASE
)
BRACKETED = re.compile(r"\([^()]*\)|\[[^\[\]]*\]")
SCHEDULE_TOKEN = re.compile(r"q\d*[wmdh]|\d+")  # "Q3W", "2", "12"
MIN_ALIAS = 3
# More salts and dosage forms than `history` drops: "Asenapine maleate transdermal patch" is
# asenapine.
MORE_SALTS_AND_FORMS = frozenset(
    """
    maleate mesylate besylate tartrate succinate fumarate citrate hydrobromide hydrate bromide
    chloride tosylate furoate propionate xinafoate transdermal ophthalmic oph susp nasal topical intravenous subcutaneous
    sc inhalation powder sterile cap patch lotion foam drops suppository syrup emulsion paste
    implant lozenge granules
    """.split()  # noqa: SIM905
)
# Words that say how or when a drug is given, or are filler: not part of which drug it is.
GENERIC_WORDS = frozenset(
    """
    drug biological device procedure radiation behavioral genetic other supplement dietary
    combination product diagnostic test treatment therapy group arm study active experimental
    comparator doses best normal usual open label approximately form dosage in parenteral
    once twice daily weekly monthly every week weeks day days month months year years hour hours
    low high dose per mg iv oral extended release cohort bid qd tid qid tablets tablet capsule
    capsules injection infusion as a an at by on then part blinded blind double single ole
    extension period main optional switched
    """.split()  # noqa: SIM905
)
SALT_AND_FORM_WORDS = HISTORY_WORDS | MORE_SALTS_AND_FORMS
IGNORED_TOKENS = SALT_AND_FORM_WORDS | PLACEBO_FILLER | GENERIC_WORDS


def without_dose(text: str) -> str:
    """`text` with every dose ("10 mg", "500 mg/500 mg", "10 mg/kg", "2%") taken out."""
    return LOOSE_DOSE.sub(" ", DOSE.sub(" ", text))


def strip_control(text: str) -> str:
    """`text` with placebo, sham, vehicle, saline and the control words of `comparisons` taken out."""
    return CONTROL_NAME.sub(" ", CONTROL_WORDS.sub(" ", PLACEBO_WORDS.sub(" ", text)))


def has_control_word(text: str) -> bool:
    return strip_control(text) != text


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def company_codes(text: str) -> set[tuple[str, str]]:
    """The company codes in `text` as (letters, digits and tail), lower case."""
    found = set()
    for pattern in (HYPHENATED_CODE, SPACED_CODE):
        for prefix, rest in pattern.findall(text):
            if prefix.lower() not in NOT_A_CODE:
                found.add((prefix.lower(), rest.lower()))
    return found


def code_forms(code: tuple[str, str]) -> set[str]:
    """One code written with a hyphen, with a space and with neither."""
    prefix, rest = code
    return {f"{prefix}-{rest}", f"{prefix} {rest}", f"{prefix}{rest}"}


def _without_salts(piece: str) -> str:
    """`piece` without salt, dosage-form, schedule and filler words and bare numbers.

    The digits of a company code stay ("BI 409306").
    """
    protected = {rest for _, rest in company_codes(piece)}
    keep = [
        w
        for w in piece.split()
        if (word := w.lower().strip(".,:")) in protected
        or (word not in IGNORED_TOKENS and not SCHEDULE_TOKEN.fullmatch(word))
    ]
    return " ".join(keep)


def _has_word(text: str) -> bool:
    """True if `text` holds something that names a drug: a real word or a company code."""
    words = {t for t in _tokens(text) if len(t) >= MIN_ALIAS and not t.isdigit()}
    words -= {t for t in words if SCHEDULE_TOKEN.fullmatch(t)}
    return (
        bool(words - IGNORED_TOKENS)
        or CODE_NAME.fullmatch(text.strip()) is not None
        or bool(company_codes(text))
    )


def _usable(alias: str) -> bool:
    """An alias needs a real word in it: not a bare dose, a salt, a form, a schedule or a control word."""
    return _has_word(alias) and not has_control_word(alias)


def core_text(name: str) -> str:
    """The drug in an intervention name, with everything that only describes it taken out.

    What follows the last colon ("Main cohort (Cohort A): Ponsegromab low dose"), without
    brackets, doses, control and placebo words, and salt, dosage-form, schedule and filler words.
    """
    text = without_dose(name)
    for label in (":", " - "):
        after = text.rsplit(label, 1)[-1]
        text = after if after.strip() else text
    return _without_salts(" ".join(strip_control(BRACKETED.sub(" ", text)).split()))


def is_control(name: str) -> bool:
    """True if nothing that names a drug is left once control and placebo words are taken out.

    "Placebo", "Standard of care", "Vehicle lotion" and "Placebo plus standard therapy" are
    controls. "Belimumab plus standard therapy" is not: the drug stays.
    """
    return not _has_word(core_text(name))


def main_word(name: str) -> str | None:
    """The one distinctive word of a drug's name ("centanafadine" in "Centanafadine SR"), if any.

    The letters of a company code do not count: "Dupilumab SAR231893" has one word, dupilumab.
    """
    text = core_text(name)
    code_tokens = {t for code in company_codes(text) for t in (*code, "".join(code))}
    words = [
        t for t in _tokens(text) if len(t) >= MIN_ALIAS and not t.isdigit() and t not in code_tokens
    ]
    return words[0] if len(words) == 1 else None


def is_combination(name: str) -> bool:
    """True if `name` joins two or more drugs ("A + B", "A/B/C", "A, B or C")."""
    clean = " ".join(strip_control(without_dose(name)).split())
    pieces = {
        _without_salts(" ".join(p.split()).strip(" .:-")).lower() for p in SPLIT_NAME.split(clean)
    }
    return len({p for p in pieces if _usable(p)}) >= 2


def name_claims(name: str, as_part: bool = False) -> tuple[set[str], set[str]]:
    """The texts that stand for the same intervention as `name`: (whole-name texts, part texts).

    Whole-name texts: the name as written, without its dose, without salt and schedule words, the
    drug's core text and its one distinctive word, and every company code in it. Part texts: each
    part of a combination or of a bracket, and each of those without salt and schedule words.
    Control words never become a text. For a combination ("A + B"), or with `as_part`, only the
    name as written and without its dose is a whole-name text: a code or word inside it also
    names a drug listed on its own, and must not take the name from it.
    """
    clean = " ".join(strip_control(without_dose(name)).split())
    whole = {" ".join(name.split()), clean}
    detail = {_without_salts(clean), core_text(name), main_word(name)} - {None}
    parts = set()
    for piece in SPLIT_NAME.split(clean):
        piece = " ".join(piece.split()).strip(" .:-")
        parts |= {piece, _without_salts(piece)}
    codes = {f for c in company_codes(without_dose(name)) for f in code_forms(c)}
    if as_part or is_combination(name):
        parts |= detail | codes
    else:
        whole |= detail | codes
    whole = {t for t in whole if _usable(t) or t in codes}
    return whole, {t for t in parts if _usable(t) or t in codes} - whole


def name_variants(name: str) -> set[str]:
    """Every text that stands for the same intervention as `name`, as the model would see it."""
    return set().union(*name_claims(name))


def intervention_claims(detail: dict) -> tuple[set[str], set[str]]:
    """`name_claims` over an intervention's name and each of its other names.

    The other names of a combination describe the combination, so they are part texts only.
    """
    whole, parts = name_claims(detail["name"])
    combination = is_combination(detail["name"])
    for other in detail["other_names"]:
        w, p = name_claims(other, as_part=combination)
        whole |= w
        parts |= p
    return whole, parts - whole


def intervention_aliases(detail: dict) -> set[str]:
    return set().union(*intervention_claims(detail))


def drug_key(name: str) -> str:
    """What makes two interventions one drug: their core text (see `core_text`)."""
    return " ".join(_tokens(core_text(name))) or name.lower()


def active_flags(details: list[dict]) -> list[bool]:
    """For each intervention, True if it is a drug to replace and False if it is a control to keep.

    A pure control (see `is_control`) is kept. A name with a control word and a drug in it is a
    drug, unless the drug is one listed on its own elsewhere ("Placebo for Zorbitrex" next to
    "Zorbitrex"): then it is that drug's control.
    """
    pure = [is_control(d["name"]) for d in details]
    mixed = [not p and has_control_word(d["name"]) for p, d in zip(pure, details, strict=True)]
    plain = set().union(
        *(
            intervention_aliases(d)
            for d, p, m in zip(details, pure, mixed, strict=True)
            if not p and not m
        )
    )
    return [
        not p and not (m and bool(intervention_aliases(d) & plain))
        for d, p, m in zip(details, pure, mixed, strict=True)
    ]


def _letter(index: int) -> str:
    return string.ascii_uppercase[index] if index < 26 else str(index + 1)


def placeholders(details: list[dict]) -> dict[str, str]:
    """A placeholder for each active intervention, by intervention name.

    One letter series per type, in the order the drugs are first listed: "Drug A", "Drug B",
    "Biological A". The same drug listed twice (two doses, the same name repeated) has one
    placeholder. A control gets none.
    """
    counts: dict[str, int] = {}
    by_drug: dict[str, str] = {}
    result = {}
    for detail, active in zip(details, active_flags(details), strict=True):
        if not active:
            continue
        key = drug_key(detail["name"])
        if key not in by_drug:
            label = TYPE_LABELS.get(detail["type"], OTHER_LABEL)
            by_drug[key] = f"{label} {_letter(counts.get(label, 0))}"
            counts[label] = counts.get(label, 0) + 1
        result[detail["name"]] = by_drug[key]
    return result


def alias_map(details: list[dict]) -> dict[str, str]:
    """Lower-case alias -> placeholder.

    A text claimed by two different drugs is left out: neither placeholder would be right. A
    text that is a whole name for one drug and a part of a combination for another goes to the
    drug it names whole.
    """
    names = placeholders(details)
    owners: dict[str, dict[str, set[str]]] = {}
    for detail in details:
        if detail["name"] not in names:
            continue
        whole, parts = intervention_claims(detail)
        for tier, aliases in (("whole", whole), ("part", parts)):
            for alias in aliases:
                owners.setdefault(alias.lower(), {"whole": set(), "part": set()})[tier].add(
                    names[detail["name"]]
                )
    found = {}
    for alias, tiers in owners.items():
        claimants = tiers["whole"] or tiers["part"]
        if len(claimants) == 1:
            found[alias] = next(iter(claimants))
    return found


def replacer(aliases: dict[str, str]):
    """A function that replaces every alias, whole tokens only, longest alias first."""
    if not aliases:
        return lambda text: text
    ordered = sorted(aliases, key=lambda a: (-len(a), a))
    body = "|".join(r"\s+".join(re.escape(w) for w in a.split()) for a in ordered)
    pattern = re.compile(rf"(?<!\w)(?:{body})(?!\w)", re.IGNORECASE)

    def replace(text: str | None) -> str | None:
        if text is None:
            return None
        return pattern.sub(lambda m: aliases[" ".join(m.group(0).lower().split())], text)

    return replace


def blind_record(record: dict) -> dict:
    """A copy of an `extract_record` record with names replaced and the start date dropped.

    Replacement is made on the full text. Cutting a long text to a length is done afterwards (see
    `llm_predictor.render_trial`), so a name is never half replaced at a cut.
    """
    details = record["intervention_details"]
    names = placeholders(details)
    replace = replacer(alias_map(details))

    def arm_intervention(listed: str) -> str:
        """An arm's "Drug: Name" entry as its placeholder, or the name with aliases replaced."""
        name = NAME_TYPE_PREFIX.sub("", listed)
        return names.get(name) or replace(name)

    listed = [
        f"{d['type'] or 'other'}: {names.get(d['name']) or replace(d['name'])}" for d in details
    ]
    return {
        **record,
        "start_date": None,
        "conditions": [replace(c) for c in record["conditions"]],
        "interventions": list(dict.fromkeys(listed)),
        "arms": [
            {
                **arm,
                "label": replace(arm["label"]),
                "description": replace(arm["description"]),
                "interventions": [arm_intervention(n) for n in arm["interventions"]],
            }
            for arm in record["arms"]
        ],
        "primary_outcomes": [
            {"measure": replace(o["measure"]), "time_frame": replace(o["time_frame"])}
            for o in record["primary_outcomes"]
        ],
    }


def leaked_tokens(blinded_text: str, record: dict) -> list[str]:
    """Words and company codes of the trial's own intervention names still in the blinded text.

    Counted: words of three letters or more in a name or other name (control words, doses,
    digits, salt, dosage-form, schedule and generic words such as "drug" are not), and each
    company code of two to five letters or one letter and three digits or more, written with a
    hyphen, a space or neither.
    """
    wanted: set[str] = set()
    codes: set[tuple[str, str]] = set()
    for detail in record["intervention_details"]:
        for name in [detail["name"], *detail["other_names"]]:
            wanted |= {
                t for t in _tokens(strip_control(without_dose(name)))
                if len(t) >= MIN_ALIAS and not t.isdigit()
            }  # fmt: skip
            codes |= company_codes(without_dose(name))
    wanted -= IGNORED_TOKENS
    leaked = wanted & set(_tokens(blinded_text))
    for prefix, rest in codes:
        pattern = rf"(?<![A-Za-z0-9]){re.escape(prefix)}[- ]?{re.escape(rest)}(?![A-Za-z0-9])"
        if re.search(pattern, blinded_text, re.IGNORECASE):
            leaked.add(f"{prefix}-{rest}")
    return sorted(leaked)


# ---------------------------------------------------------------- did the model name the drug?


# Words that say what kind of product something is, not which one. A guess made only of these
# ("monoclonal antibody", "an inactivated vaccine") names a class, not the tested drug.
CLASS_WORDS = frozenset(
    """
    monoclonal antibody antibodies mab inhibitor inhibitors agonist antagonist blocker vaccine
    vaccines cell cells car gene protein peptide recombinant human humanised humanized
    inactivated live attenuated autologous allogeneic anti small molecule oral topical injectable
    novel investigational
    """.split()  # noqa: SIM905
)


def distinctive(text: str) -> bool:
    """True if `text` has a word that could be a drug's name: not filler, not a class word."""
    return any(
        len(w) >= MIN_ALIAS and w not in IGNORED_TOKENS and w not in CLASS_WORDS
        for w in _tokens(text)
    )


def tested_aliases(record: dict) -> set[str]:
    """Every text that stands for the trial's tested intervention (the experimental arms').

    A trial with no experimental arm uses all its active interventions. An intervention that is
    also given in a control arm (a background drug both groups take) is not the tested one, unless
    that would leave nothing.
    """
    details = record["intervention_details"]

    def listed_in(arms: list[dict]) -> set[str]:
        return {NAME_TYPE_PREFIX.sub("", n) for arm in arms for n in arm["interventions"]}

    experimental = [arm for arm in record["arms"] if arm["type"] == "experimental"]
    control = [arm for arm in record["arms"] if arm["type"] != "experimental"]
    in_experimental = listed_in(experimental)
    chosen = [d for d in details if d["name"] in in_experimental]
    foreground = [d for d in chosen if d["name"] not in listed_in(control)]
    chosen = foreground or chosen
    if not chosen:
        chosen = [d for d, active in zip(details, active_flags(details), strict=True) if active]
    return {alias.lower() for d in chosen for alias in intervention_aliases(d)}


def _contains(text: str, phrase: str) -> bool:
    return f" {phrase} " in f" {text} "


def alias_phrases(record: dict) -> set[str]:
    """The tested intervention's aliases as plain word groups, leaving out any that only name a
    class ("monoclonal antibody") or are filler."""
    phrases = {" ".join(_tokens(alias)) for alias in tested_aliases(record)}
    return {p for p in phrases if p and distinctive(p)}


def text_names_drug(text: str | None, record: dict) -> bool:
    """True if any alias of the tested intervention is a whole word group of `text`."""
    words = " ".join(_tokens(text or ""))
    return any(_contains(words, phrase) for phrase in alias_phrases(record))


def guess_matches(guess: str | None, record: dict) -> bool:
    """True if the model's `drug_guess` names the trial's tested intervention.

    Case does not matter. It matches if any alias of the intervention (its name, its other names,
    the name without dose and salt, each part of a combination, each company code) is a whole word
    group of the guess, or the guess is a whole word group of an alias ("insulin" for "insulin
    glargine"). A guess made only of filler or class words ("monoclonal antibody") matches
    nothing, and neither does an alias that is only that. A drug given in a control arm too is not
    the tested one (see `tested_aliases`). An empty guess matches nothing.

    This only scores stored replies; no prompt is made from it. It still misses a synonym or
    brand that is not among the aliases (nirsevimab for MEDI8897): see `llm_report`.
    """
    guess_words = _tokens(guess or "")
    if not distinctive(" ".join(guess_words)):
        return False
    text = " ".join(guess_words)
    return any(_contains(text, p) or _contains(p, text) for p in alias_phrases(record))
