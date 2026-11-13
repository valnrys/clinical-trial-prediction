"""Find the treatment-versus-control comparisons on a trial's primary outcomes.

A posted p-value is only evidence that "the treatment worked" if it compares a
treatment arm with a control arm. Many do not: they test one group against its
own baseline, several arms at once, two doses against each other, an
established reference drug against placebo, or non-inferiority. This module
keeps the comparisons that qualify and says which group is the control, using
only the trial's own record and fixed rules.

The record does not label the groups of an analysis as treatment or control,
so the control is recognised from three signals, tried in this order:

1. Exactly one group is a placebo, sham or vehicle arm, by its title.
2. The registered arm types mark exactly one group as a comparator, or one as
   a placebo arm and the other as an active comparator.
3. Exactly one group's title uses a phrase such as "control" or "standard of care".

The rules prefer to leave a trial out than to guess. No recognisable control
means the comparison is dropped: head-to-head trials where the record does not
say which drug is the new one, and comparisons of two doses or schedules of
the same drug.

Left out on purpose: inferring an "add-on" design when one arm's listed
interventions contain the other's. On a read of pre-2021 trials it was wrong
about as often as it was right, because a longer course of the same drug, or a
drug with and without a booster, look the same in the record.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass

NON_INFERIORITY_TYPES = {
    "NON_INFERIORITY",
    "EQUIVALENCE",
    "NON_INFERIORITY_OR_EQUIVALENCE",
    "NON_INFERIORITY_OR_EQUIVALENCE_LEGACY",
}
COMPARATOR_ARM_TYPES = {
    "PLACEBO_COMPARATOR": "placebo",
    "SHAM_COMPARATOR": "placebo",
    "NO_INTERVENTION": "no treatment",
    "ACTIVE_COMPARATOR": "active",
}
PLACEBO_WORDS = re.compile(r"\b(placebos?|sham|vehicle)\b", re.IGNORECASE)
# "Drug A + placebo" can be a treatment arm in a double-dummy design.
COMBINED_WITH = re.compile(r"\+|&|/|,|\b(plus|and|with)\b", re.IGNORECASE)
# Words that can sit beside "placebo" in the title of an arm that is nothing but placebo.
PLACEBO_FILLER = frozenset(
    """
    placebo placebos sham vehicle saline pbo matching matched match
    to for of the a and plus with arm group control comparator only dose pooled
    capsule capsules tablet tablets pill pills cream gel ointment solution spray patch
    injection injections infusion oral iv sc im topical inhaled inhaler
    once twice daily weekly bid qd tid qid
    """.split()  # noqa: SIM905
)
# "Arm B: Placebo + MTX" and "2) Placebo": a label in front of the title proper.
ARM_LABEL = re.compile(
    r"^\s*(?:(?:arm|group|cohort|part)\s*\w{1,3}\s*[:)\-]|\w{1,2}[:)])\s*", re.IGNORECASE
)
CONTROL_WORDS = re.compile(
    r"\b(control|comparator|standard (of )?care|standard therapy|standard treatment|"
    r"usual care|no treatment|observation|surveillance|watchful waiting|supportive care|"
    r"untreated)\b",
    re.IGNORECASE,
)
_NUMBER = r"(?<![\d.,])(\d{1,3}(?:,\d{3})+|\d+(?:[.,]\d+)?)"
# A unit followed by "/" ("mg/day", "IU/kg") is a different unit, so it is not matched.
_UNITS = r"mg/kg|mcg/kg|µg/kg|μg/kg|ug/kg|mg/m2|mg/m\^2|mg|mcg|µg|μg|ug|iu|units?|ml|g|ng"
DOSE = re.compile(rf"{_NUMBER}\s*((?:{_UNITS})\b(?![/²^])|%)", re.IGNORECASE)
# "10-20 mg", "10 to 20 mg" and "2 x 10 mg" do not state one dose.
DOSE_RANGE = re.compile(r"\d\s*(?:-|–|to|x|×)\s*\d", re.IGNORECASE)
UNIT_SPELLINGS = {
    "units": "unit",
    "mg/m^2": "mg/m2",
    **dict.fromkeys(["µg", "μg", "ug"], "mcg"),
    **dict.fromkeys(["µg/kg", "μg/kg", "ug/kg"], "mcg/kg"),
}

# Why an analysis with a p-value is not a treatment-versus-control comparison.
SINGLE_GROUP = "single group"
MANY_GROUPS = "three or more groups"
NON_INFERIORITY = "non-inferiority or equivalence"
NO_CONTROL = "no recognisable control"
REFERENCE_ARM = "reference drug versus control"


@dataclass(frozen=True)
class Comparison:
    """One two-group analysis with a p-value in which exactly one group is a control."""

    outcome_index: int  # position among the trial's primary outcomes
    control_id: str
    treatment_id: str
    treatment_title: str
    control_kind: str  # "placebo", "active", "no treatment" or "unspecified"
    recognised_by: str  # "placebo title", "arm type" or "control word"
    analysis: dict  # the raw analysis record


def _key(text: str | None) -> str:
    """Letters and digits only, so "Drug-20mg" and "Drug 20 mg" are the same arm."""
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _words(text: str | None) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", (text or "").lower()))


def _arms_by_title(arms: list[dict]) -> dict[str, dict]:
    """Registered arms keyed by title. Arms whose titles collide are left out."""
    counts = Counter(_key(arm.get("label")) for arm in arms)
    return {
        _key(arm.get("label")): arm
        for arm in arms
        if _key(arm.get("label")) and counts[_key(arm.get("label"))] == 1
    }


def _content_words(title: str) -> set[str]:
    """The words of a title that say what the arm received: no label, dose or filler."""
    title = DOSE.sub(" ", ARM_LABEL.sub("", title or ""))
    return {word for word in _words(title) if not word.isdigit()} - PLACEBO_FILLER


def _placebo_control(titles: tuple[str, str]) -> int | None:
    """Index of the placebo control by title, or None if the titles do not settle it."""
    has_placebo = [bool(PLACEBO_WORDS.search(title or "")) for title in titles]
    if not any(has_placebo):
        return None
    if all(has_placebo):
        # Both arms carry a placebo. Only an arm that is nothing but placebo is the control.
        pure = [not _content_words(title) for title in titles]
        return pure.index(True) if sum(pure) == 1 else None
    index = has_placebo.index(True)
    title, other = titles[index], titles[1 - index]
    if not COMBINED_WITH.search(ARM_LABEL.sub("", title)):
        return index
    # "Placebo + methotrexate" against "Drug X + methotrexate" is a control: what
    # it is combined with is in the other arm too. "Drug Y + placebo for X"
    # against "Drug X" is a double-dummy head-to-head, and is not.
    return index if _content_words(title) <= _words(other) else None


def find_control(titles: tuple[str, str], arms: list[dict]) -> tuple[int, str, str] | None:
    """Say which of two groups is the control.

    `titles` are the two group titles from the results section; `arms` are the
    registered arm groups from the protocol section. Returns (index of the
    control group, kind of control, how it was recognised), or None.
    """
    # 1. Placebo, sham or vehicle in the title.
    placebo = _placebo_control(titles)
    if placebo is not None:
        return placebo, "placebo", "placebo title"

    # 2. Registered arm types.
    by_title = _arms_by_title(arms)
    matched = [by_title.get(_key(title)) for title in titles]
    kinds = [COMPARATOR_ARM_TYPES.get(arm.get("type")) if arm else None for arm in matched]
    if all(kinds):
        # A placebo arm against an "active comparator" arm: the placebo is the control.
        is_placebo = [kind == "placebo" for kind in kinds]
        if sum(is_placebo) == 1:
            return is_placebo.index(True), "placebo", "arm type"
        return None  # two comparators: nothing marks either as the treatment
    if any(kinds):
        index = 0 if kinds[0] else 1
        other, other_title = matched[1 - index], titles[1 - index] or ""
        if other is not None:
            # The other group must be an experimental arm; "OTHER" leaves the roles unclear.
            if other.get("type") == "EXPERIMENTAL":
                return index, kinds[index], "arm type"
            return None
        # The other title matches no registered arm (often a pooled or renamed
        # group). Accept it as the treatment only if it cannot be a second
        # comparator: the trial registers just one, and the title does not read as a control.
        comparators = sum(arm.get("type") in COMPARATOR_ARM_TYPES for arm in arms)
        if comparators == 1 and not CONTROL_WORDS.search(other_title):
            return index, kinds[index], "arm type"
        return None

    # 3. A control phrase in exactly one title. What kind of control is not known.
    # Not tried when a placebo is involved but the titles did not settle which
    # arm is the control: "Placebo + SOC" against "Drug X plus standard of care"
    # would otherwise pick the drug arm.
    if any(PLACEBO_WORDS.search(title or "") for title in titles):
        return None
    has_phrase = [bool(CONTROL_WORDS.search(title or "")) for title in titles]
    if sum(has_phrase) == 1:
        return has_phrase.index(True), "unspecified", "control word"
    return None


def _is_reference_arm(title: str, arms: list[dict]) -> bool:
    """True if this "treatment" is registered as an active comparator.

    Placebo-controlled trials often carry an established drug as a reference.
    "Reference beats placebo" shows the trial could detect an effect, not that
    the new treatment works.
    """
    arm = _arms_by_title(arms).get(_key(title))
    return arm is not None and arm.get("type") == "ACTIVE_COMPARATOR"


def has_p_value(analysis: dict) -> bool:
    """True if the analysis posts anything as its p-value, usable or not.

    An unusable value such as "NS" still marks a test that was run. Keeping it
    means the arm stays in the trial and the label comes out undetermined,
    where dropping it would silently remove an arm that did not succeed.
    """
    return bool((analysis.get("pValue") or "").strip())


def primary_outcomes(study: dict) -> list[dict]:
    outcomes = (
        study.get("resultsSection", {}).get("outcomeMeasuresModule", {}).get("outcomeMeasures", [])
    )
    return [outcome for outcome in outcomes if outcome.get("type") == "PRIMARY"]


def classify_analyses(study: dict) -> tuple[list[Comparison], list[str]]:
    """Sort every primary-outcome analysis that has a p-value.

    Returns the treatment-versus-control comparisons, and for each analysis
    that is not one, the reason. Together they account for every such analysis.
    """
    arms = study.get("protocolSection", {}).get("armsInterventionsModule", {}).get("armGroups", [])
    comparisons: list[Comparison] = []
    rejected: list[str] = []
    for outcome_index, outcome in enumerate(primary_outcomes(study)):
        titles = {group["id"]: group.get("title", "") for group in outcome.get("groups", [])}
        for analysis in outcome.get("analyses", []):
            if not has_p_value(analysis):
                continue
            group_ids = list(dict.fromkeys(analysis.get("groupIds", [])))
            if len(group_ids) < 2:
                rejected.append(SINGLE_GROUP)
                continue
            if len(group_ids) > 2:
                rejected.append(MANY_GROUPS)
                continue
            if analysis.get("nonInferiorityType") in NON_INFERIORITY_TYPES:
                rejected.append(NON_INFERIORITY)
                continue
            if not all(titles.get(group_id) for group_id in group_ids):
                rejected.append(NO_CONTROL)
                continue
            pair = (titles[group_ids[0]], titles[group_ids[1]])
            found = find_control(pair, arms)
            if found is None:
                rejected.append(NO_CONTROL)
                continue
            control, kind, how = found
            comparisons.append(
                Comparison(
                    outcome_index=outcome_index,
                    control_id=group_ids[control],
                    treatment_id=group_ids[1 - control],
                    treatment_title=pair[1 - control],
                    control_kind=kind,
                    recognised_by=how,
                    analysis=analysis,
                )
            )
    # Drop reference drugs, but only when another treatment arm remains. If
    # every "treatment" is typed as an active comparator, the registry has
    # mistyped the investigational arm, which is common, and it is kept.
    is_reference = [_is_reference_arm(c.treatment_title, arms) for c in comparisons]
    if any(is_reference) and not all(is_reference):
        rejected += [REFERENCE_ARM] * sum(is_reference)
        comparisons = [c for c, ref in zip(comparisons, is_reference, strict=True) if not ref]
    return comparisons, rejected


def first_eligible_outcome(comparisons: list[Comparison]) -> list[Comparison]:
    """The comparisons on the first primary outcome that has any."""
    if not comparisons:
        return []
    first = min(comparison.outcome_index for comparison in comparisons)
    return [comparison for comparison in comparisons if comparison.outcome_index == first]


def dose_of(title: str) -> tuple[float, str] | None:
    """The dose in an arm title, if it states exactly one."""
    title = title or ""
    found = list(DOSE.finditer(title))
    if len(found) != 1 or DOSE_RANGE.search(title):
        return None
    number, unit = found[0].group(1), found[0].group(2).lower()
    if re.fullmatch(r"\d{1,3}(,\d{3})+", number):
        number = number.replace(",", "")  # 1,000 is a thousand
    return float(number.replace(",", ".")), UNIT_SPELLINGS.get(unit, unit)


def dose_skeleton(title: str) -> str:
    """An arm title with its dose replaced by a marker.

    Two arms are rungs of one ladder only if their skeletons match. The marker
    keeps the dose's position, so "X 10 mg + Y" and "X + Y 20 mg" differ.
    """
    return _key(DOSE.sub(" dose ", title or ""))


def highest_dose_arm(treatment_titles: dict[str, str]) -> str | None:
    """Among several treatment arms, the id of the one with the highest dose.

    Returns None unless the arms form a clean ladder: every title states
    exactly one dose, all in the same unit, all different, and the titles are
    otherwise identical. That last condition rules out arms that are different
    drugs, or the same dose on different schedules ("10 mg twice daily" against
    "15 mg once daily"), where the larger number is not the larger dose.
    """
    if len(treatment_titles) < 2:
        return None
    doses = {group_id: dose_of(title) for group_id, title in treatment_titles.items()}
    if not all(doses.values()):
        return None
    values = [dose[0] for dose in doses.values()]
    units = {dose[1] for dose in doses.values()}
    rest = {dose_skeleton(title) for title in treatment_titles.values()}
    if len(units) != 1 or len(set(values)) != len(values) or len(rest) != 1:
        return None
    return max(doses, key=lambda group_id: doses[group_id][0])
