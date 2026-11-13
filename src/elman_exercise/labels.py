"""Label a trial: did a treatment arm beat control on its primary endpoint?

The label is read from the posted results by fixed rules, with no judgement
and no outside information:

1. Take the first primary outcome that has a treatment-versus-control
   comparison (see comparisons.py).
2. Use one control group: the placebo if the outcome has one, otherwise the
   first control listed. So a trial with both a placebo and an active
   comparator is labelled against placebo.
3. For each treatment arm, take its first-listed comparison with that
   control. The arm "beats control" if that comparison has p < 0.05. Later
   comparisons of the same arm are mostly sensitivity analyses, other
   timepoints or other populations, and are not used: counting any of them
   would give an arm several chances.
4. The headline label, `any_arm`, is whether at least one arm beats control.
   `all_arms` is the strict variant. `highest_dose` is the same question for
   the top dose alone, where the arms form a clean dose ladder.

What the label does not know, and so cannot check:

- The direction of the effect. The record does not say whether higher or
  lower is better, so a significant result in the wrong direction counts.
- The trial's own significance threshold. 0.05 is applied to every trial,
  though some pre-specify another alpha or adjust for several comparisons.
- Whether the endpoint measures clinical benefit. A biomarker counts.
- Whether the first-listed comparison is the pre-specified primary analysis.
  It usually is, but nothing in the record says so.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from elman_exercise.comparisons import (
    Comparison,
    classify_analyses,
    dose_of,
    dose_skeleton,
    first_eligible_outcome,
    highest_dose_arm,
    primary_outcomes,
)

ALPHA = 0.05
_P_VALUE = re.compile(r"^(<=|=<|>=|=>|[<>≤≥=]?)\s*(\d*\.?\d+)$")
_BOUNDS = {"<": "<", ">": ">", "<=": "≤", "=<": "≤", "≤": "≤", ">=": "≥", "=>": "≥", "≥": "≥"}


@dataclass(frozen=True)
class PValue:
    value: float
    bound: str  # "=" exact; "<", "≤", ">", "≥": the true value lies on that side of `value`


def parse_p_value(text: str | None) -> PValue | None:
    """Read a posted p-value. Returns None for anything that is not a number from 0 to 1."""
    match = _P_VALUE.match((text or "").strip())
    if not match:
        return None
    sign, number = match.groups()
    value = float(number)
    if not 0 <= value <= 1:
        return None
    bound = _BOUNDS.get(sign, "=")
    if value == 0 and bound == "=":
        bound = "<"  # a posted "0.000" is a small value rounded down, not zero
    return PValue(value, bound)


def is_significant(p: PValue | None, alpha: float = ALPHA) -> bool | None:
    """True or False where the posted value settles it, None where it does not.

    "0.049" and "<0.05" are significant; "0.05" and ">0.05" are not. "≤0.05"
    could be exactly 0.05, and "<0.2" could be either side, so both are None.
    """
    if p is None:
        return None
    if p.bound == "=":
        return p.value < alpha
    if p.bound == "<":
        return True if p.value <= alpha else None
    if p.bound == "≤":
        return True if p.value < alpha else None
    return False if p.value >= alpha else None  # ">" and "≥"


def _any(results: list[bool | None]) -> bool | None:
    """True if any is True; False if all are False; None if it hangs on an unknown."""
    if any(result is True for result in results):
        return True
    return None if any(result is None for result in results) else False


def _all(results: list[bool | None]) -> bool | None:
    """True if all are True; False if any is False; None if it hangs on an unknown."""
    if any(result is False for result in results):
        return False
    return None if any(result is None for result in results) else True


@dataclass(frozen=True)
class TrialLabel:
    any_arm: bool | None  # headline: at least one treatment arm beat control
    all_arms: bool | None  # strict: every treatment arm beat control
    highest_dose: bool | None  # the top dose beat control; None without a clean dose ladder
    has_dose_ladder: bool
    n_arms: int  # treatment arms compared with the control on the chosen outcome
    outcome_index: int  # which primary outcome was used (0 is the first listed)
    control_kind: str  # "placebo", "active", "no treatment" or "unspecified"
    recognised_by: str  # how the control was recognised


def one_control(comparisons: list[Comparison]) -> list[Comparison]:
    """Keep the comparisons against a single control group: the placebo if there is one."""
    placebo = [c for c in comparisons if c.control_kind == "placebo"]
    control_id = (placebo or comparisons)[0].control_id
    return [c for c in comparisons if c.control_id == control_id]


def _top_dose_arm(arms: dict[str, str], outcome_groups: dict[str, str]) -> str | None:
    """The highest-dose arm, unless a higher dose of the same drug went untested.

    If the outcome reports a higher dose that has no comparison with control,
    the top *compared* dose is not the trial's top dose, and there is no answer.
    """
    top = highest_dose_arm(arms)
    if top is None:
        return None
    top_value, unit = dose_of(arms[top])
    for group_id, title in outcome_groups.items():
        if group_id in arms or dose_skeleton(title) != dose_skeleton(arms[top]):
            continue
        dose = dose_of(title)
        if dose and dose[1] == unit and dose[0] > top_value:
            return None
    return top


def label_comparisons(comparisons: list[Comparison], outcome_groups: dict[str, str]) -> TrialLabel:
    """Label a trial from its comparisons on one primary outcome.

    `outcome_groups` maps every group id reported on that outcome to its title.
    """
    comparisons = one_control(comparisons)
    arm_beats_control: dict[str, bool | None] = {}
    titles: dict[str, str] = {}
    for comparison in comparisons:
        if comparison.treatment_id in arm_beats_control:
            continue  # only an arm's first-listed comparison counts
        p = parse_p_value(comparison.analysis.get("pValue"))
        arm_beats_control[comparison.treatment_id] = is_significant(p)
        titles[comparison.treatment_id] = comparison.treatment_title

    top_dose = _top_dose_arm(titles, outcome_groups)
    return TrialLabel(
        any_arm=_any(list(arm_beats_control.values())),
        all_arms=_all(list(arm_beats_control.values())),
        highest_dose=arm_beats_control[top_dose] if top_dose else None,
        has_dose_ladder=top_dose is not None,
        n_arms=len(arm_beats_control),
        outcome_index=comparisons[0].outcome_index,
        control_kind=comparisons[0].control_kind,
        recognised_by=comparisons[0].recognised_by,
    )


def label_trial(study: dict) -> TrialLabel | None:
    """Label a trial, or None if it has no treatment-versus-control comparison.

    A returned label can still have `any_arm` of None: the comparison exists
    but its p-value does not settle the question (text, out of range, or a
    bound that straddles 0.05). Such trials are counted and left out.
    """
    comparisons, _ = classify_analyses(study)
    chosen = first_eligible_outcome(comparisons)
    if not chosen:
        return None
    outcome = primary_outcomes(study)[chosen[0].outcome_index]
    groups = {group["id"]: group.get("title", "") for group in outcome.get("groups", [])}
    return label_comparisons(chosen, groups)
