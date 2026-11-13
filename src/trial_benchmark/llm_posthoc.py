"""Phrases that sponsors write into a trial record after the results are in.

The protocol record is the current version, and a sponsor can edit it when posting results. An
outcome's time frame then sometimes says what happened, not what was planned: "through database
cut-off date of 26 Oct 2017", "the median duration of follow-up was 21.8 months", "up to early
termination of the study". Such a phrase tells a reader about the trial's course, so it is not
something a forecaster could know at the start.

This module finds those phrases with plain rules (`PATTERNS`, `TIME_FRAME_PATTERNS`), flags a trial
that has one, and cuts the clause that carries it out of a piece of text. The rules are for
sensitivity: the report scores the trials with no flag separately, and the "blinded" prompt drops the
flagged clauses. A rule cannot find every rewrite, so a trial with no flag may still have one.
"""

from __future__ import annotations

import re

CASE = re.IGNORECASE
PATTERNS = [
    re.compile(p, CASE)
    for p in (
        r"\b(data|database|analysis)\s+cut-?\s?off\b|\bcut-?\s?off\s+date\b|\bdatabase\s+lock\b",
        r"\bmedian\s+(duration|time|follow-?\s?up|treatment)\b[^.;]{0,60}\b(was|were)\b",
        r"\b(maximum|minimum|actual|achieved)\s+(duration|follow-?\s?up)\b[^.;]{0,30}\b(was|were)\b",
        r"\bearly\s+terminat\w*|\bterminat\w*\s+(of|by)\s+the\s+(study|sponsor|trial)\b",
        r"\b(study|trial)\s+(was|were)\s+(terminated|stopped|halted|discontinued)\b",
        r"\bdid\s+not\s+meet\b|\bfailed\s+to\s+meet\b|\bresults?\s+(showed|demonstrated)\b",
    )
]
# A calendar year in a time frame is a date that was only known once it had passed. It is not a
# year when followed by a unit ("2000 mg", "1000 days").
YEAR = re.compile(
    r"(?<![\d.,])(?:19[89]\d|20[0-3]\d)(?![\d%.]|\s*(?:mg|mcg|ml|ng|g|kg|days?|weeks?|months?|"
    r"years?|hours?|patients|participants|subjects|cells|copies|iu|units?)\b)",
    CASE,
)
TIME_FRAME_PATTERNS = [*PATTERNS, YEAR]
CLAUSE_END = re.compile(r"(?<=[;,.])\s+")
BRACKETED = re.compile(r"\s*\([^()]*\)")


def has_phrase(text: str | None, time_frame: bool = False) -> bool:
    """True if `text` holds a phrase of the rules; a time frame also counts any calendar year."""
    patterns = TIME_FRAME_PATTERNS if time_frame else PATTERNS
    return bool(text) and any(p.search(text) for p in patterns)


def strip_clauses(text: str | None, time_frame: bool = False) -> str | None:
    """`text` without the bracket or clause that carries a phrase; the rest is kept.

    A clause ends at a semicolon, comma or full stop followed by a space. A text with no phrase is
    returned as it is; nothing left after the cut gives None.
    """
    if not has_phrase(text, time_frame):
        return text  # untouched, so punctuation is not rewritten
    text = BRACKETED.sub(lambda m: "" if has_phrase(m.group(0), time_frame) else m.group(0), text)
    clauses = [c for c in CLAUSE_END.split(text) if not has_phrase(c, time_frame)]
    kept = " ".join(clauses).strip(" ;,.")
    return kept or None


def flagged(record: dict) -> bool:
    """True if any outcome measure, time frame, arm label or arm description has a phrase."""
    outcomes = record["primary_outcomes"]
    texts = [(a["label"], False) for a in record["arms"]]
    texts += [(a["description"], False) for a in record["arms"]]
    texts += [(o["measure"], False) for o in outcomes] + [(o["time_frame"], True) for o in outcomes]
    return any(has_phrase(text, time_frame) for text, time_frame in texts)


def strip_record(record: dict) -> dict:
    """A copy of `record` with the flagged clauses taken out of arm and outcome text."""
    return {
        **record,
        "arms": [
            {
                **a,
                "label": strip_clauses(a["label"]),
                "description": strip_clauses(a["description"]),
            }
            for a in record["arms"]
        ],
        "primary_outcomes": [
            {
                "measure": strip_clauses(o["measure"]),
                "time_frame": strip_clauses(o["time_frame"], time_frame=True),
            }
            for o in record["primary_outcomes"]
        ],
    }
