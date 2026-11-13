"""Models that add the precedent flag to the registration columns.

"Precedent": before the trial started, another molecule on the same target had already started a
Phase 3 trial in the same disease (`trial_benchmark.precedent`; a Phase 3 is a record whose phase
list is exactly PHASE3, and a withdrawn record never counts). Six models, each the registration
logistic plus one yes/no column, so the gain over "logistic: registration" is the flag's alone:

- "logistic: registration + precedent (today's links)" and "(2020 links)": the main flag. The
  links from a drug to its target genes are Open Targets 26.09's or 20.06's.
- "(2020 names and links)": the trial-to-drug name match is dated to June 2020 as well.
- "... experimental arms only": the stricter flag, experimental-arm drugs on both sides.

`sensitivity_specs()` gives three more, not in `SPECS`: the strict flag with a combined Phase 2/3
record also counted as a Phase 3 precedent. They exist so the report can show how much the
choice matters.

The flag table (`data/interim/precedent.parquet`, build it with `python -m trial_benchmark.precedent`)
needs no outcome. It depends on a release published after the trials started, so its dating is
"after the start" and these specs are not marked honest (`precedent.FEATURES`).
"""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import polars as pl

from trial_benchmark import baselines as bl
from trial_benchmark import dating, precedent
from trial_benchmark.models.registry import ModelSpec, Predictor, fit_pipeline

LINKS_TEXT = {
    "today": "today's links",
    "2020": "2020 links",
    "2020_names": "2020 names and links",
}


def fit_with_flag(column: str, train: pl.DataFrame) -> Predictor:
    """The registration logistic with `column` (yes/no) added."""
    base = bl.FEATURES[bl.REGISTRATION]
    return fit_pipeline(train, replace(base, boolean=base.boolean + (column,)))


def spec_for(column: str, strict: bool, version: str, combined: bool = False) -> ModelSpec:
    suffix = ", experimental arms only" if strict else ""
    suffix += ", with Phase 2/3" if combined else ""
    return ModelSpec(
        f"logistic: registration + precedent ({LINKS_TEXT[version]}{suffix})",
        "precedent",
        "The registration logistic plus one yes/no column: an earlier Phase 3 trial of another "
        f"molecule on the same target in the same disease ({LINKS_TEXT[version]}"
        f"{'; experimental-arm drugs on both sides' if strict else ''}"
        f"{'; a combined Phase 2/3 record also counts' if combined else ''}).",
        partial(fit_with_flag, column),
        tables=(precedent.TABLE,),
        honest=precedent.FEATURES[column] in dating.HONEST,
    )


SPECS = [
    spec_for(f"precedent_{'exp_' if strict else ''}{version}", strict, version)
    for strict in (False, True)
    for version in ("today", "2020", "2020_names")
]


def sensitivity_specs() -> list[ModelSpec]:
    """The strict models with a combined Phase 2/3 record also counted as a Phase 3 precedent."""
    return [
        spec_for(f"precedent_exp_{v}_p23", True, v, combined=True)
        for v in ("today", "2020", "2020_names")
    ]
