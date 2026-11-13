"""The deliberately wrong models from baselines.py, as specs.

They exist to show what a leak is worth, and that the benchmark would notice one.
They are marked not honest, and are listed apart from the honest models in reports.

- leaky: registration plus fields edited after the trial ended (enrollment, status, duration...).
- results structure: registration plus the number of arms and the kind of control,
  both taken from the posted results.
"""

from __future__ import annotations

from trial_benchmark import baselines as bl
from trial_benchmark.models.registry import BaselinePredictor, ModelSpec


def _fit(name: str):
    return lambda train: BaselinePredictor(bl.fit_logistic(train, bl.FEATURES[name]))


SPECS = [
    ModelSpec(
        "leaky: registration + post-hoc fields",
        "leaky",
        "The registration logistic plus fields that are edited after the trial ends.",
        _fit(bl.LEAKY),
        honest=False,
    ),
    ModelSpec(
        "leaky: registration + results structure",
        "leaky",
        "The registration logistic plus arm count and control kind, read from posted results.",
        _fit(bl.RESULTS_STRUCTURE),
        honest=False,
    ),
]
