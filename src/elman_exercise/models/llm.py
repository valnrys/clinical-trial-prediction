"""A language model as predictor: Claude Haiku and Claude Sonnet, each with three prompts.

Six specs, one per model and prompt variant. The predictions are made in advance by
`elman_exercise.llm_predictor` (see its docstring for the prompt, the isolation of each call
and the sampling) and stored in `data/interim/llm_predictions.parquet`. A spec does not fit
anything: it looks up the stored probability for each trial.

Every spec is marked `pretrained=True`: a language model that has seen a trial's published
result is a leak that no date in the registration can rule out. The "identified" prompt shows the
registry id, title and sponsor name, which is what lets a model recognise a specific trial. The
"masked" prompt hides those three. The "blinded" prompt also replaces every intervention name with
a placeholder and drops the start date. The gaps between the three scores separate reasoning from
recall (see `llm_predictor` and `llm_blinding`).

Honesty is worked out as in the other families, from the datings of the fields the prompt shows
(`llm_predictor.PROMPT_FIELDS`). All of them come from the protocol record, so every variant is
honest in the benchmark's sense. That says the prompt holds nothing from after the start. It
cannot say what the model remembers: recall is not bounded, in any variant, and "honest" here
does not mean the model has not seen the answer.

Trials outside the sample
-------------------------
Only sampled development trials (`results/llm_sample.csv`) have a prediction a spec can return:
stored test and late predictions are dropped on load. For any other trial the predictor returns
one constant, the model's own mean probability over its stored predictions for the sampled
development trials, as the science-only models do for trials with no mapped target. Its score on
trials that are mostly outside the sample compares a constant with real predictions. So these
specs are meant to be scored only on the sampled trials, and not ranked against models scored on
all trials: the specs are marked `ranked=False`. `results/dev_llm.md` scores them, on the
sampled development trials.
They read the predictions table themselves (`own_tables`), so `validate` skips them, and says so,
when it has not been made.
"""

from __future__ import annotations

from functools import partial

import numpy as np
import polars as pl

from elman_exercise.dating import HONEST
from elman_exercise.llm_predictor import (
    MODELS,
    NOT_SHOWN,
    PREDICTIONS_PATH,
    PROMPT_FIELDS,
    VARIANTS,
    read_predictions,
    read_sample,
)
from elman_exercise.models.registry import ModelSpec

NOT_RANKED = (
    "predictions exist only for the sampled trials and every other trial gets one constant; "
    "scored on the sampled development trials in `results/dev_llm.md`"
)
SHOWN = {
    "identified": "the registry id, title and sponsor name",
    "masked": "no registry id, title or sponsor name (the sponsor class only)",
    "blinded": (
        "no registry id, title, sponsor name or start date, and every intervention name "
        "replaced by a placeholder"
    ),
}


def fields_read(variant: str) -> list[str]:
    """The prompt fields a variant shows."""
    return [f for f in PROMPT_FIELDS if f not in NOT_SHOWN[variant]]


def reads_honestly(variant: str) -> bool:
    return all(PROMPT_FIELDS[f] in HONEST for f in fields_read(variant))


class StoredProbabilities:
    """Looks up a stored probability by trial id; one constant for any trial with none."""

    def __init__(self, probabilities: dict[str, float], constant: float):
        if not probabilities:
            raise ValueError("No stored predictions for this model and variant.")
        self.probabilities = probabilities
        self.constant = constant

    def predict(self, frame: pl.DataFrame) -> np.ndarray:
        return np.array([self.probabilities.get(i, self.constant) for i in frame["nct_id"]])


def fit_stored(train: pl.DataFrame, model: str, variant: str) -> StoredProbabilities:
    """The stored first-run predictions of one model and variant. `train` is not used.

    Only sampled development trials are kept, and the constant for any other trial is their mean.
    Predictions for test and late trials exist, but they must not reach any score here.
    """
    table = read_predictions(PREDICTIONS_PATH).filter(
        (pl.col("model") == model)
        & (pl.col("variant") == variant)
        & (pl.col("run") == 0)
        & pl.col("probability").is_not_null()
    )
    if table["nct_id"].n_unique() != table.height:
        raise ValueError("The predictions table has a trial twice for one model, variant and run.")
    development = set(read_sample().filter(pl.col("split") == "dev")["nct_id"])
    # Only the sampled development trials: the harness is development-only, so a test or late
    # prediction (stored, because it is a feature) must never be one a spec can return.
    probabilities = {
        i: p for i, p in zip(table["nct_id"], table["probability"], strict=True) if i in development
    }
    if not probabilities:
        raise ValueError("No stored predictions for the sampled development trials.")
    constant = float(np.mean(list(probabilities.values())))
    return StoredProbabilities(probabilities, constant)


def llm_spec(model: str, variant: str) -> ModelSpec:
    return ModelSpec(
        f"llm: {model} ({variant})",
        "llm",
        f"Claude {model.title()} reads the protocol record with {SHOWN[variant]}, then gives a probability. "
        "Defined on the sampled trials only; the others get one constant. Recall is not bounded: "
        "the model may have read about the trial.",
        partial(fit_stored, model=model, variant=variant),
        honest=reads_honestly(variant),
        own_tables=(PREDICTIONS_PATH.stem,),
        pretrained=True,
        ranked=False,
        not_ranked_reason=NOT_RANKED,
    )


SPECS = [llm_spec(model, variant) for model in MODELS for variant in VARIANTS]
